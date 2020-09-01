"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
from collections import namedtuple
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import BahdanauMonotonicAttention
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper

from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope


#From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer, z):
	"""Computes the attention and alignments for a given attention_mechanism."""
	if z is not None:
		alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state, z=z)
	else:
		alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

	# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# Context is the inner product of alignments and values along the
	# memory time dimension.
	# alignments shape is
	#   [batch_size, 1, memory_time]
	# attention_mechanism.values shape is
	#   [batch_size, memory_time, memory_size]
	# the batched matmul is over memory_time, so the output shape is
	#   [batch_size, 1, memory_size].
	# we then squeeze out the singleton dim.
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state

class LocationSensitiveSoftAttention(BahdanauAttention):
	"""Implements Location Sensitive Attention from:
	Chorowski, Jan et al. 'Attention-Based Models for Speech Recognition'
	https://arxiv.org/abs/1506.07503
	"""

	def __init__(self,
			   num_units,
			   memory,
			   memory_sequence_length=None,
			   filters=32,
			   kernel_size=31,
			   smoothing=False,
			   normalize=False,
			   windowing=False,
			   left_window_width=20,
			   right_window_width=30,
			   sharpen=False,
			   sharpen_factor=2.0,
			   cumulate_weights=True,
			   name='LocationSensitiveSoftAttention'):
		"""Construct the Attention mechanism. See superclass for argument details.

		Args:
		  num_units: cell dim.
		  memory: encoder memory.
		  memory_sequence_length: memory length
		  filters: location conv filters
		  kernel_size: location conv kernel size
		  smoothing: if smoothing score
		  normalize: if normalize score
		  windowing: windowing score for long sequence
		  left_window_width: left window size
		  right_window_width: right window size
		  sharpen: if sharpen score
		  cumulate_weights: Whether to cumulate all previous attention weights
		"""
		normalization_function = _smoothing_normalization if smoothing else None
		super(LocationSensitiveSoftAttention, self).__init__(
			num_units,
			memory,
			memory_sequence_length=memory_sequence_length,
			probability_fn=normalization_function,
			name=name)
		self.location_conv = tf.layers.Conv1D(filters,
											  kernel_size,
											  padding='same',
											  use_bias=False,
											  name='location_conv')
		self.location_layer = tf.layers.Dense(num_units,
											  use_bias=False,
											  dtype=tf.float32,
											  name='location_layer')
		self._normalize = normalize
		self._windowing = windowing
		self._left_window_width = left_window_width
		self._right_window_width = right_window_width
		self._sharpen = sharpen
		self._sharpen_factor = sharpen_factor
		self._cumulate_weights = cumulate_weights

	def __call__(self, query, state, z=None):
		"""Score the query based on the keys and values.

        This replaces the superclass implementation in order to add in the location
        term.

        Args:
          query: Tensor of shape `[N, num_units]`.
          state: Tensor of shape `[N, T_in]`

        Returns:
          alignments: Tensor of shape `[N, T_in]`
        """
		with tf.variable_scope(None, 'location_sensitive_attention', [query]):
			expanded_alignments = tf.expand_dims(state, axis=2)  # [N, T_in, 1]
			f = self.location_conv(expanded_alignments)  # [N, T_in, 10]
			processed_location = self.location_layer(f)  # [N, T_in, num_units]

			processed_query = self.query_layer(
				query) if self.query_layer else query  # [N, num_units]
			processed_query = tf.expand_dims(processed_query,
											 axis=1)  # [N, 1, num_units]
			score = _location_sensitive_score(processed_query, processed_location,
											  self.keys, self._normalize)
			if self._sharpen:
				score *= self._sharpen_factor
			if self._windowing:
				cum_alignment = tf.cumsum(state, 1)
				half_step = cum_alignment > 0.5
				shifted_left = tf.pad(half_step[:, self._left_window_width + 1:],
									  [[0, 0], [0, self._left_window_width + 1]],
									  constant_values=True)
				shifted_right = tf.pad(half_step[:, :-self._right_window_width],
									   [[0, 0], [self._right_window_width, 0]],
									   constant_values=False)
				window = tf.logical_xor(shifted_left, shifted_right)
				# mask the score using the window
				score = tf.where(
					window, score,
					tf.ones_like(score) * tf.float32.as_numpy_dtype(-np.inf))

			alignments = self._probability_fn(score, state)
			if self._cumulate_weights:
				next_state = alignments + state
			else:
				next_state = alignments
			return alignments, next_state

def _location_sensitive_score(processed_query, processed_location, keys,
							  normalize=False):
	"""Location-sensitive attention score function.

	Based on _bahdanau_score from
	tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
	"""
	# Get the number of hidden units from the trailing dimension of keys
	dtype = processed_query.dtype
	num_units = keys.shape[2].value or tf.shape(keys)[2]
	v = tf.get_variable('attention_score_v', [num_units], dtype=dtype)
	b = tf.get_variable("attention_score_b", [num_units],
						dtype=dtype,
						initializer=tf.zeros_initializer())

	# Scalar used in weight normalization
	g = tf.get_variable("attention_score_g",
						dtype=dtype,
						initializer=math.sqrt((1. / num_units)))

	if normalize:
		normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
		score_bias = tf.get_variable("attention_score_r",
									 dtype=dtype,
									 initializer=-4.0)
		return tf.reduce_sum(
			normed_v * tf.tanh(keys + processed_query + processed_location + b),
			[2]) + score_bias
	else:
		return tf.reduce_sum(
			v * tf.tanh(keys + processed_query + processed_location + b), [2])


def _smoothing_normalization(e):
	"""Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)