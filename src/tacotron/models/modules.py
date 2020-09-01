from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tacotron.utils.util import shape_list


class ReferenceEncoder(object):

	def __init__(self,
		hparams,
		training,
		scope=None):
		self._average_pooling = hparams.ref_enc_average_pooling
		self.is_training = training
		self._scope = scope
		self._conv_layers = []
		for i, filters in enumerate(hparams.ref_enc_filters):
			self._conv_layers.append(
			  Conv2D(filters, hparams.ref_enc_conv_kernel, hparams.ref_enc_conv_stride, training, tf.nn.relu,
					 'conv2d_%d' % i))

		self._rnn_cell = GRUCell(hparams.ref_enc_rnn_units)

	def __call__(self, inputs):
		with tf.variable_scope(self._scope or 'reference_encoder'):
			ref_outputs = tf.expand_dims(inputs, axis=-1)

			for layer in self._conv_layers:
				ref_outputs = layer(ref_outputs)

			shapes = shape_list(ref_outputs)
			ref_outputs = tf.reshape(ref_outputs, shapes[:-2] + [shapes[-2] * shapes[-1]])

			# Single layer GRU
			outputs, states = tf.nn.dynamic_rnn(self._rnn_cell, ref_outputs, dtype=tf.float32)
			# average pooling or final states
			if self._average_pooling:
				pool_state = tf.reduce_mean(outputs, axis=1)
				return pool_state
			else:
				return states

class TacotronEncoder(object):
	"""Tacotron 2 Encoder Cell
	Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
	layer to predict the hidden representation vector (or memory)
	"""

	def __init__(self, convolutional_layers, lstm_layer):
		"""Initialize encoder parameters

		Args:
			convolutional_layers: Encoder convolutional block class
			lstm_layer: encoder bidirectional lstm layer class
		"""
		#Initialize encoder layers
		self._convolutions = convolutional_layers
		self._cell = lstm_layer

	def __call__(self, inputs, input_lengths=None):
		#Pass input sequence through a stack of convolutional layers
		conv_output = self._convolutions(inputs)

		#Extract hidden representation from encoder lstm cells
		hidden_representation = self._cell(conv_output, input_lengths)

		#For shape visualization
		self.conv_output_shape = conv_output.shape
		return hidden_representation


class HighwayNet:
	def __init__(self, units, name=None):
		self.units = units
		self.scope = 'HighwayNet' if name is None else name

		self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name='H')
		self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			H = self.H_layer(inputs)
			T = self.T_layer(inputs)
			return H * T + inputs * (1. - T)


class CBHG:
	def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, training, name=None):
		self.K = K

		self.pool_size = pool_size
		self.is_training = training
		self.scope = 'CBHG' if name is None else name

		self.highway_units = highway_units
		self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}'.format(self.scope, i+1)) for i in range(n_highwaynet_layers)]
		self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_forward_RNN'.format(self.scope))
		self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_backward_RNN'.format(self.scope))

		self._conv_bank_layers = []
		for i in range(K):
			self._conv_bank_layers.append(
				Conv1D(conv_channels,
					   i+1,
					   dropout_rate=0.,
					   training=training,
					   activation=tf.nn.relu,
					   scope='encoder_conv_%d' % i))
		self._proj1 = Conv1D(projections[0], projection_kernel_size, 0., training, tf.nn.relu, 'proj1')
		self._proj2 = Conv1D(projections[1], projection_kernel_size, 0., training, lambda _: _, 'proj2')


	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('conv_bank'):
				#Convolution bank: concatenate on the last axis to stack channels from all convolutions
				#The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
				#This makes one of the strengths of the CBHG block on sequences.
				conv_outputs = tf.concat([conv_layer(inputs) for i, conv_layer in enumerate(self._conv_bank_layers)],axis=-1)

			#Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
			maxpool_output = tf.layers.max_pooling1d(
				conv_outputs,
				pool_size=self.pool_size,
				strides=1,
				padding='same')

			#Two projection layers
			proj1_output = self._proj1(maxpool_output)
			proj2_output = self._proj2(proj1_output)

			#Residual connection
			highway_input = proj2_output + inputs

			#Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
			if highway_input.shape[2] != self.highway_units:
				highway_input = tf.layers.dense(highway_input, self.highway_units)

			#4-layer HighwayNet
			for highwaynet in self.highwaynet_layers:
				highway_input = highwaynet(highway_input)
			rnn_input = highway_input

			#Bidirectional RNN
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				rnn_input,
				sequence_length=input_lengths,
				dtype=tf.float32)
			return tf.concat(outputs, axis=2) #Concat forward and backward outputs


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
	'''Wrapper for tf LSTM to create Zoneout LSTM Cell

	inspired by:
	https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

	Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

	Many thanks to @Ondal90 for pointing this out. You sir are a hero!
	'''
	def __init__(self, num_units, training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
		'''Initializer with possibility to set different zoneout values for cell/hidden states.
		'''
		zm = min(zoneout_factor_output, zoneout_factor_cell)
		zs = max(zoneout_factor_output, zoneout_factor_cell)

		if zm < 0. or zs > 1.:
			raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

		self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
		self._zoneout_cell = zoneout_factor_cell
		self._zoneout_outputs = zoneout_factor_output
		self._training = training
		self.state_is_tuple = state_is_tuple

	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._cell.output_size

	def __call__(self, inputs, state, scope=None):
		'''Runs vanilla LSTM Cell and applies zoneout.
		'''
		#Apply vanilla LSTM
		output, new_state = self._cell(inputs, state, scope)

		if self.state_is_tuple:
			(prev_c, prev_h) = state
			(new_c, new_h) = new_state
		else:
			num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
			prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
			prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
			new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
			new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

		#Apply zoneout
		if self._training:
			#nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
			c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
			h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

		else:
			c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
			h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

		new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

		return output, new_state


class EncoderConvolutions(object):
	"""Encoder convolutional layers used to find local dependencies in inputs characters.
	"""
	def __init__(self, training, hparams, activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		self._training = training
		self._conv_layers = []
		for i in range(hparams.enc_conv_num_layers):
			self._conv_layers.append(
				Conv1D(hparams.enc_conv_filters,
					   hparams.enc_conv_kernel_size,
					   dropout_rate=hparams.tacotron_dropout_rate,
					   training=training,
					   activation=activation,
					   scope='encoder_conv_%d' % i))
		self.scope = 'enc_conv_layers' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i, conv_layer in enumerate(self._conv_layers):
				x = conv_layer(x)
		return x


class EncoderRNN(object):
	"""Encoder bidirectional one layer LSTM
	"""
	def __init__(self, training, size=256, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control zoneout
			size: integer, the number of LSTM units for each direction
			zoneout: the zoneout factor
			scope: EncoderRNN scope.
		"""
		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create forward LSTM Cell
		self._fw_cell = ZoneoutLSTMCell(size, training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_fw_LSTM')

		#Create backward LSTM Cell
		self._bw_cell = ZoneoutLSTMCell(size, training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_bw_LSTM')

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32,
				swap_memory=True)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs


class Prenet(object):
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, hparams, training, activation=tf.nn.relu, scope=None):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		self._drop_rate = hparams.tacotron_dropout_rate
		self._scope = 'prenet' if scope is None else scope
		self._training = training
		self._layers = []
		for i, size in enumerate(hparams.prenet_layers):
			self._layers.append(
				tf.layers.Dense(size,
					activation=activation,
					name='dense_%d' % (i + 1)))


	def __call__(self, inputs):
		x = inputs
		with tf.variable_scope(self._scope):
			for i, layer in enumerate(self._layers):
				x = layer(x)
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(x, rate=self._drop_rate, training=True,
					name='dropout_{}'.format(i + 1))
		return x


class DecoderRNN(object):
	"""Decoder two uni directional LSTM Cells
	"""
	def __init__(self, training, layers=2, size=1024, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control zoneout
			layers: integer, the number of LSTM layers in the decoder
			size: integer, the number of LSTM units in each layer
			zoneout: the zoneout factor
		"""
		self._training = training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, self._training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)


class FrameProjection(object):
	"""Projection layer to r * num_mels dimensions or num_mels dimensions
	"""
	def __init__(self, shape=80, activation=None, scope=None):
		"""
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
		self.shape = shape
		self.activation = activation

		self.scope = 'Linear_projection' if scope is None else scope
		self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			#If activation==None, this returns a simple Linear projection
			#else the projection will be passed through an activation function
			# output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
			# 	name='projection_{}'.format(self.scope))
			output = self.dense(inputs)

			return output


class StopProjection(object):
	"""Projection to a scalar and through a sigmoid activation
	"""
	def __init__(self, training, shape=1, activation=tf.nn.sigmoid, scope=None):
		"""
		Args:
			is_training: Boolean, to control the use of sigmoid function as it is useless to use it
				during training since it is integrate inside the sigmoid_crossentropy loss
			shape: integer, dimensionality of output space. Defaults to 1 (scalar)
			activation: callable, activation function. only used during inference
			scope: StopProjection scope.
		"""
		self.shape = shape
		self.activation = activation
		self.scope = 'stop_token_projection' if scope is None else scope
		self._training = training

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape,
				activation=None, name='projection_{}'.format(self.scope))

			#During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
			if self._training:
				return output
			return self.activation(output)


class Postnet(object):
	"""Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""
	def __init__(self, hparams, training, output_size, activation=tf.nn.tanh, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		self._scope = 'postnet_convolutions' if scope is None else scope
		self._training = training
		self._projection = tf.layers.Dense(output_size)

		self._conv_layers = []
		for i in range(hparams.postnet_conv_layers):
			self._conv_layers.append(
				Conv1D(hparams.postnet_conv_firters,
					   hparams.postnet_conv_kernel_size,
					   dropout_rate=hparams.tacotron_dropout_rate,
					   training=training,
					   activation=activation if i < hparams.postnet_conv_layers - 1 else None,
					   scope='postnet_conv_%d' % i))


	def __call__(self, inputs):
		with tf.variable_scope(self._scope):
			x = inputs
			for i, conv_layer in enumerate(self._conv_layers):
				x = conv_layer(x)
			x = self._projection(x)
		return x

class Conv1D(object):

	def __init__(self,
		filters,
		kernel_size,
		dropout_rate,
		training,
		activation=None,
		scope=None):
		self._scope = scope
		self._dropout_rate = dropout_rate
		self._activation = activation
		self._conv = tf.layers.Conv1D(filters, kernel_size, padding='same')
		self._bn = tf.layers.BatchNormalization()
		self._training = training

	def __call__(self, x):
		with tf.variable_scope(self._scope or 'conv1d'):
			x = self._bn(self._conv(x), training=self._training)
			if self._activation is not None:
				x = self._activation(x)
			return tf.layers.dropout(x, self._dropout_rate, training=self._training, name='dropout')


class Conv2D(object):
	def __init__(self, filters, kernel_size, strides, training, activation=None,
               scope=None):
		self._scope = scope
		self._activation = activation
		self._training = training
		self._conv = tf.layers.Conv2D(filters, kernel_size, strides, padding='same')
		self._bn = tf.layers.BatchNormalization()

	def __call__(self, x):
		with tf.variable_scope(self._scope or 'conv2d'):
			x = self._bn(self._conv(x), training=self._training)
			if self._activation is not None:
				x = self._activation(x)
			return x



def _round_up_tf(x, multiple):
	# Tf version of remainder = x % multiple
	remainder = tf.mod(x, multiple)
	# Tf version of return x if remainder == 0 else x + multiple - remainder
	x_round =  tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
		lambda: x,
		lambda: x + multiple - remainder)

	return x_round

def sequence_mask(lengths, r, expand=True):
	'''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
	'''
	max_len = tf.reduce_max(lengths)
	max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
	if expand:
		return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked Mean Squared Error
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(mels)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)

def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked SigmoidCrossEntropy with logits
	'''

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[1., 0., 0., 0., 0.],
	#							    [1., 1., 1., 0., 0.],
	#							    [1., 1., 0., 0., 0.]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		#Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.cross_entropy_pos_weight to 1
		#will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
		losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs, pos_weight=hparams.cross_entropy_pos_weight)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def MaskedLinearLoss(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked MAE loss with priority to low frequencies
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(freq)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	l1 = tf.abs(targets - outputs)
	n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		masked_l1 = l1 * mask_
		masked_l1_low = masked_l1[:,:,0:n_priority_freq]

	mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
	mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)

	return 0.5 * mean_l1 + 0.5 * mean_l1_low
class Speaker_Classifier(object):

	def __init__(self,
		hparams,
		training,
		scope=None):
		self._average_pooling = hparams.cls_enc_average_pooling
		self.hp=hparams
		self.is_training = training
		self._scope = scope
		self._layers = []
		for i, filters in enumerate(hparams.cls_mid_dense):
			self._layers.append(
				tf.layers.Dense(filters, activation=tf.nn.relu))
		self._layers.append(tf.layers.Dense(hparams.speaker_num,activation=tf.nn.softmax))



	def __call__(self, inputs):
		with tf.variable_scope(self._scope or 'speaker_classifier'):

			#ref_outputs = tf.expand_dims(inputs, axis=-1)
			ref_outputs=inputs
			for layer in self._layers:
				ref_outputs = layer(ref_outputs)



			# Single layer GRU
			outputs = ref_outputs
			# average pooling or final states
			if self._average_pooling:
				pool_state = tf.reduce_mean(outputs, axis=1)
				return pool_state
			else:
				return outputs
