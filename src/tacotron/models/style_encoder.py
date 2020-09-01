from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tacotron.models.multihead_attention import MultiheadAttention

class GstEncoder(object):
	"""Global style token style encoder."""
	def __init__(self, hparams, training, scope=None):
		self._num_gst = hparams.num_gst
		self._style_embed_depth = hparams.style_embed_depth
		self._num_heads = hparams.num_heads
		self._style_att_dim = hparams.style_att_dim
		self._scope = scope
		self._style_att_type = hparams.style_att_type
		self._training = training

	def __call__(self, reference):
		with tf.variable_scope(self._scope or 'gst'):
			gst_tokens = tf.get_variable(
				'style_tokens', [self._num_gst, self._style_embed_depth // self._num_heads], dtype=tf.float32)

			style_attention = MultiheadAttention(
				tf.expand_dims(reference, axis=1),  # [N, 1, 128]
				tf.tile(tf.expand_dims(gst_tokens, axis=0),[tf.shape(reference)[0], 1, 1]),
				# [N, hp.num_gst, hp.style_embed_depth/hp.num_heads]
				num_heads=self._num_heads,
				num_units=self._style_att_dim,
				attention_type=self._style_att_type)
			style_embedding = style_attention.multi_head_attention()
			return {'style_embedding': style_embedding}

class VaeEncoder(object):
	def __init__(self, hparams, training, scope=None):
		self._style_embed_depth = hparams.style_embed_depth
		self._z_latent_dim = hparams.z_latent_dim
		self._no_sampling_in_syn = hparams.no_sampling_in_syn
		self._control_duration = hparams.control_duration
		self._scope = scope

		self._mu_proj = tf.layers.Dense(self._z_latent_dim, name="mean")
		self._var_proj = tf.layers.Dense(self._z_latent_dim, name="var")
		self._style_emb_proj = tf.layers.Dense(self._style_embed_depth, name='style_dense')
		self._training = training

	def __call__(self, reference, batch_size):
		with tf.variable_scope(self._scope or 'vae'):
			if self._training and reference is None:
				raise ValueError("No reference speech is provided when vae training!")
			if reference is None:
				z_mu = tf.zeros([batch_size,self._z_latent_dim])
				z_log_var = tf.zeros(tf.shape(z_mu))
				z_embedding = tf.truncated_normal(tf.shape(z_mu))
				#styleControl = 1 - tf.one_hot(0,32)
				#z_embedding = z_mu#*styleControl + 1*tf.one_hot(0,32)
				#z_embedding = tf.Print(z_embedding,[z_embedding])
			else:
				z_mu = self._mu_proj(reference)
				z_log_var = self._var_proj(reference)
				z_std = tf.exp(z_log_var * 0.5)
				if not self._training and self._no_sampling_in_syn:
					z_embedding = z_mu
				else:
					gaussian_noise = tf.truncated_normal(tf.shape(z_mu))
					z_embedding = z_mu + gaussian_noise * z_std
			if self._control_duration:
				style_embedding = None
			else:
				style_embedding = tf.expand_dims(self._style_emb_proj(z_embedding), axis=1)
				z_embedding = None
			return {'style_embedding':style_embedding, 'z_embedding': z_embedding, 'z_mu': z_mu, 'z_log_var':z_log_var}
