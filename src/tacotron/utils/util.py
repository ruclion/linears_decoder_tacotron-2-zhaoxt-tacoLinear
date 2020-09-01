import tensorflow as tf
import numpy as np
from hparams import hparams as hp

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def vae_weight(global_step):
    warm_up_step = hp.vae_warming_up
    w1 = tf.cond(
        global_step < warm_up_step,
        lambda: tf.cond(
            global_step % hp.update_kl_before < 1,
            lambda: 1.0 / (1.0 + tf.exp(-hp.kl_k * tf.to_float(global_step - hp.kl_x))),
            lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
         ),
        lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
    )

    w2 = tf.cond(
        global_step > warm_up_step,
        lambda: tf.cond(
             global_step % hp.update_kl_after < 1,
             lambda: 1.0 / (1.0 + tf.exp(-hp.kl_k * tf.to_float(warm_up_step - hp.kl_x))),
             lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
         ),
        lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
    )
    return tf.maximum(w1, w2)



