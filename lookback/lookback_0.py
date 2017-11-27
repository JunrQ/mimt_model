"""Only get the field.
Do not take care of feature.
"""
import numpy as np
import tensorflow as tf

from .lookback_ops import max_model, my_get_variables_by_name, \
            unpool2x2_no_mask


def lb_tf_max(net, mask, config):
  """
  """
  # net: [1, config.annotation_number]
  # output: [1, net_global_feature + fc_o] which is [1, config.net_max_features_nums+config.net_global_dim]
  kernel_0 = my_get_variables_by_name('adaption/adaption_output/kernel:0')[0]

  net = tf.matmul(net, kernel_0) # take it as prob
  net = tf.multiply(net, mask[-1])

  # global part, the ori model fc_o = tf.concat([net_global_feature, fc_o], 1)
  # so, first config.net_global_dim[-1] are global part
  net_global = net[0][:config.net_global_dim[-1]] # [config.net_global_dim[-1]]
  net_global = tf.reshape(net_global, [1, config.net_global_dim[-1]]) # [1, config.net_global_dim[-1]]
  # un dense
  # [1, 4 * 10 * net_global_dim[1]] --> [1, 4 * 10 * net_global_dim[0]]
  w_tensor = my_get_variables_by_name('adaption/dense/kernel:0')[0]
  w_tensor = tf.transpose(w_tensor)
  net_global = tf.matmul(net_global, w_tensor)
  net_global = tf.multiply(net_global, mask[-2])

  # un flatten
  net_global = tf.reshape(net_global, [1, 4, 10, config.net_global_dim[0]])
  # un avgpool

  w_tensor = tf.ones([6, 6, config.net_global_dim[0], config.net_global_dim[0]], dtype=tf.float32)
  net_global = tf.nn.conv2d_transpose(net_global, w_tensor, output_shape=[1, 16, 40, config.net_global_dim[0]],
                                   strides=[1, 4, 4, 1], padding='SAME')
  net_global = tf.div(net_global, 6*6)

  #
  w_tensor = my_get_variables_by_name('adaption/net_reduce_fc/kernel:0')[0]
  bias_tensor = my_get_variables_by_name('adaption/net_reduce_fc/bias:0')[0]
  output_shape = [1, 16, 40, config.adaption_layer_filters[-1]]
  net_global = tf.nn.relu(net_global)  # unrelu
  # unbiases
  net_global = tf.nn.bias_add(net_global, tf.negative(bias_tensor))
  # unconv
  net_global = tf.nn.conv2d_transpose(net_global, tf.zeros_like(w_tensor),
                               output_shape=output_shape,
                               strides=[1, 1, 1, 1])

  net_max = net[0][config.net_global_dim[-1]:] # [config.net_max_features_nums, ]
  net_max = tf.reshape(net_max, [1, config.net_max_features_nums]) # [1, config.net_max_features_nums]
  net_max = tf.multiply(net_max, net_max_feature_mask) # mask

  net_max = tf.nn.relu(net_max)
  net_max = tf.nn.bias_add(net_max, tf.negative(my_get_variables_by_name('adaption/net_max_feature/bias:0')[0]))
  if config.adaption_fc_layers_num:
    raise ValueError("config.adaption_fc_layers_num not supported.")
  else:
    tmp_shape_ = config.adaption_layer_filters[-1]
  net_max = tf.nn.conv2d_transpose(net_max, my_get_variables_by_name('adaption/net_max_feature/kernel:0')[0],
                               output_shape=[1, 16, 40, tmp_shape_],
                               strides=[1, 1, 1, 1])

  # connect max with global feature
  net = tf.add(net_max, net_global)
  return net





# infer
def inference(w, b):
  input_img = np.ones((128, 320, 3), dtype=np.float32)

  # vgg_16/conv
  input_img = my_conv_3x3_1x1(input_img, w, b)
