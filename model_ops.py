import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim

from resnet_v2_101 import resnet_v2_101, resnet_arg_scope


def partial_match_tensor_name(tensor_dict, name):
  for key, value in tensor_dict.items():
    p = re.compile(name)
    if p.search(key):
      return value
  raise KeyError(f"Can not find any tensor with {name}")


def extract_feature_resnet(images, is_training, weight_decay):
  """ Extract feature from image set.

  Args:
      images: Tensor of shape [N, T, H, W, C].
      is_training: boolean tensor, indicate whether extractor acts in training mode
          or inference mode.
      weight_decay: l2 regularization parameter.

  Return:
      recover_ts: Tensor of shape [N, T, F].
  """
  # images = image_preprocess(images)
  with slim.arg_scope(resnet_arg_scope()):
    with slim.arg_scope([slim.conv2d], trainable=False, weights_regularizer=None):
      with slim.arg_scope([slim.batch_norm], trainable=False):
        _, end_points = resnet_v2_101(
          images, is_training=is_training)
  feature = end_points['resnet_v2_101/block3']
  return feature


def extract_feature_vgg(images,
                        weight_decay=0.00004,
                        output_layer='conv4/conv4_3',
                        trainable=False,
                        stddev=0.1):
  """
  """
  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None
  # images = image_preprocess(images)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=weights_regularizer,
      trainable=trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(
                          stddev=stddev),
                        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        net, end_points = vgg_16(
          images, scope='vgg_16')
        output = end_points['vgg_16/' + output_layer]
  return output


def vgg_16(inputs,
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d,
                        64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')

      end_points = slim.utils.convert_collection_to_dict(
        end_points_collection)

      return net, end_points


"""
Input:
  Parameter:
    max_img: a group with images more than max_img will be abandoned,
             otherswise, will be repeated to max_img
             Example: max_img = 5, a group of images [[img1], [img2], [img3]], will be
                      [[img1], [img2], [img3], [img1], [img2]]
    stage: gene stage
    top_k_label: only take care of most frequent labels
  shape: [max_img, 128, 320, 3]
  dtype: tf.float32
Label:
  Parameters:
    top_k_label: top_k_label is also number of classes(labels)
  shape: [1, top_k_laebls]
  dtype: tf.float32
"""


class MaxModel():
  """Model.
  extract_feature will return vgg features,
  """

  def __init__(self, images, labels, config, is_training):
    """Model constructor.
    Args:
    """
    self.config = config
    self.images = images
    # self.labels = labels
    labels = tf.cast(labels, tf.float32)
    self.is_training = is_training
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    # kernel_initializer = tf.truncated_normal_initializer(0.001)

    if self.config.vgg_or_resnet == 'vgg':
      feature = extract_feature_vgg(self.images,
                                    trainable=self.config.vgg_trainable,
                                    output_layer=self.config.vgg_output_layer,
                                    weight_decay=self.config.weight_decay)
      tf.summary.histogram('vgg_output', feature)

    elif self.config.vgg_or_resnet == 'resnet':
      feature = extract_feature_resnet(self.images,
                                       weight_decay=self.config.weight_decay)
      tf.summary.histogram('resnet_output', feature)
    net = feature
    with tf.variable_scope("adaption", values=[net]) as scope:
      # pool0 -- 8 * 20
      # net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # net = slim.max_pool2d(self.vgg_output, [2, 2], scope='pool0')
      # conv1
      for tmp_idx in range(len(self.config.adaption_layer_filters)):
        net = tf.layers.conv2d(net, self.config.adaption_layer_filters[tmp_idx],
                               self.config.adaption_kernels_size[tmp_idx], self.config.adaption_layer_strides[tmp_idx],
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv' + str(tmp_idx + 1))
        net = tf.layers.dropout(net, training=self.is_training)

      tf.summary.histogram('adaption_conv2d', net)
      if self.config.adaption_fc_layers_num:
        if self.config.adaption_fc_layers_num != len(self.config.adaption_fc_filters):
          raise ValueError("adaption_fc_layers_num should equal len()")
        for tmp_idx in range(self.config.adaption_fc_layers_num):
          net = tf.layers.conv2d(net, self.config.adaption_fc_filters[tmp_idx], [1, 1],
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                 kernel_initializer=kernel_initializer,
                                 activation=tf.nn.relu,
                                 padding='same',
                                 name='fc' + str(tmp_idx + 1))
          net = tf.layers.dropout(net, training=self.is_training)

      if config.global_conv2d:
        net_reduce_dim = net
        for tmp_idx in range(len(self.config.net_global_dim) - 1):
          net_reduce_dim = tf.layers.conv2d(net_reduce_dim, self.config.net_global_dim[tmp_idx],
                                            self.config.net_global_kernel_size[tmp_idx],
                                            self.config.net_global_stride[tmp_idx],
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                              self.config.weight_decay),
                                            activation=tf.nn.relu, padding='same', name='net_reduce_fc' + str(tmp_idx))
          # net_reduce_dim = tf.layers.dropout(net_reduce_dim, training=self.is_training)
      else:
        net_reduce_dim = tf.layers.conv2d(net, self.config.net_global_dim[0], [1, 1],
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                          activation=tf.nn.relu, name='net_reduce_fc')
        # reduce area from 16*40 to 4 * 10
        net_reduce_dim = tf.nn.avg_pool(net_reduce_dim, ksize=[1, 6, 6, 1], strides=[1, 4, 4, 1], padding='SAME')
      net_reduce_dim = tf.contrib.layers.flatten(net_reduce_dim)
      # fully connect layer, make dim to self.net_global_dim[1]
      net_global_feature = tf.layers.dense(net_reduce_dim, units=self.config.net_global_dim[-1],
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                             self.config.weight_decay),
                                           activation=tf.nn.relu)
      tf.summary.histogram('net_global_feature', net_global_feature)

      # max feature, net to a conv2d layers
      net_max_feature = net
      # net_max_feature = tf.nn.avg_pool(net_max_feature, ksize=[1, 6, 6, 1], strides=[1, 4, 4, 1], padding='SAME')
      for tmp_idx in range(len(self.config.net_max_features_nums)):
        net_max_feature = tf.layers.conv2d(net_max_feature, self.config.net_max_features_nums[tmp_idx],
                                           self.config.net_max_features_size[tmp_idx],
                                           self.config.net_max_features_stride[tmp_idx],
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                             self.config.weight_decay),
                                           activation=tf.nn.relu, padding='same', name='net_max_feature' + str(tmp_idx))
        net_max_feature = tf.layers.dropout(net_max_feature, training=self.is_training)
      fc_o = tf.reduce_max(net_max_feature, axis=(1, 2), keep_dims=False)
      tf.summary.histogram('net_max_feature', fc_o)
      if self.config.plus_global_feature:
        # concatenate them
        # [batch_size, max_img, net_global_dim[-1] + net_max_features_nums]
        fc_o = tf.concat([net_global_feature, fc_o], 1)

      if len(self.config.output_units) > 0:
        for tmp_idx in range(len(self.config.output_units)):
          fc_o = tf.layers.dense(fc_o, self.config.output_units[tmp_idx],
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                   self.config.weight_decay),
                                 activation=tf.nn.relu, name='adaption_output' + str(tmp_idx + 1))
          fc_o = tf.layers.dropout(fc_o, training=self.is_training)

      self.adaption_output = tf.layers.dense(fc_o, self.config.annotation_number,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                               self.config.weight_decay),
                                             activation=None, name='adaption_output_final')
      tf.summary.histogram('adaption_output', self.adaption_output)
      self.output = tf.reduce_max(self.adaption_output, axis=[0, ], keep_dims=True)

    output_prob = tf.sigmoid(self.output)
    adaption_prob = tf.sigmoid(self.adaption_output)

    logits_neg = tf.where(tf.greater(adaption_prob, self.config.neg_threshold),
                          tf.subtract(tf.ones_like(adaption_prob), labels),
                          tf.zeros_like(adaption_prob))
    logits_pos = tf.where(tf.less(output_prob, self.config.pos_threshold),
                          labels,
                          tf.zeros_like(labels))

    self.cross_entropy = -(tf.reduce_sum(tf.multiply(logits_neg, tf.log(1. - adaption_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(logits_pos, tf.log(output_prob + 1e-10)))
                           )

    '''
    self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - labels), tf.log(1. - adaption_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(labels, tf.log(output_prob + 1e-10)))
                           )
    '''
    '''
    self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - labels), tf.log(1. - output_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(labels, tf.log(output_prob + 1e-10)))
                           )
    '''
    tf.summary.scalar('cross_entropy', self.cross_entropy)

    # auc, _ = tf.metrics.auc(labels, output_prob, updates_collections=[tf.GraphKeys.UPDATE_OPS])
    # tf.summary.scalar('training_auc', auc)

    self.prediction = tf.where(tf.greater(output_prob, self.config.threshold),
                               tf.ones_like(labels, dtype=tf.int32), tf.zeros_like(labels, dtype=tf.int32))
    self.prediction = tf.cast(self.prediction, tf.float32)

    self.output_prob = output_prob
    # wrong num
    self.wrong_number = tf.reduce_sum(
      tf.where(tf.equal(labels, self.prediction), tf.zeros_like(labels), tf.ones_like(labels))) \
                        / (self.config.annotation_number * self.config.batch_size)
    tf.summary.scalar('wrong_number', self.wrong_number)
    # loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.cost = tf.add_n([self.cross_entropy] + regularization_losses)
    tf.summary.scalar('loss', self.cost)

  def optimizer(self):
    # Not used.
    lr = tf.get_variable('learning_rate', shape=(),
                         dtype=tf.float32, trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    return tf.train.AdamOptimizer(learning_rate=lr)


class ConvModel():
  """Model.
  extract_feature will return vgg features,
  """

  def __init__(self, images, labels, config, is_training):
    """Model constructor.
    Args:
    """
    self.config = config
    self.images = images
    # self.labels = labels
    labels = tf.cast(labels, tf.float32)
    labels = tf.tile(labels, [self.config.max_sequence_length, 1])
    self.is_training = is_training
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    # kernel_initializer = tf.truncated_normal_initializer(0.001)

    if self.config.vgg_or_resnet == 'vgg':
      feature = extract_feature_vgg(self.images,
                                    trainable=self.config.vgg_trainable,
                                    output_layer=self.config.vgg_output_layer,
                                    weight_decay=self.config.weight_decay)
      tf.summary.histogram('vgg_output', feature)

    elif self.config.vgg_or_resnet == 'resnet':
      feature = extract_feature_resnet(self.images,
                                       weight_decay=self.config.weight_decay)
      tf.summary.histogram('resnet_output', feature)
    net = feature
    with tf.variable_scope("adaption", values=[net]) as scope:
      # pool0 -- 8 * 20
      # net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # net = slim.max_pool2d(self.vgg_output, [2, 2], scope='pool0')
      # conv1
      for tmp_idx in range(len(self.config.adaption_layer_filters)):
        net = tf.layers.conv2d(net, self.config.adaption_layer_filters[tmp_idx],
                               self.config.adaption_kernels_size[tmp_idx], self.config.adaption_layer_strides[tmp_idx],
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv' + str(tmp_idx + 1))
        tf.summary.histogram('adaption_conv' + str(tmp_idx + 1), net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.dropout(net, training=self.is_training)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 512, activation=tf.nn.relu)
    tf.summary.histogram('dense0', net)
    net = tf.layers.dropout(net, training=self.is_training)
    net = tf.layers.dense(net, self.config.annotation_number, activation=tf.nn.relu)
    tf.summary.histogram('output', net)
    output_prob = tf.sigmoid(net)
    # adaption_prob = tf.sigmoid(self.adaption_output)
    logits_neg = tf.where(tf.greater(output_prob, self.config.neg_threshold),
                          tf.subtract(tf.ones_like(labels), labels),
                          tf.zeros_like(labels))
    logits_pos = tf.where(tf.less(output_prob, self.config.pos_threshold),
                          labels,
                          tf.zeros_like(labels))

    self.cross_entropy = -(tf.reduce_sum(tf.multiply(logits_neg, tf.log(1. - output_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(logits_pos, tf.log(output_prob + 1e-10)))
                           )

    '''
    self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - labels), tf.log(1. - adaption_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(labels, tf.log(output_prob + 1e-10)))
                           )
    '''
    '''
    self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - labels), tf.log(1. - output_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(labels, tf.log(output_prob + 1e-10)))
                           )
    '''
    tf.summary.scalar('cross_entropy', self.cross_entropy)

    # auc, _ = tf.metrics.auc(labels, output_prob, updates_collections=[tf.GraphKeys.UPDATE_OPS])
    # tf.summary.scalar('training_auc', auc)
    self.output = tf.reduce_max(net, axis=(0,), keep_dims=True)
    self.output_prob = tf.reduce_max(output_prob, axis=(0,), keep_dims=True)
    self.prediction = tf.where(tf.greater(self.output_prob, self.config.threshold),
                               tf.ones_like(self.output_prob, dtype=tf.int32),
                               tf.zeros_like(self.output_prob, dtype=tf.int32))
    self.prediction = tf.cast(self.prediction, tf.float32)

    # wrong num
    self.wrong_number = tf.reduce_sum(
      tf.where(tf.equal(labels, self.prediction), tf.zeros_like(labels), tf.ones_like(labels))) \
                        / (self.config.annotation_number * self.config.batch_size)
    tf.summary.scalar('wrong_number', self.wrong_number)
    # loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.cost = tf.add_n([self.cross_entropy] + regularization_losses)
    tf.summary.scalar('loss', self.cost)

  def optimizer(self):
    # Not used.
    lr = tf.get_variable('learning_rate', shape=(),
                         dtype=tf.float32, trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    return tf.train.AdamOptimizer(learning_rate=lr)


