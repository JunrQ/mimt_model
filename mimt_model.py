import tensorflow as tf

import tensorflow.contrib.slim as slim


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
      net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv4/conv4_1')
      end_points = slim.utils.convert_collection_to_dict(
        end_points_collection)
      return net, end_points


def extract_feature_vgg(images,
                        weight_decay=0.00004,
                        trainable=False,
                        stddev=0.1):
  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None
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
  return net


class MaxModel():
  """Model.
  extract_feature will return vgg features,
  """

  def __init__(self, images, labels, config, is_training):
    """Model constructor.
    Args:
    """
    self.config = config
    self.is_training = is_training
    labels = tf.cast(labels, tf.float32)

    images = tf.reshape(images, [-1, self.config.height, self.config.width, 3])

    feature = extract_feature_vgg(images,
                                  trainable=self.config.vgg_trainable,
                                  weight_decay=self.config.weight_decay)
    net = feature  # [8, 20]
    tf.summary.histogram('vgg_conv4_1_output', net)

    with tf.variable_scope("adaption", values=[net]) as scope:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                          biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                          biases_initializer=tf.zeros_initializer(),
                          activation_fn=tf.nn.relu
                          ):
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv_vgg4')
        net = slim.max_pool2d(net, [2, 2], scope='pool_vgg4')
        net = slim.conv2d(net, 256, [3, 3], padding='SAME', scope='conv_vgg5')
        net = slim.max_pool2d(net, [2, 2], scope='pool_vgg5')
        tf.summary.histogram('vgg_5_output', net)

        net = tf.shape(net, [self.config.batch_size, self.config.max_sequence_length, self.config.height, self.config.width, 3])

        tmp_net = []
        for tmp_ in tf.unstack(net):
          tmp_ = tf.concat([tmp for tmp in tf.unstack(tmp_)], axis=-1)  # [4, 10, 512*15=7680]
          tmp_ = tf.expand_dims(tmp_, 0)  # [1, 4, 10, 512*15]
          tmp_net.append(tmp_)
        net = tf.stack(tmp_net)

        net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv1')
        net = tf.layers.dropout(net, training=self.is_training)
        net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv2')
        net = tf.layers.dropout(net, training=self.is_training)
        net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv3')
        tf.summary.histogram('adaption_conv_output', net)

        net = tf.reduce_mean(net, axis=(1, 2))

        net = slim.fully_connected(net, 1024, scope='fc1')
        net = tf.layers.dropout(net, training=self.is_training)
        net = slim.fully_connected(net, 1024, scope='fc2')
        tf.summary.histogram('adaption_fc_output', net)

        logits = slim.fully_connected(net, self.config.annotation_number, activation_fn=None, scope='logits')
        tf.summary.histogram('logits_output', logits)

    self.output = logits
    self.output_prob = tf.nn.sigmoid(logits)

    logits_neg = tf.where(tf.greater(self.output_prob, self.config.neg_threshold),
                          tf.subtract(tf.ones_like(labels), labels),
                          tf.zeros_like(labels))
    logits_pos = tf.where(tf.less(self.output_prob, self.config.pos_threshold),
                          labels,
                          tf.zeros_like(labels))

    self.prediction = tf.where(tf.greater(self.output_prob, self.config.threshold),
                               tf.ones_like(self.output_prob, dtype=tf.float32),
                               tf.zeros_like(self.output_prob, dtype=tf.float32))
    # output
    # self.cross_entropy = tf.losses.sigmoid_cross_entropy(labels, logits)

    self.cross_entropy = -(tf.reduce_sum(tf.multiply(tf.pow(self.output_prob, self.config.gamme*tf.ones_like(labels)),
                                        tf.multiply((1.0 - labels), tf.log(1. - self.output_prob + 1e-10)))
                                      ) +
        self.config.loss_ratio * tf.reduce_sum(
                                tf.multiply(tf.pow(1. - self.output_prob, self.config.gamme*tf.ones_like(labels)),
                                            tf.multiply(labels, tf.log(self.output_prob + 1e-10)))
                             )
                           )

    '''
    self.cross_entropy = -(tf.reduce_sum(tf.multiply(logits_neg, tf.log(1. - self.output_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(logits_pos, tf.log(self.output_prob + 1e-10)))
                           )
    '''
    # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    tf.summary.scalar('cross_entropy', self.cross_entropy)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    self.cost = tf.add(tf.reduce_sum(self.cross_entropy), tf.reduce_sum(regularization_losses))
    tf.summary.scalar('loss', self.cost)

  def _get_optimizer(self):
    lr = tf.get_variable('learning_rate', shape=(),
                         dtype=tf.float32, trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    return tf.train.AdamOptimizer(learning_rate=lr)


