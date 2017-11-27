
import tensorflow as tf
slim = tf.contrib.slim
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from functional import seq

import skimage
import skimage.io
import os
import numpy as np

from config import ModelConfig
from input_ops import Dataset
from ops import normalize

def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

# ops for build model
def vgg_16(inputs,
           scope='vgg_16'):
  """Return: net1, net2, net3, net4 are tensor before pool"""
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net1m = slim.max_pool2d(net1, [2, 2], scope='pool1')
      net2 = slim.repeat(net1m, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net2m = slim.max_pool2d(net2, [2, 2], scope='pool2')
      net3 = slim.repeat(net2m, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net3m = slim.max_pool2d(net3, [2, 2], scope='pool3')
      net4 = slim.repeat(net3m, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net4m = slim.max_pool2d(net4, [2, 2], scope='pool4')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net4m, end_points, net1, net2, net3, net4

# extract func, used for extracting features from vgg16
def extract_feature(images,
                    output_layer='conv4/conv4_3',
                    stddev=0.1):
  """
  Extract feature from vgg.
  net1m, net2m, net3m, net4m are tensor before pool
  """
  weights_regularizer = None
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=weights_regularizer,
      trainable=False):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        _, end_points, net1m, net2m, net3m, net4m = vgg_16(images)
        output = end_points['vgg_16/' + output_layer]
  return output, net1m, net2m, net3m, net4m

# Re build the model, same as model for training, but return
# more tensors.
def max_model(images, config):
  """
  No need for training.
  Return:
     adaption_output, net1m, net2m, net3m, net4m, net_max_feature
  """
  vgg_out, net1m, net2m, net3m, net4m = extract_feature(images)
  net = vgg_out
  # if output_layer.startswith('vgg_16/'):
  #   return net, net1m, net2m, net3m, None
  with tf.variable_scope("adaption", values=[net]) as scope:
    # if output_layer.startswith("adaption/conv"):
    #   end = int(output_layer[-1])
    # else:
    #   end = len(config.adaption_layer_filters) + 1
    for tmp_idx in range(len(config.adaption_layer_filters)):
      net = tf.layers.conv2d(net, config.adaption_layer_filters[tmp_idx],
                             config.adaption_kernels_size[tmp_idx], config.adaption_layer_strides[tmp_idx],
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=None,
                             activation=tf.nn.relu,
                             padding='same',
                             name='conv' + str(tmp_idx + 1))
    adaption_conv_output = net
      # if tmp_idx == (end - 1):
      #   return net, net1m, net2m, net3m, None
    if config.adaption_fc_layers_num:
      for tmp_idx in range(config.adaption_fc_layers_num):
        net = tf.layers.conv2d(net, config.adaption_fc_filters[tmp_idx], [1, 1],
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=None,
                               activation=tf.nn.relu,
                               padding='same',
                               name='fc' + str(tmp_idx + 1))
    net_reduce_dim = tf.layers.conv2d(net, config.net_global_dim[0], [1, 1],
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      kernel_regularizer=None,
                                      activation=tf.nn.relu, name='net_reduce_fc')
    net_reduce_area = tf.nn.avg_pool(net_reduce_dim, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    net_reduce_area = tf.contrib.layers.flatten(net_reduce_area)
    net_global_feature = tf.layers.dense(net_reduce_area, units=config.net_global_dim[1],
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         kernel_regularizer=None,
                                         activation=tf.nn.relu)
    net_max_feature = tf.layers.conv2d(net, config.net_max_features_nums, [1, 1],
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       kernel_regularizer=None,
                                       activation=tf.nn.relu, name='net_max_feature')
    fc_o = tf.reduce_max(net_max_feature, axis=(1, 2), keep_dims=False)
    fc_o = tf.concat([net_global_feature, fc_o], 1)
    adaption_output = tf.layers.dense(fc_o, config.annotation_number,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       kernel_regularizer=None,
                                       activation=None, name='adaption_output')
  return adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature

def get_top_activation(outputs, top_k=10):
  """Traverse model.
  Visualizing and Understanding Convolutional Networks.
  Args:
    outputs: input to traverse model
    top_k: choose most active k
  Return the images, according to outpus.
  i.e. given [1, 128, 320, 64]
      return [1, 128, 320, top_k]
  """
  if outputs.shape[0] != 1:
    raise ValueError("outputs should has shape [1, ...], batch should be 1")
  outputs = outputs.copy()
  # output_dim = outputs.shape[-1]
  output_sum = np.sum(outputs, axis=(0, 1, 2))
  top_activated_idx = np.argsort(output_sum)[-top_k:]
  test_image = outputs[:, :, :, top_activated_idx]
  return test_image, top_activated_idx

def get_top_activation2d(img, top_k=10):
  """Get the top activation in 2d image.
  Return: img with only top_k numbers kept.
  """
  tmp = img.flatten()
  tmp[np.argsort(tmp)[:-top_k]] = 0
  return tmp.reshape(img.shape)

def unpool2x2(x, mask):
  """Unpooling."""
  # first, make net shape as mask.
  out = tf.concat([x, x], 3)
  out = tf.concat([out, out], 2)
  sh = x.get_shape().as_list()
  out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
  out = tf.reshape(out, out_size)
  ret = tf.multiply(out, mask)
  return ret

def img_to_mask(img):
  """Give a 2 dim img, return 2x2 pooling mask."""
  for i in [0, 1]:
    if img.shape[i] % 2 != 0:
      raise ValueError("shape[%d] = %d is not an even" % (i, img.shape[i]))
  mask = np.zeros_like(img)
  for idx_0 in range(0, img.shape[0], 2):
    for idx_1 in range(0, img.shape[1], 2):
      max_idx = np.argmax(img[idx_0:idx_0+2, idx_1:idx_1+2].flatten())
      mask[idx_0 + max_idx//2, idx_1 + max_idx%2] = 1
  return mask

def array_to_mask(array):
  """Given an array, return 2x2 pool mask.
  Which is the inverse op of max-pool with size [2, 2], stride [2, 2]
  Args:
    array.shape[1, 2] of array must be even.
    array should have dim 4.
  Return:
    mask for unpooling.
  """
  for i in [1, 2]:
    if array.shape[i] % 2 != 0:
      raise ValueError("shape[%d] = %d is not an even"%(i, array.shape[i]))
  # mask have the same size of array
  mask = np.zeros_like(array)
  for idx_3 in range(array.shape[-1]):
    for idx_0 in range(array.shape[0]):
      # img is an single batch of size array.shape[1, 2]
      mask[idx_0, ..., idx_3] = img_to_mask(array[idx_0, ..., idx_3])
  return mask

def return_img_mask(array):
  """Return the mask of tf.reduce_max(net_max_feature, axis=(1, 2), keep_dims=False)
  Args:
    array is the tensor before the reduce_max ops.
  """
  b, h, w, c = array.shape
  if b != 1:
    raise ValueError('array should have batch_size=1, not %d'%b)
  mask = np.zeros_like(array)
  for idx in range(c):
    tmp_img = array[0, :, :, idx].flatten()
    tmp_arg = np.argmax(tmp_img)
    mask[0, tmp_arg//w, tmp_arg%w, idx] = 1.0
  return mask

def unpool2x2_no_mask(x):
  """Unpooling."""
  # first, make net shape as mask.
  out = tf.concat([x, x], 3)
  out = tf.concat([out, out], 2)
  sh = x.get_shape().as_list()
  out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
  out = tf.reshape(out, out_size)
  return out

def my_get_variables_by_name(name):
  return [v for v in tf.global_variables() if v.name == name]

def lb_tf_vgg(net, mask, output_layer='conv4'):
  """
  net: tensor
  mask: placeholder need to be feeded, corrsponsed to 'vgg_16/conv1', 'vgg_16/conv2', 'vgg_16/conv3'
  output_layer: like 'conv1' 'vgg_16/conv1', '1'
  """
  # has to be one of ['vgg_16/conv4', 'vgg_16/conv3', 'vgg_16/conv2', 'vgg_16/conv1']
  # vgg part
  vgg_output_layer = int(output_layer[-1])
  vgg_conv = [[1, [2, 1]],
              [2, [2, 1]],
              [3, [3, 2, 1]],
              [4, [2, 1]]]
  # vgg part
  for first_conv_idx in range(vgg_output_layer-1, -1, -1):
    first_conv = vgg_conv[first_conv_idx][0]
    for second_conv in vgg_conv[first_conv_idx][1]:
      conv_name = 'vgg_16/conv%s/conv%s_%s/' % (str(first_conv), str(first_conv), str(second_conv))
      weights = conv_name + 'weights:0'
      bias = conv_name + 'biases:0'
      # get tensor
      w_tensor = my_get_variables_by_name(weights)[0]
      bias_tensor = my_get_variables_by_name(bias)[0]
      # output_shape, for filters, shape[-2] is the input, [-1] is the output in inference
      # but for lookback, [-2] is the output, [-1] is the input
      output_filters = w_tensor.get_shape().as_list()[-2]
      output_shape = [1, int(128 / (2 ** (first_conv - 1))), int(320 / (2 ** (first_conv - 1))), output_filters]
      # unrelu
      net = tf.nn.relu(net)
      # unbiases
      net = tf.nn.bias_add(net, tf.negative(bias_tensor))
      # unconv
      net = tf.nn.conv2d_transpose(net, w_tensor,
                                   output_shape=output_shape,
                                   strides=[1, 1, 1, 1])
    if first_conv != 1:
      # unpool
      # net = unpool2x2(net, mask[first_conv - 2])
      net = unpool2x2_no_mask(net)
  return net

def lb_tf_adaption_conv(net, config):
  """Just like lb_tf_vgg, bulid the tensorflow model for adaption conv.
  Args:
    net: tensor of the input to deconv model, corresponding to output_layter
    output_layter: name of input to deconv model, 'conv1' means 'adaption/conv1'
  """
  # adaption conv2d part, no max-pooling
  # ['adaption/conv1', 'adaption/conv2', 'adaption/conv3']:
    # output: [1, 16, 40, adaption_layer_filters[-1]]
    # input: [1, 16, 40, adaption_layer_filters[0]]
  adaption_output_layer = len(config.adaption_layer_filters)
  for tmp_layer in range(adaption_output_layer, 0, -1):  # conv3 --> conv2 --> conv1
    conv_name = 'adaption/conv%s/'%(str(tmp_layer))
    weights = conv_name + 'kernel:0'
    bias = conv_name + 'bias:0'
    if tmp_layer > 1:
      # conv3 --> adaption_layer_filters[1]
      # conv2 --> adaption_layer_filters[0]
      output_shape = [1, 16, 40, config.adaption_layer_filters[tmp_layer - 2]]
    else:
      # conv1 --> vgg16
      output_shape = [1, 16, 40, 512]  # above layer is vgg16, conv4/conv4_*, so is 512
    w_tensor = my_get_variables_by_name(weights)[0]
    bias_tensor = my_get_variables_by_name(bias)[0]
    net = tf.nn.relu(net)  # unrelu
    # unbiases
    net = tf.nn.bias_add(net, tf.negative(bias_tensor))
    # unconv
    net = tf.nn.conv2d_transpose(net, w_tensor,
                                 output_shape=output_shape,
                                 strides=[1, 1, 1, 1])
  return net

def lb_tf_max(net, net_max_feature_mask, config):
  """Just like lb_tf_vgg, bulid the deconv model for max part.
  Args:
    net: tensor of input for max part, which is output of model.
    net_max_feature: in max part, fc_o = tf.reduce_max(net_max_feature, axis=(1, 2), keep_dims=False)
                    we need to record the information in order to do inverse.
    config: config object
  """
  # net: [1, config.annotation_number]
  # output: [1, net_global_feature + fc_o] which is [1, config.net_max_features_nums+config.net_global_dim]
  kernel_0 = my_get_variables_by_name('adaption/adaption_output/kernel:0')[0]
  bias_0 = my_get_variables_by_name('adaption/adaption_output/bias:0')[0]
  # un activation, no activation
  # un bias
  net = tf.nn.bias_add(net, tf.negative(bias_0))
  # un dense
  kernel_0 = tf.transpose(kernel_0)
  net = tf.matmul(net, kernel_0)

  # global part, the ori model fc_o = tf.concat([net_global_feature, fc_o], 1)
  # so, first config.net_global_dim[-1] are global part
  net_global = net[0][:config.net_global_dim[-1]] # [config.net_global_dim[-1]]
  net_global = tf.reshape(net_global, [1, config.net_global_dim[-1]]) # [1, config.net_global_dim[-1]]
  # un dense
  # [1, 4 * 10 * net_global_dim[1]] --> [1, 4 * 10 * net_global_dim[0]]
  w_tensor = my_get_variables_by_name('adaption/dense/kernel:0')[0]
  bias_tensor = my_get_variables_by_name('adaption/dense/bias:0')[0]
  net_global = tf.nn.bias_add(net_global, tf.negative(bias_tensor))
  w_tensor = tf.transpose(w_tensor)
  net_global = tf.matmul(net_global, w_tensor)
  # un flatten
  net_global = tf.reshape(net_global, [1, 4, 10, config.net_global_dim[0]])
  # un avgpool
  # [1, 4, 10, config.net_global_dim[0]] --> [1, 16, 40, config.net_global_dim[0]]
  # due to the difficulty, just duplicate 4*4 times
  net_global = unpool2x2_no_mask(net_global)
  net_global = unpool2x2_no_mask(net_global)
  # un conv2d
  w_tensor = my_get_variables_by_name('adaption/net_reduce_fc/kernel:0')[0]
  bias_tensor = my_get_variables_by_name('adaption/net_reduce_fc/bias:0')[0]
  output_shape = [1, 16, 40, config.adaption_layer_filters[-1]]
  net_global = tf.nn.relu(net_global)  # unrelu
  # unbiases
  net_global = tf.nn.bias_add(net_global, tf.negative(bias_tensor))
  # unconv
  net_global = tf.nn.conv2d_transpose(net_global, w_tensor,
                               output_shape=output_shape,
                               strides=[1, 1, 1, 1])

  # max part
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


def config_to_dataset(config):
  """Given config, return dataset."""
  Ds = Dataset(config)
  return Ds

def dataset_element_to_batch_data(single_ds, binarizer, config):
  """Given a single data, return batch images, labels."""
  images_path = single_ds['urls']
  labels = single_ds['annot']
  gent_stage = single_ds['gene stage']
  l = binarizer.fit_transform([labels])
  imgs = seq(images_path) \
    .map(lambda path: config.IMAGE_PARENT_PATH + '/' + path.split('/')[-1]) \
    .map(lambda path: skimage.io.imread(path)) \
    .map(lambda img: skimage.img_as_float(img)) \
    .map(lambda img: (img - 0.5) * 2) \
    .list()
  imgs = np.array(imgs)
  temp_image = imgs
  for tmp_idx in range(int(config.max_sequence_length / imgs.shape[0] + 1)):
    temp_image = np.concatenate((temp_image, imgs))
  i = temp_image[:config.max_sequence_length]  # [15, 128, 320, 3]
  return i, l, gent_stage

def forward_model(images, config):
  """Forward model, return vgg conv output, net max model.
  Args:
    images: placeholder
  """
  adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature = max_model(images, config)
  return adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature

def test_lb_vgg(config, output_layer='conv1', top_k_activation=5):
  """Give an image, get the output of conv1, return the
  look back image of it.
  Args:
    conv1_output: output of a group(batch) of images through vgg model,
                  which is infer_output[1] in lookback.py
    vgg_conv_output: ndarray, used to get mask
    netXm: output of vgg/conv* before max pooling
  """
  # get the index of vgg output layer
  vgg_output_layer = int(output_layer[-1])
  shape_feed = [1, int(256/(2**vgg_output_layer)), int(640/(2**vgg_output_layer)), 32 * (2 ** vgg_output_layer)]

  # build forward model
  images = tf.placeholder(tf.float32, [config.max_sequence_length, 128, 320, 3])
  adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature = forward_model(images, config)

  # build model
  feed_input = tf.placeholder(tf.float32, shape_feed) # input for deconv model
  feed_m1 = tf.placeholder(tf.float32, shape=[1, 128, 320, 64]) # mask for vgg/conv1
  feed_m2 = tf.placeholder(tf.float32, shape=[1, 64, 160, 128]) # mask for vgg/conv2
  feed_m3 = tf.placeholder(tf.float32, shape=[1, 32, 80, 256])  # mask for vgg/conv3
  ori_img_output = lb_tf_vgg(feed_input, [feed_m1, feed_m2, feed_m3], output_layer)

  # build sess
  sess = tf.Session()

  # restore
  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)
  saver.restore(sess, '/Users/junr/Documents/prp/pic_data/model/6-10/max.ckpt')

  # prepare the data
  ds = config_to_dataset(config)
  vocab = np.array(ds.vocab)
  binarizer = MultiLabelBinarizer(classes=ds.vocab)
  i, l, gene_stage = dataset_element_to_batch_data(ds.raw_dataset[29], binarizer, config) # just pick one randomly

  # run forward model
  infer_output = sess.run([adaption_output, net1m, net2m, net3m, vgg_out, net_max_feature],
                          feed_dict={
                            images: i
                          })

  # conv1_output: output of a group(batch) of images through vgg model,
  #                 which is infer_output[1] in lookback.py
  # vgg_conv_output: ndarray, used to get mask
  vgg_conv_output = [infer_output[1], infer_output[2], infer_output[3], infer_output[4]]
  conv_output = vgg_conv_output[vgg_output_layer - 1]

  # do some evaluation
  prediction = ''
  output_before_sigmoid = np.max(infer_output[0], 0)
  pred_result = np.argsort(output_before_sigmoid)
  target = binarizer.inverse_transform(l)

  prob = sigmoid(output_before_sigmoid)
  for s in range(5):
    prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                   + str(prob[pred_result[-(s + 1)]]) + '\n')
  result = 'Target: %s \n' % target + 'Prediction: %s ' % prediction

  # for each labels, which image in the batch is the max
  max_images = np.argmax(infer_output[0], 0)

  # batch_img_list: element for every batch
  batch_img_list = []
  labels_index = np.nonzero(l[0])  # only take care of positive labels

  for single_img_idx in labels_index[0]:
    tmp_label = vocab[single_img_idx]
    single_batch = max_images[single_img_idx]
    # top_activation: [1, 128, 320, top_k_activation]
    # this is top activated among kernels, but what if we want to get
    # top activated in a single image.
    # do that in inner loop
    top_activation, top_activation_index = get_top_activation(
                conv_output[single_batch][None, :], top_k=top_k_activation)

    # img_list: element for every top activation img
    img_list = []
    tensor_to_feed = np.zeros(shape_feed)
    for idx in range(top_k_activation):
      # build feed dict for deconv model
      img_to_feed = top_activation[..., idx]
      # img_to_feed[0] = get_top_activation2d(img_to_feed[0], 2)
      img_to_feed_idx = top_activation_index[idx]
      tensor_to_feed[..., img_to_feed_idx] = img_to_feed

    sess_output = sess.run([ori_img_output], feed_dict={
      feed_input: tensor_to_feed,
      feed_m1: array_to_mask(vgg_conv_output[0][single_batch, ...][None, :]),
      feed_m2: array_to_mask(vgg_conv_output[1][single_batch, ...][None, :]),
      feed_m3: array_to_mask(vgg_conv_output[2][single_batch, ...][None, :])
    })
    img_list.append({'img': sess_output[0][0],
                    'label': tmp_label})
    batch_img_list.append([img_list, i[single_batch]/2.0+0.5])
  return batch_img_list, result, binarizer.inverse_transform(l)[0], gene_stage

def test_lb_adaption_conv(config, top_k_activation=2):
  """Same like test_lb_vgg.
  Args:
    output_layer: means adaption/* not vgg/*
  """
  # build forward model
  images = tf.placeholder(tf.float32, [config.max_sequence_length, 128, 320, 3])
  adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature = forward_model(images, config)

  # build model
  # adaption part
  shape_feed = [1, 16, 40, config.adaption_layer_filters[-1]]
  adaption_feed_input = tf.placeholder(tf.float32, shape=shape_feed)
  vgg_input = lb_tf_adaption_conv(adaption_feed_input, config)

  # vgg part
  # vgg_input = tf.placeholder(tf.float32, [1, 16, 40, 512])  # input for deconv model
  feed_m1 = tf.placeholder(tf.float32, shape=[1, 128, 320, 64])  # mask for vgg/conv1
  feed_m2 = tf.placeholder(tf.float32, shape=[1, 64, 160, 128])  # mask for vgg/conv2
  feed_m3 = tf.placeholder(tf.float32, shape=[1, 32, 80, 256])  # mask for vgg/conv3
  ori_img_output = lb_tf_vgg(vgg_input, [feed_m1, feed_m2, feed_m3], output_layer='conv4')

  # build sess
  sess = tf.Session()

  # restore
  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)
  saver.restore(sess, '/Users/junr/Documents/prp/pic_data/model/6-10/max.ckpt')

  # prepare the data
  ds = config_to_dataset(config)
  vocab = np.array(ds.vocab)
  binarizer = MultiLabelBinarizer(classes=ds.vocab)
  i, l, gene_stage = dataset_element_to_batch_data(ds.raw_dataset[23], binarizer, config)  # just pick one randomly

  # run forward model
  infer_output = sess.run([adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature],
                          feed_dict={
                            images: i
                          })

  # do some evaluation
  prediction = ''
  output_before_sigmoid = np.max(infer_output[0], 0)
  pred_result = np.argsort(output_before_sigmoid)
  target = binarizer.inverse_transform(l)
  # print('Target: %s' % target)

  prob = sigmoid(output_before_sigmoid)
  for s in range(5):
    prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                   + str(prob[pred_result[-(s + 1)]]) + '\n')
  # print('Prediction: %s ' % prediction)

  result = 'Target: %s \n' % target + 'Prediction: %s ' % prediction

  # conv1_output: output of a group(batch) of images through vgg model,
  #                 which is infer_output[1] in lookback.py
  # vgg_conv_output: ndarray, used to get mask
  vgg_conv_output = [infer_output[1], infer_output[2], infer_output[3], infer_output[4]]
  conv_output = infer_output[5] # adaption_conv_output

  # for each labels, which image in the batch is the max
  max_images = np.argmax(infer_output[0], 0)

  # batch_img_list: element for every batch
  batch_img_list = []
  labels_index = np.nonzero(l[0])  # only take care of positive labels
  for single_img_idx in labels_index[0]:
    tmp_label = vocab[single_img_idx]
    single_batch = max_images[single_img_idx]

    # top_activation: [1, 128, 320, top_k_activation]
    # this is top activated among kernels, but what if we want to get
    # top activated in a single image.
    # do that in inner loop
    top_activation, top_activation_index = get_top_activation(
      conv_output[single_batch][None, :], top_k=top_k_activation)

    # img_list: element for every top activation img
    img_list = []
    for idx in range(top_k_activation):
      # build feed dict for deconv model
      img_to_feed = top_activation[..., idx]
      # img_to_feed means top activation among different kernels
      # next we want get top activation in an img
      # img_to_feed has batch 1
      # do not work well for lower layer
      # img_to_feed[0] = get_top_activation2d(img_to_feed[0], 10)

      img_to_feed_idx = top_activation_index[idx]
      tensor_to_feed = np.zeros(shape_feed)
      tensor_to_feed[..., img_to_feed_idx] = img_to_feed

      sess_output = sess.run([ori_img_output], feed_dict={
        adaption_feed_input: tensor_to_feed,
        feed_m1: array_to_mask(vgg_conv_output[0][single_batch, ...][None, :]),
        feed_m2: array_to_mask(vgg_conv_output[1][single_batch, ...][None, :]),
        feed_m3: array_to_mask(vgg_conv_output[2][single_batch, ...][None, :])
      })
      img_list.append({'img': sess_output[0][0],
                      'label': tmp_label})
    batch_img_list.append([img_list, i[single_batch] / 2.0 + 0.5])
  return batch_img_list, result, binarizer.inverse_transform(l)[0], gene_stage


def test_lb_whole(config):
  # build forward model
  images = tf.placeholder(tf.float32, [config.max_sequence_length, 128, 320, 3])
  adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature = forward_model(images, config)

  # build model
  # output part
  feature_feed = tf.placeholder(tf.float32, shape=[1, config.annotation_number])
  shape_feed = [1, config.annotation_number]
  net_max_feature_mask_feed = tf.placeholder(tf.float32, shape=[1, 16, 40, config.net_max_features_nums])
  adaption_input = lb_tf_max(feature_feed, net_max_feature_mask_feed, config)

  # adaption part
  vgg_input = lb_tf_adaption_conv(adaption_input, config)

  # vgg part
  # vgg_input = tf.placeholder(tf.float32, [1, 16, 40, 512])  # input for deconv model
  feed_m1 = tf.placeholder(tf.float32, shape=[1, 128, 320, 64])  # mask for vgg/conv1
  feed_m2 = tf.placeholder(tf.float32, shape=[1, 64, 160, 128])  # mask for vgg/conv2
  feed_m3 = tf.placeholder(tf.float32, shape=[1, 32, 80, 256])  # mask for vgg/conv3
  ori_img_output = lb_tf_vgg(vgg_input, [feed_m1, feed_m2, feed_m3], output_layer='conv4')

  # build sess
  sess = tf.Session()

  # restore
  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)
  saver.restore(sess, '/Users/junr/Documents/prp/pic_data/model/6-10/max.ckpt')

  # prepare the data
  ds = config_to_dataset(config)
  vocab = np.array(ds.vocab)
  binarizer = MultiLabelBinarizer(classes=ds.vocab)
  i, l, gene_stage = dataset_element_to_batch_data(ds.raw_dataset[23], binarizer, config)  # just pick one randomly

  # run forward model
  infer_output = sess.run([adaption_output, net1m, net2m, net3m, vgg_out, adaption_conv_output, net_max_feature],
                          feed_dict={
                            images: i
                          })

  # do some evaluation
  prediction = ''
  output_before_sigmoid = np.max(infer_output[0], 0)
  pred_result = np.argsort(output_before_sigmoid)
  target = binarizer.inverse_transform(l)
  # print('Target: %s' % target)

  prob = sigmoid(output_before_sigmoid)
  for s in range(5):
    prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                   + str(prob[pred_result[-(s + 1)]]) + '\n')
  # print('Prediction: %s ' % prediction)

  result = 'Target: %s \n' % target + 'Prediction: %s ' % prediction

  # for each labels, which image in the batch is the max
  max_images = np.argmax(infer_output[0], 0)

  labels_index = np.nonzero(l[0]) # only take care of positive labels
  batch_img_list = []
  for single_img_idx in labels_index[0]:
    tmp_label = vocab[single_img_idx]
    # used as input to deconv model
    max_infer_out = np.zeros((1, config.annotation_number))
    max_infer_out[0, single_img_idx] = infer_output[0][max_images[single_img_idx], single_img_idx]

    lb_output = sess.run([ori_img_output], feed_dict={
      feature_feed: max_infer_out,
      net_max_feature_mask_feed: return_img_mask(infer_output[6][single_img_idx, ...][None, :]),
      feed_m1: array_to_mask(infer_output[1][single_img_idx, ...][None, :]),
      feed_m2: array_to_mask(infer_output[2][single_img_idx, ...][None, :]),
      feed_m3: array_to_mask(infer_output[3][single_img_idx, ...][None, :])
    })

    batch_img_list.append({'img': lb_output[0][0],
                           'label': tmp_label})
  return batch_img_list, i, result, binarizer.inverse_transform(l)[0], gene_stage


def return_a_config():
  config = ModelConfig(max_sequence_length=15, annotation_number=10,
                       adaption_layer_filters=[1024, 1024, 512],
                       net_global_dim=[128, 256],
                       net_max_features_nums=512,
                       stages=[6],
                       GRAND_PARENT_PATH='/Users/junr/Documents/prp/pic_data',
                       IMAGE_PARENT_PATH='/Users/junr/Documents/prp/pic_data/pic_data'
                       )
  config.finish()
  return config

def simple_gray_nor_imshow(img):
  plt.imshow(normalize(img), cmap ='gray')

def simple_save_result_from_test_lb_whole(batch_img_list, i, result, labels, stage, config):
  """
  """
  parent_path = os.path.join(config.PARENT_PATH, stage)
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)
  # save labels
  labels_str = ''
  for l in labels:
    labels_str += (l + '\n')
  with open(parent_path + '/label.txt', 'w') as f:
    f.write(labels_str)

  parent_path = os.path.join(parent_path, 'whole_output')
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

  # save the ori
  for idx in range(i.shape[0]):
    skimage.io.imsave(parent_path + '/ori_' + str(idx) + '.bmp', i[idx])
  # save the result
  with open(parent_path+'/result.txt', 'w') as f:
    f.write(result)

  for img in batch_img_list:
    tmp_img = img['img']
    tmp_label = img['label'].replace('/', '-')
    skimage.io.imsave(parent_path+'/'+tmp_label+'.bmp', normalize(tmp_img))

def simple_save_result_from_test_lb(batch_img_list, result, labels, stage, config, output_layer):
  """Simple save the result.
  """
  if '/' in output_layer:
    raise ValueError('output_layer should not have /, but %s'%output_layer)
  parent_path = os.path.join(config.PARENT_PATH, stage)
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

  # save labels
  labels_str = ''
  for l in labels:
    labels_str += (l + '\n')
  with open(parent_path + '/label.txt', 'w') as f:
    f.write(labels_str)

  with open(parent_path + '/result.txt', 'w') as f:
    f.write(result)

  parent_path = os.path.join(parent_path, output_layer)
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

  tmp_count = 0
  for single_batch in batch_img_list:
    tmp_count += 1
    tmp_path = os.path.join(parent_path, str(tmp_count))
    if not os.path.isdir(tmp_path):
      os.mkdir(tmp_path)
    ori_img = single_batch[1]
    skimage.io.imsave(tmp_path + '/ori.bmp', ori_img)
    tmp_count_inner = 0
    for single_kernel_lb in single_batch[0]:
      tmp_count_inner += 1
      skimage.io.imsave(tmp_path + '/' + single_kernel_lb['label'].replace('/', '-') + str(tmp_count_inner) + '.bmp',
                        normalize(single_kernel_lb['img']))











