# make it sparse
# if weights are too large, normalize it


# For detail, see lookback_cpu.py
import tensorflow as tf
slim = tf.contrib.slim
from sklearn.preprocessing import MultiLabelBinarizer
from functional import seq

import skimage
import skimage.io
import os
import numpy as np

from ..config import ModelConfig
from ..input_ops import Dataset
from .lookback_ops import unpool2x2, unpool2x2_no_mask, \
                        my_get_variables_by_name


# get the dataset
def dataset(config):
  Ds = Dataset(config)
  binarizer = MultiLabelBinarizer(classes=dataset.vocab)
  return Ds, binarizer

def deconv_model(net, netXm, output_layer, config):
  """Deconv model.
  Model should be built before this func, for that this func use my_get_variables_by_name.
  Args:
    net: tensor output
    netXm: [mask_1, mask_2, mask_3, net_max_feature]
    output_layer: name of output
    config:
  """
  FLAG = False
  if output_layer == 'adaption/adaption_output':
    # output: [1, config.annotation_number]
    # input: [1, net_global_feature + fc_o] which is [1, config.net_max_features_nums+config.net_global_dim]
    FLAG = True # all other layer should be considered
    kernel_0 = my_get_variables_by_name('adaption/adaption_output/kernel:0')[0]
    bias_0 = my_get_variables_by_name('adaption/adaption_output/bias:0')[0]
    # un activation, no activation
    # un bias
    net = tf.nn.bias_add(net, tf.negative(bias_0))
    # un dense
    kernel_0 = tf.transpose(kernel_0)
    net = tf.matmul(net, kernel_0)

  if FLAG or output_layer == 'adaption/global_net_feature':
    FLAG = True
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
    # if (not FLAG) and output_layer != 'adaption/net_max_feature':
    #   net = net_global

  if FLAG or output_layer == 'adaption/net_max_feature':
    FLAG = True
    # output: [1, config.net_max_features_nums]
    # input: [1, 16, 40, config.adaption_layer_filters[-1]]
    # un reduce_max
    net_max = net[0][config.net_global_dim[-1]:] # [config.net_max_features_nums, ]
    net_max = tf.reshape(net_max, [1, config.net_max_features_nums]) # [1, config.net_max_features_nums]
    net_max_feature_mask_np = netXm[-1]
    net_max = tf.multiply(net_max, net_max_feature_mask_np) # mask

    net_max = tf.nn.relu(net_max)
    net_max = tf.nn.bias_add(net_max, tf.negative(my_get_variables_by_name('adaption/net_max_feature/bias:0')[0]))
    if config.adaption_fc_layers_num:
      raise ValueError("config.adaption_fc_layers_num not supported.")
    else:
      tmp_shape_ = config.adaption_layer_filters[-1]
    net_max = tf.nn.conv2d_transpose(net_max, my_get_variables_by_name('adaption/net_max_feature/kernel:0')[0],
                                 output_shape=[1, 16, 40, tmp_shape_],
                                 strides=[1, 1, 1, 1])

    if output_layer == 'adaption/global_net_feature':
      # connect max with global feature
      net = tf.add(net_max, net_global)
    else:
      net = net_max

  # adaption conv2d part, no max-pooling
  if FLAG or output_layer in ['adaption/conv1', 'adaption/conv2', 'adaption/conv3']:
    # output: [1, 16, 40, adaption_layer_filters[-1]]
    # input: [1, 16, 40, adaption_layer_filters[0]]
    if FLAG:
      adaption_output_layer = len(config.adaption_layer_filters)
    else:
      FLAG = True
      adaption_output_layer = int(output_layer[-1])
    for tmp_layer in range(adaption_output_layer, 0, -1): # conv3 --> conv2 --> conv2
      conv_name = 'adaption/conv%s/' % (str(tmp_layer))
      weights = conv_name + 'kernel:0'
      bias = conv_name + 'bias:0'
      if tmp_layer != 1:
        # conv3 --> adaption_layer_filters[1]
        # conv2 --> adaption_layer_filters[0]
        output_shape = [1, 16, 40, config.adaption_layer_filters[tmp_layer-2]]
      else:
        # conv1 --> vgg16
        output_shape = [1, 16, 40, 512] # above layer is vgg16, conv4/conv4_3, so is 512
      w_tensor = my_get_variables_by_name(weights)[0]
      bias_tensor = my_get_variables_by_name(bias)[0]
      net = tf.nn.relu(net) # unrelu
      # unbiases
      net = tf.nn.bias_add(net, tf.negative(bias_tensor))
      # unconv
      net = tf.nn.conv2d_transpose(net, w_tensor,
                                   output_shape=output_shape,
                                   strides=[1, 1, 1, 1])

  if FLAG or output_layer in ['vgg_16/conv4', 'vgg_16/conv3', 'vgg_16/conv2', 'vgg_16/conv1']:
    if FLAG:
      vgg_output_layer = 4
    else:
      FLAG = True
      # has to be one of ['vgg_16/conv4', 'vgg_16/conv3', 'vgg_16/conv2', 'vgg_16/conv1']
      # vgg part
      vgg_output_layer = int(output_layer[-1])
    vgg_conv = [[4, [3, 2, 1]],
                [3, [3, 2, 1]],
                [2, [2, 1]],
                [1, [2, 1]]]
    # vgg part
    for first_conv_idx in range(vgg_output_layer):
      first_conv = vgg_conv[first_conv_idx][0]
      for second_conv in vgg_conv[first_conv_idx][1]:
        conv_name = 'vgg_16/conv%s/conv%s_%s/' % (str(first_conv), str(first_conv), str(second_conv))
        weights = conv_name + 'weights:0'
        bias = conv_name + 'biases:0'
        # get tensor
        w_tensor = my_get_variables_by_name(weights)[0]
        bias_tensor = my_get_variables_by_name(bias)[0]
        # output_shape
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
        net = unpool2x2(net, netXm[first_conv - 2])
  return net

# inference func
def inference(output_layer, single_ds, binarizer, config):
  """Return some important information.
  Args:
    single_ds: has element as {'urls': tmp_urls,
                            'annot': ele['annot'],
                            'gene stage': ele['gene stage']}
  """
  # prepare images, label
  images_path = single_ds['urls']
  labels = single_ds['annot']
  l = binarizer.fit_transform([labels])
  vocab = binarizer.classes
  vocab = np.array(vocab)
  imgs = seq(images_path) \
      .map(lambda path: config.IMAGE_PARENT_PATH + '/' + path.split('/')[-1]) \
      .map(lambda path: skimage.io.imread(path)) \
      .map(lambda img: skimage.img_as_float(img)) \
      .map(lambda img: (img- 0.5) * 2)\
      .list()

  imgs = np.array(imgs)
  temp_image = imgs
  for tmp_idx in range(int(config.max_sequence_length / imgs.shape[0] + 1)):
    temp_image = np.concatenate((temp_image, imgs))
  i = temp_image[:config.max_sequence_length] # [15, 128, 320, 3]
  # forward placeholder
  images = tf.placeholder(dtype=tf.float32, shape=[config.max_sequence_length, 128, 320, 3],
                               name="image_feed")
  targets = tf.placeholder(dtype=tf.float32,
                                shape=[1, config.annotation_number],  # batch_size
                                name="input_feed")
  # deconv placeholder
  feature = tf.placeholder(tf.float32, shape=[1, config.annotation_number])
  m1 = tf.placeholder(tf.float32, shape=[1, 128, 320, 64])
  m2 = tf.placeholder(tf.float32, shape=[1, 64, 160, 128])
  m3 = tf.placeholder(tf.float32, shape=[1, 32, 80, 256])
  net_max_feature_mask_np = tf.placeholder(tf.float32, shape=[1, 16, 40, config.net_max_features_nums])

  # forward model
  adaption_output, net1m, net2m, net3m, net_max_feature = model(images, config)
  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)
  # deconv model
  lb_img = deconv_model(feature, [m1, m2, m3, net_max_feature_mask_np], output_layer, config)
  with tf.Session() as sess:
    # restore
    saver.restore(sess, '/Users/junr/Documents/prp/pic_data/model/6-10/max.ckpt')
    # run inference(forward)
    infer_output = sess.run([adaption_output, net1m, net2m, net3m, net_max_feature],
                      feed_dict={
                        images: i,
                        targets: l
                      })
    # do some evaluation
    prediction = ''
    output_before_sigmoid = np.max(infer_output[0], 0)
    pred_result = np.argsort(output_before_sigmoid)
    target = binarizer.inverse_transform(l)
    print('Target: %s' % target)

    prob = sigmoid(output_before_sigmoid)
    # print(pred_result)
    # print(prob)
    for s in range(5):
      prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                    + str(prob[pred_result[-(s + 1)]]) + '\n')
    print('Prediction: %s ' % prediction)

    # for a group of images, find the max img of each label
    # infer_output is a [config.max_sequence_length, config.annotation_number]
    # run deconv
    # first, max in axi=0, which mean max between images
    # print(infer_output[0].shape) # (15, 20), means 15 imgs in a group, 20 labels
    max_images = np.argmax(infer_output[0], 0)
    # print(max_images.shape) (20,)
    # get the index of target
    labels_index = np.nonzero(l[0]) # array([0, 0]), array([13, 14])
    # print(max_images, '\n', labels_index)

    for single_img_idx in labels_index[0]:
      # print(single_img_idx)
      tmp_label = vocab[single_img_idx]
      # print(tmp_label)
      # used as input to deconv model
      max_infer_out = np.zeros((1, config.annotation_number))
      max_infer_out[0, single_img_idx] = infer_output[0][max_images[single_img_idx], single_img_idx]
      # print(max_infer_out)

      # Note, there is not the mask WORNG!!!!
      lb_output = sess.run([lb_img], feed_dict={
        feature: max_infer_out,
        net_max_feature_mask_np: infer_output[4][max_images[single_img_idx]][None, :],
        m1: infer_output[1][max_images[single_img_idx]][None, :],
        m2: infer_output[2][max_images[single_img_idx]][None, :],
        m3: infer_output[3][max_images[single_img_idx]][None, :]
      })
      if not os.path.isdir(os.path.join(config.PARENT_PATH, 'lb_imgs')):
        os.mkdir(os.path.join(config.PARENT_PATH, 'lb_imgs'))
      # save image at config.PARENT_PATH
      save_path = os.path.join(config.PARENT_PATH, 'lb_imgs/'+single_ds['gene stage']+' '+tmp_label.replace('/', '-')+'.jpg')
      img_to_save = lb_output[0][0]
      print(np.max(img_to_save))
      print(np.min(img_to_save))
      img_to_save = (img_to_save - np.min(img_to_save)) / (np.max(img_to_save) - np.min(img_to_save))
      skimage.io.imsave(save_path, img_to_save)

      ori_save_path = os.path.join(config.PARENT_PATH, 'lb_imgs/'+single_ds['gene stage']+' '+tmp_label.replace('/', '-')+'-ori.jpg')
      ori_img_to_save = i[max_images[single_img_idx]]
      skimage.io.imsave(ori_save_path, ori_img_to_save)

# process for one group
def test(output_layer, group_idx=282):
  """The func do following step by step.
  1. get the data
  2. given a group, do inference.
  3. do deconv
  """
  config = ModelConfig(max_sequence_length=15, annotation_number=10,
                       adaption_layer_filters=[1024, 1024, 512],
                       net_global_dim=[128, 256],
                       net_max_features_nums=512,
                       stages=[6],
                       GRAND_PARENT_PATH='/Users/junr/Documents/prp/pic_data',
                       IMAGE_PARENT_PATH='/Users/junr/Documents/prp/pic_data/pic_data'
                       )
  config.finish()
  # 1. get data
  Ds = Dataset(config)
  binarizer = MultiLabelBinarizer(classes=Ds.vocab)
  single_ds = Ds.raw_dataset[group_idx]

  inference(output_layer, single_ds, binarizer, config)

def run_for_one_label(label):
  """Run the deconv model, for specified label.
  Like: ventral nerve cord, embryonic dorsal epidermis
      faint ubiquitous, embryonic brain,
      embryonic head epidermis
  """
  # for now, just test train dataset
  config = ModelConfig(max_sequence_length=15, annotation_number=10,
                       adaption_layer_filters=[1024, 1024, 1024],
                       net_global_dim=[64, 256],
                       net_max_features_nums=512,
                       stages=[6],
                       )
  config.finish()
  # 1. get data
  Ds = Dataset(config)
  binarizer = MultiLabelBinarizer(classes=Ds.vocab)
  new_ds = []
  for ele in Ds:
    labels = Ds['annot']
    if label in labels:
      new_ds.append({'annot': labels,
                     'urls': ele['urls']})

  # next, build model for specified label


  images = tf.placeholder(dtype=tf.float32, shape=[config.max_sequence_length, 128, 320, 3],
                          name="image_feed")
  targets = tf.placeholder(dtype=tf.float32,
                           shape=[1, config.annotation_number],  # batch_size
                           name="input_feed")
  # deconv placeholder
  feature = tf.placeholder(tf.float32, shape=[1, config.annotation_number])
  m1 = tf.placeholder(tf.float32, shape=[1, 128, 320, 64])
  m2 = tf.placeholder(tf.float32, shape=[1, 64, 160, 128])
  m3 = tf.placeholder(tf.float32, shape=[1, 32, 80, 256])
  net_max_feature_mask_np = tf.placeholder(tf.float32, shape=[1, 16, 40, config.net_max_features_nums])

  # forward model
  adaption_output, net1m, net2m, net3m, net_max_feature = model(images, config)
  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)
  # deconv model
  lb_img = deconv_model(feature, [m1, m2, m3, net_max_feature_mask_np], 'adaption/adaption_output', config)
  with tf.InteractiveSession() as sess:
    # restore
    saver.restore(sess, '/Users/junr/Documents/prp/pic_data/model/6-10/max.ckpt')

    # run the whole model
    for single_ds in new_ds:
      images_path = single_ds['urls']
      labels = single_ds['annot']
      l = binarizer.fit_transform([labels])
      vocab = binarizer.classes
      imgs = seq(images_path) \
        .map(lambda path: skimage.io.imread(path)) \
        .map(lambda img: skimage.img_as_float(img)) \
        .map(lambda img: (img - 0.5) * 2) \
        .list()

      imgs = np.array(imgs)
      temp_image = imgs
      for tmp_idx in range(int(config.max_sequence_length / imgs.shape[0] + 1)):
        temp_image = np.concatenate((temp_image, imgs))
      i = temp_image[:config.max_sequence_length]
      # run inference(forward)
      infer_output = sess.run([adaption_output, net1m, net2m, net3m, net_max_feature],
                              feed_dict={
                                images: i,
                                targets: l
                              })
      # do some evaluation
      target = ''
      prediction = ''
      pred_result = np.argsort(infer_output[0][0])
      for s in vocab[l[0] == 1.]:
        target += (s + ' \n')
      print('Target: %s' % target)
      prob = sigmoid(infer_output[0])
      for s in range(5):
        prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                       + str(prob[0][pred_result[-(s + 1)]]) + '\n')
      print('Prediction: %s ' % prediction)

      # run deconv
      max_infer_out_idx = np.argmax(infer_output[0])
      max_infer_out = np.zeros((1, len(infer_output[0])))
      max_infer_out[0, max_infer_out_idx] = 1
      lb_output = sess.run([lb_img], feed_dict={
        feature: max_infer_out,
        net_max_feature_mask_np: infer_output[4],
        m1: infer_output[1],
        m2: infer_output[2],
        m3: infer_output[3]
      })

      # save image at config.PARENT_PATH
      save_path = os.path.join(config.PARENT_PATH, single_ds['gene stage'])
      skimage.io.imsave(save_path, lb_output[0])
    sess.close()




if __name__ == '__main__':
  test('adaption/adaption_output')