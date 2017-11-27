import tensorflow as tf
import functools

import sys
from config import ModelConfig

from train_ops import return_dataset
from ops import fancy_vocab
from model_ops import MaxModel, ConvModel
from validation import calcu_metrics
import matplotlib.pyplot as plt

import numpy as np
import os
import datetime

import time

tf.logging.set_verbosity(tf.logging.INFO)


def simple_train(stage=[6],
                 model_type='max',
                 annotation_number=20,
                 epoch=3,
                 save_per_epoch=2,
                 SAVE_PATH='model.ckpt',
                 MAX_SAVE_PATH='max.ckpt',
                 summary_frequency=100,
                 net_global_dim=[256, 256],
                 net_global_kernel_size=[[4, 4]],
                 net_global_stride=[(2, 2)],
                 net_max_features_nums=[512, 256],
                 net_max_features_stride=[[4, 4]],
                 net_max_features_size=[[2, 2]],
                 adaption_layer_filters=[512, 512, 512],
                 adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
                 adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
                 output_units=[],
                 global_conv2d=True,
                 weight_decay=1e-5,
                 ):
  """Main func, run train, eval."""

  # Session configuration.
  sess_config = tf.ConfigProto()
  sess_config.allow_soft_placement = True
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.allocator_type = "BFC"

  # Model config.
  model_config = ModelConfig(
    # model arch parameters
    adaption_layer_filters=adaption_layer_filters,
    adaption_kernels_size=adaption_kernels_size,
    adaption_layer_strides=adaption_layer_strides,
    net_global_kernel_size=net_global_kernel_size,
    net_global_stride=net_global_stride,
    net_max_features_stride=net_max_features_stride,
    net_max_features_size=net_max_features_size,
    global_conv2d=global_conv2d,
    plus_global_feature=1,
    output_units=output_units,
    net_global_dim=net_global_dim,
    net_max_features_nums=net_max_features_nums,
    proportion={'train': 0.6, 'val': 0.2, 'test': 0.2},
    stages=stage,
    vgg_output_layer='conv4/conv4_3',
    loss_ratio=32.0,
    neg_threshold=0.3,
    pos_threshold=0.9,
    annotation_number=annotation_number,
    max_sequence_length=10,
    image_size=(128, 320),
    threshold=0.5,
    summary_frequency=summary_frequency,
    save_per_epoch=save_per_epoch,
    save_max_metrics='macro_f1',
    SAVE_PATH=SAVE_PATH,
    MAX_SAVE_PATH=MAX_SAVE_PATH,
    weight_decay=weight_decay,
    vgg_trainable=False)
  model_config.finish()

  validation_metrics = ['mean_average_precision', 'macro_auc', 'micro_auc',
                        'macro_f1', 'micro_f1', 'ranking_mean_average_precision',
                        'coverage', 'ranking_loss', 'one_error']
  save_max_metrics_idx = validation_metrics.index(model_config.save_max_metrics)
  last_save_metrics = 0

  train_ds, train_num, vocab = return_dataset(model_config, 'train')
  vocab = np.array(vocab)
  val_ds, val_num, _ = return_dataset(model_config, 'val')
  test_ds, test_num, _ = return_dataset(model_config, 'test')

  vocab_str = fancy_vocab(vocab)

  tf.logging.info("\n%-20s%s\n" \
                  "%-20s%d\n" \
                  "%-20s%d\n" \
                  "%-20s%d\n" % ('Dataset', 'Number of groups',
                                 'train', train_num,
                                 'validation', val_num,
                                 'test', test_num) + \
                  "Vocabulary: %d\n" % len(vocab) + "%s" % vocab_str)

  tf.logging.info("model_dir: %s" % model_config.PARENT_PATH)

  train_iterator = train_ds.make_initializable_iterator()
  val_iterator = val_ds.make_initializable_iterator()
  test_iterator = test_ds.make_initializable_iterator()

  train_next_element = train_iterator.get_next()
  val_next_element = val_iterator.get_next()
  test_next_element = test_iterator.get_next()

  # build placeholder
  images = tf.placeholder(dtype=tf.float32, shape=[model_config.max_sequence_length, 128, 320, 3])
  labels = tf.placeholder(dtype=tf.float32, shape=[1, model_config.annotation_number])
  is_training = tf.Variable(True, dtype=tf.bool)

  raw_image = images / 2 + 0.5
  tf.summary.image('input_image', raw_image, max_outputs=10)

  # build model
  if model_type == 'max':
    tf.logging.info('Max model built')
    model = MaxModel(images, labels, model_config, is_training)
  elif model_type == 'conv':
    tf.logging.info('Conv model built')
    model = ConvModel(images, labels, model_config, is_training)
  loss = model.cost
  prediction = model.prediction
  output = model.output  # output, not prob
  output_prob = model.output_prob

  # define global step
  # global_step = tf.Variable(
  #       initial_value=0,
  #       name="global_step",
  #       trainable=False,
  #       collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])


  # define learning rate
  # boundaries = [x for x in np.array([2700*2, 2700*50, 2700*60], dtype=np.int32)
  # ]
  # staged_lr = [x for x in [1e-3, 1e-4, 1e-5, 1e-6]]
  # learning_rate = tf.train.piecewise_constant(global_step,
  #                                             boundaries, staged_lr)

  # Create a nicely-named tensor for logging
  # tf.summary.scalar('learning_rate', learning_rate)
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # grads = optimizer.compute_gradients(loss=loss)
  # for i, (g, v) in enumerate(grads):
  #   if g is not None:
  #     grads[i] = (tf.clip_by_norm(g, 10), v)  # clip gradients
  # train_op = optimizer.apply_gradients(grads, global_step=global_step)

  optimizer = tf.train.AdamOptimizer(1e-4)
  train_op = optimizer.minimize(
    loss=loss
  )

  if model_config.vgg_or_resnet == 'vgg':
    # resotre vgg
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    vgg_variables = [v for v in all_vars if v.name.startswith('vgg_16')]
    saver = tf.train.Saver(vgg_variables)
  elif model_config.vgg_or_resnet == 'resnet':
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    res_variables = [v for v in all_vars if v.name.startswith('resnet')]
    saver = tf.train.Saver(res_variables)
  model_saver = tf.train.Saver()

  for tmp_var in all_vars:
    if (not tmp_var.name.startswith('vgg')) and ('kernel' in tmp_var.name) and not ('Adam' in tmp_var.name):
      tf.summary.histogram(tmp_var.name[:-2], tmp_var)

  # summary op
  validation_tensor = []
  for tmp_name in validation_metrics:
    var = tf.Variable(0.0, name='validation/' + tmp_name, dtype=tf.float32)
    tf.summary.scalar('validation/' + tmp_name, var)
    validation_tensor.append(var)
  merged_summary_op = tf.summary.merge_all()

  # is_training op
  assing_is_training_true_op = tf.assign(is_training, True)
  assing_is_training_false_op = tf.assign(is_training, False)

  with tf.Session(config=sess_config) as sess:
    # initial model
    init = tf.global_variables_initializer()
    # init = tf.initialize_variables([i for i in all_vars if not i.name.startswith('vgg')])
    sess.run(init)
    # print(model_config.MAX_SAVE_PATH)
    if os.path.isfile(model_config.MAX_SAVE_PATH + '.data-00000-of-00001'):
      tf.logging.info("Using previously saved model %s" % (model_config.MAX_SAVE_PATH))
      tmp_var = [v for v in all_vars if (v.name.startswith('adaption') and not 'Adam' in v.name)] + vgg_variables
      tmp_saver = tf.train.Saver(tmp_var)
      tmp_saver.restore(sess, model_config.MAX_SAVE_PATH)
    else:
      if model_config.vgg_or_resnet == 'vgg':
        tf.logging.info("Restoring trained variables from checkpoint file %s",
                        model_config.CKPT_PATH)
        saver.restore(sess, model_config.CKPT_PATH)  # vgg
      elif model_config.vgg_or_resnet == 'resnet':
        tf.logging.info("Restoring trained variables from checkpoint file %s",
                        model_config.RESNET_CKPT_PATH)
        saver.restore(sess, model_config.RESNET_CKPT_PATH)  # vgg
    # summary filewrite
    summary_writer = tf.summary.FileWriter(model_config.PARENT_PATH, sess.graph)
    # total_step
    total_step = 0
    total_epoch = 0
    for _ in range(epoch):
      sess.run(train_iterator.initializer)
      sess.run(val_iterator.initializer)

      total_epoch += 1
      if (total_step > 1) and (total_epoch % model_config.save_per_epoch == 0):
        tf.logging.info(
          "Save model with global_step: %d at path: %s" % (total_step, model_config.SAVE_PATH + '-' + str(total_step)))
        model_saver.save(sess, model_config.SAVE_PATH, global_step=total_step)
      tf.logging.info(
        "Starting train epoch: %d at time: %s" % (total_epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

      # assign is_training true
      sess.run(assing_is_training_true_op)
      # record time
      starttime = datetime.datetime.now()
      while True:
        total_step += 1
        try:
          images_s, labels_s = sess.run(train_next_element)
          # tf.summary.image('train_image', raw_image, max_outputs=10)
          if (total_step > 1) and (total_step % model_config.summary_frequency == 0):
            _, loss_, summary_str, prob_, output_ = sess.run([train_op, loss, merged_summary_op, output_prob, output],
                                                             feed_dict={images: images_s,
                                                                        labels: labels_s})
            # tf.logging.info(str(output_))
            summary_writer.add_summary(summary_str, total_step)
          else:
            _, loss_, output_ = sess.run([train_op, loss, output],
                                         feed_dict={images: images_s,
                                                    labels: labels_s})
            # print(output_)
        except tf.errors.OutOfRangeError:
          # log time spent
          endtime = datetime.datetime.now()
          tf.logging.info(
            "Epoch %d total_step %d takes %d seconds" % (total_epoch, total_step, (endtime - starttime).seconds))
          sess.run(assing_is_training_false_op)

          # validation
          tf.logging.info(
            "Starting validation epoch: %d at time: %s" % (
            total_epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
          # list to store labels, prediction, prob
          prob_list = []
          label_list = []

          while True:
            try:
              images_s, labels_s = sess.run(val_next_element)
              feed_dict = {images: images_s,
                           labels: labels_s}
              predict_, prob_, logits_ = sess.run([prediction, output_prob, output], feed_dict=feed_dict)
              prob_list.append(prob_[0])
              label_list.append(labels_s[0])

            except tf.errors.OutOfRangeError:
              '''
              for tmp_idx in range(len(prob_list)):
                info_str = ''
                tmp_l = label_list[tmp_idx]
                tmp_p = prob_list[tmp_idx]
                for tmp_ in range(len(tmp_l)):
                  if tmp_l[tmp_] == 1:
                    info_str += '%s %f\n' % (vocab[tmp_], tmp_p[tmp_])
                print(info_str)
              '''

              tf.logging.info(
                "Done validation epoch: %d at time: %s" % (
                total_epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
              # some thing after val
              metrics_val = calcu_metrics(np.array(prob_list), np.array(label_list, dtype=np.float32),
                                          validation_metrics, model_config.threshold)
              # add summary
              for idx in range(len(validation_tensor)):
                var = validation_tensor[idx]
                assign_op = var.assign(metrics_val[idx])
                sess.run(assign_op)
              # print result
              result = ''
              for idx in range(len(validation_metrics)):
                result += '%-10s : %f \n' % (validation_metrics[idx], metrics_val[idx])
              print(result)
              # test if better
              save_metrics = metrics_val[save_max_metrics_idx]
              if save_metrics > last_save_metrics:
                # tmp_path = os.path.join(model_config.PARENT_PATH, model_config.save_max_metrics)
                tf.logging.info(
                  "Save model with better metrics\n : %s : %s at path: %s" % (
                    model_config.save_max_metrics, save_metrics, model_config.MAX_SAVE_PATH))
                model_saver.save(sess, model_config.MAX_SAVE_PATH)
                last_save_metrics = save_metrics
              break
          break

    # test
    sess.run(test_iterator.initializer)
    tf.logging.info(
      "Starting test at time: %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # list to store labels, prediction, prob
    pred_list = []
    prob_list = []
    label_list = []
    logits_list = []
    while True:
      try:
        images_s, labels_s = sess.run(test_next_element)
        feed_dict = {images: images_s,
                     labels: labels_s}
        predict_, prob_, logits_ = sess.run([prediction, output_prob, output], feed_dict=feed_dict)
        pred_list.append(predict_[0])
        prob_list.append(prob_[0])
        label_list.append(labels_s[0])
        logits_list.append(logits_[0])

      except tf.errors.OutOfRangeError:
        tf.logging.info(
          "Done test epoch: %d at time: %s" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # some thing after val
        metrics_val = calcu_metrics(np.array(prob_list), np.array(label_list),
                                    validation_metrics, model_config.threshold)
        # print result
        result = ''
        for idx in range(len(validation_metrics)):
          result += '%-10s : %f \n' % (validation_metrics[idx], metrics_val[idx])
        print(result)
        break
  return metrics_val


if __name__ == '__main__':
  stage = 5
  for annotation_number in [10, 20, 30, 40, 50, 60]:
    print("Traing with gpu: %d, stage: %d, annotation_number: %d" % (1, stage, annotation_number))
    with tf.device("/gpu:1"):
      result = simple_train(stage=[stage],
                            annotation_number=annotation_number,
                            epoch=120,
                            save_per_epoch=40,
                            SAVE_PATH='model.ckpt',
                            MAX_SAVE_PATH='max.ckpt')


