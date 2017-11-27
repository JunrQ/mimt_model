import tensorflow as tf
import functools

import sys
from mimt_config import ModelConfig

from train_ops import return_dataset
from ops import fancy_vocab
from mimt_model import MaxModel
from validation import calcu_metrics

import numpy as np
import os
import datetime

import time

tf.logging.set_verbosity(tf.logging.INFO)


def simple_train(annotation_number=20,
                 max_sequence_length=15,
                 vgg_or_resnet='vgg',
                 epoch=120,
                 stage=5,
                 summary_frequency=500,
                 save_per_epoch=40,
                 SAVE_PATH='model.ckpt',
                 MAX_SAVE_PATH='max.ckpt',
                 vgg_trainable=False,
                 batch_size=5
                 ):
  """Main func, run train, eval."""

  # Session configuration.
  sess_config = tf.ConfigProto()
  sess_config.allow_soft_placement = True
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.allocator_type = "BFC"

  # Model config.
  model_config = ModelConfig(
    stage=stage,  # just for different parameters
    annotation_number=annotation_number,
    max_sequence_length=max_sequence_length,
    save_per_epoch=save_per_epoch,
    SAVE_PATH=SAVE_PATH,
    MAX_SAVE_PATH=MAX_SAVE_PATH,
    loss_ratio=10.0,
    vgg_trainable=vgg_trainable,
    batch_size=batch_size
  )
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
  images = tf.placeholder(dtype=tf.float32, shape=[batch_size, model_config.max_sequence_length, 128, 320, 3])
  labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, model_config.annotation_number])
  is_training = tf.Variable(True, dtype=tf.bool)

  model = MaxModel(images, labels, model_config, is_training)

  loss = model.cost
  prediction = model.prediction
  output = model.output  # output, not prob
  output_prob = model.output_prob

  # define global step
  global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

  # define learning rate

  boundaries = [x for x in np.array([int(train_num / model_config.batch_size * 80),
                                     int(train_num / model_config.batch_size * 110)],
                                    dtype=np.int32)]

  staged_lr = [x for x in [1e-4, 1e-5, 1e-6]]
  learning_rate = tf.train.piecewise_constant(global_step,
                                              boundaries, staged_lr)
  # learning_rate = tf.Variable(initial_value=1e-3, trainable=False, name='learning_rate')

  # Create a nicely-named tensor for logging
  tf.summary.scalar('learning_rate', learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

  grads = optimizer.compute_gradients(loss=loss)
  for i, (g, v) in enumerate(grads):
    if g is not None:
      g = tf.clip_by_norm(g, 5)
      grads[i] = (g, v)  # clip gradients
      tf.summary.histogram('gradient/' + v.name[:-2], g)
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  if vgg_or_resnet == 'vgg':
    # resotre vgg
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    vgg_variables = [v for v in all_vars if
                     (v.name.startswith('vgg_16') and (not ('Ada' in v.name or 'RMSProp' in v.name)))]
    saver = tf.train.Saver(vgg_variables)
  elif vgg_or_resnet == 'resnet':
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    res_variables = [v for v in all_vars if
                     (v.name.startswith('resnet') and (not ('Ada' in v.name or 'RMSProp' in v.name)))]
    saver = tf.train.Saver(res_variables)
  model_saver = tf.train.Saver()

  for tmp_var in all_vars:
    if (not tmp_var.name.startswith('vgg')) and \
        (('weights' in tmp_var.name) or ('biases' in tmp_var.name)) and \
        (not (('Ada' in tmp_var.name) or ('RMS' in tmp_var.name))):
      tf.summary.histogram(tmp_var.name[:-2], tmp_var)
  for tmp_var in all_vars:
    if tmp_var.name.startswith('vgg') and (not (('Ada' in tmp_var.name) or ('RMS' in tmp_var.name))):
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
      tmp_var = [v for v in all_vars if
                 (v.name.startswith('adaption') and not (('Adam' in v.name) or ('RMS' in v.name)))] + vgg_variables
      tmp_saver = tf.train.Saver(tmp_var)
      tmp_saver.restore(sess, model_config.MAX_SAVE_PATH)
    else:
      if vgg_or_resnet == 'vgg':
        tf.logging.info("Restoring trained variables from checkpoint file %s",
                        model_config.CKPT_PATH)
        saver.restore(sess, model_config.CKPT_PATH)  # vgg
      elif vgg_or_resnet == 'resnet':
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
          if images_s.shape[0] != batch_size:
            continue
          # tf.summary.image('train_image', raw_image, max_outputs=10)
          if (total_step > 1) and (total_step % summary_frequency == 0):
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
          logits_list = []

          while True:
            try:
              images_s, labels_s = sess.run(val_next_element)
              if images_s.shape[0] != batch_size:
                continue
              feed_dict = {images: images_s,
                           labels: labels_s}
              predict_, prob_, logits_ = sess.run([prediction, output_prob, output], feed_dict=feed_dict)
              prob_list.append(prob_)
              label_list.append(labels_s)
              logits_list.append(logits_)

            except tf.errors.OutOfRangeError:
              prob_list = np.concatenate(prob_list)
              label_list = np.concatenate(label_list)
              logits_list = np.concatenate(logits_list)

              for tmp_idx in np.random.choice(range(len(prob_list)), 3):
                info_str_t = ''
                info_str_f = ''
                tmp_l = label_list[tmp_idx]
                tmp_p = prob_list[tmp_idx]
                tmp_logit = logits_list[tmp_idx]
                for tmp_ in range(tmp_l.shape[0]):
                  if tmp_l[tmp_] == 1:
                    info_str_t += '%-20s: %f %f\n' % (vocab[tmp_], tmp_p[tmp_], tmp_logit[tmp_])
                  else:
                    info_str_f += '%-20s %f  %f\n' % (vocab[tmp_], tmp_p[tmp_], tmp_logit[tmp_])
                tf.logging.info('\nPositive: \n' + info_str_t)
                tf.logging.info('\nNegative: \n' + info_str_f)

              tf.logging.info(
                "Done validation epoch: %d at time: %s" % (
                  total_epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
              # some thing after val
              metrics_val = calcu_metrics(prob_list, label_list,
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
    sess.run(assing_is_training_false_op)
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
        if images_s.shape[0] != batch_size:
          continue
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

def simple_test(annotation_number=20,
                 max_sequence_length=15,
                 vgg_or_resnet='vgg',
                 epoch=120,
                 stage=5,
                 summary_frequency=500,
                 save_per_epoch=40,
                 SAVE_PATH='model.ckpt',
                 MAX_SAVE_PATH='max.ckpt',
                 vgg_trainable=False,
                 batch_size=5
                 ):
  """Main func, run train, eval."""

  # Session configuration.
  sess_config = tf.ConfigProto()
  sess_config.allow_soft_placement = True
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.allocator_type = "BFC"

  # Model config.
  model_config = ModelConfig(
    stage=stage,  # just for different parameters
    annotation_number=annotation_number,
    max_sequence_length=max_sequence_length,
    save_per_epoch=save_per_epoch,
    SAVE_PATH=SAVE_PATH,
    MAX_SAVE_PATH=MAX_SAVE_PATH,
    loss_ratio=10.0,
    vgg_trainable=vgg_trainable,
    batch_size=batch_size
  )
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
  images = tf.placeholder(dtype=tf.float32, shape=[batch_size, model_config.max_sequence_length, 128, 320, 3])
  labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, model_config.annotation_number])
  is_training = tf.Variable(True, dtype=tf.bool)

  model = MaxModel(images, labels, model_config, is_training)

  loss = model.cost
  prediction = model.prediction
  output = model.output  # output, not prob
  output_prob = model.output_prob

  # define global step
  global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

  # define learning rate

  boundaries = [x for x in np.array([int(train_num / model_config.batch_size * 80),
                                     int(train_num / model_config.batch_size * 110)],
                                    dtype=np.int32)]

  staged_lr = [x for x in [1e-4, 1e-5, 1e-6]]
  learning_rate = tf.train.piecewise_constant(global_step,
                                              boundaries, staged_lr)
  # learning_rate = tf.Variable(initial_value=1e-3, trainable=False, name='learning_rate')

  # Create a nicely-named tensor for logging
  tf.summary.scalar('learning_rate', learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

  grads = optimizer.compute_gradients(loss=loss)
  for i, (g, v) in enumerate(grads):
    if g is not None:
      g = tf.clip_by_norm(g, 5)
      grads[i] = (g, v)  # clip gradients
      tf.summary.histogram('gradient/' + v.name[:-2], g)
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  if vgg_or_resnet == 'vgg':
    # resotre vgg
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    vgg_variables = [v for v in all_vars if
                     (v.name.startswith('vgg_16') and (not ('Ada' in v.name or 'RMSProp' in v.name)))]
    saver = tf.train.Saver(vgg_variables)
  elif vgg_or_resnet == 'resnet':
    all_vars = tf.global_variables()
    for tmp in all_vars:
      print(tmp)
    res_variables = [v for v in all_vars if
                     (v.name.startswith('resnet') and (not ('Ada' in v.name or 'RMSProp' in v.name)))]
    saver = tf.train.Saver(res_variables)
  model_saver = tf.train.Saver()

  for tmp_var in all_vars:
    if (not tmp_var.name.startswith('vgg')) and \
        (('weights' in tmp_var.name) or ('biases' in tmp_var.name)) and \
        (not (('Ada' in tmp_var.name) or ('RMS' in tmp_var.name))):
      tf.summary.histogram(tmp_var.name[:-2], tmp_var)

  # is_training op
  assing_is_training_true_op = tf.assign(is_training, True)
  assing_is_training_false_op = tf.assign(is_training, False)

  with tf.Session(config=sess_config) as sess:
    # initial model
    init = tf.global_variables_initializer()
    # init = tf.initialize_variables([i for i in all_vars if not i.name.startswith('vgg')])
    sess.run(init)
    # print(model_config.MAX_SAVE_PATH)
    vgg_variables = [v for v in all_vars if
                     (v.name.startswith('vgg_16') and (not ('Ada' in v.name or 'RMSProp' in v.name)))]
    latest_cp = tf.train.latest_checkpoint(model_config.PARENT_PATH)
    tf.logging.info("Using previously saved model %s" % (latest_cp))
    tmp_var = [v for v in all_vars if
                 (v.name.startswith('adaption') and not (('Adam' in v.name) or ('RMS' in v.name)))] + vgg_variables
    tmp_saver = tf.train.Saver(tmp_var)
    tmp_saver.restore(sess, latest_cp)

    # test
    sess.run(test_iterator.initializer)
    sess.run(assing_is_training_false_op)
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
        if images_s.shape[0] != batch_size:
          continue
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
  with tf.device("/gpu:1"):
    result = simple_train(annotation_number=10,
                          epoch=120,
                          summary_frequency=100,
                          save_per_epoch=40,
                          SAVE_PATH='model.ckpt',
                          MAX_SAVE_PATH='max.ckpt')


