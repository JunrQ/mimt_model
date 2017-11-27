import tensorflow as tf
import functools

from config import ModelConfig

from train_ops import input_fn, ExamplesPerSecondHook, \
    get_experiment_fn, _model_fn

import time


# def main(run_experiment=0):
"""Main func, run train, eval."""

# Session configuration.
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth=True

# Model config.
model_config = ModelConfig(
    # model arch parameters
    adaption_layer_filters=[1024, 1024, 512],
    adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
    adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
    plus_global_feature=1,
    net_global_dim=[128, 256],
    net_max_features_nums=512,
    proportion={'train': 0.7, 'val': 0.1, 'test': 0.2},
    stages=[6],
    vgg_output_layer='conv4/conv4_2',
    loss_ratio=5.0,
    neg_threshold=0.2,
    pos_threshold=0.9,
    annotation_number=20,
    max_sequence_length=15,
    image_size=(128, 320),
    batch_size=10,
    threshold=0.5)
model_config.finish()
train_steps = 99999
eval_steps = None
run_experiment = 0

model_params = {'config': model_config}
train_input_fn = functools.partial(input_fn, 'train', model_config)
eval_input_fn = functools.partial(input_fn, 'val', model_config)
test_input_fn = functools.partial(input_fn, 'test', model_config)

# Hook.
tensors_to_log = {'learning_rate': 'learning_rate',
                  'loss': 'loss',
                  'global_step': 'global_step',
                  'cross_entropy': 'cross_entropy',
                  #'training_auc': 'training_auc',
                  'wrong_number': 'wrong_number'}

logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=200)

hooks = [logging_hook]

if run_experiment:
  # experiment
  config = tf.contrib.learn.RunConfig(model_dir=model_config.PARENT_PATH)
  # config = config.replace(session_config=sess_config)
  tf.contrib.learn.learn_runner.run(
    get_experiment_fn(train_input_fn, eval_input_fn,
                      train_steps, eval_steps,
                      hooks, model_params), run_config=config)

else:
  config = tf.estimator.RunConfig()
  config = config.replace(model_dir=model_config.PARENT_PATH,
                          )
  config = config.replace(session_config=sess_config)
  classifier = tf.estimator.Estimator(
    model_fn=_model_fn, params=model_params, config=config)

  epoch = 0
  while True:
    epoch += 1
    print("Starting train epoch: %d at time: %s"%(epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ))
    classifier.train(input_fn=train_input_fn,
                     hooks=hooks)

    print("Starting eval epoch: %d at time: %s" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    eval_results = classifier.evaluate(
      input_fn=eval_input_fn,
      steps=eval_steps)
    print(type(eval_results))
    print(eval_results)

