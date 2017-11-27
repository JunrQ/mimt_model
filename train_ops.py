

import tensorflow as tf
import skimage.transform
from scipy.misc import imread

from functional import seq
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

from mimt_config import ModelConfig
from input_ops import Dataset, Dataset_0
from ops import _cut_to_max_length, annot2vec
from model_ops import MaxModel

def preprocess_image(img):
  """Do some preprocessing here."""
  img = (img - 0.5) * 2
  return img

def _preprocess_py_function(file_list, label, image_size=[128, 320]):
  # file_list = file_list.strip().split(' ')
  # [NOTE]: unknown error, TypeError: a bytes-like object is required, not 'str'
  imgs = seq(file_list) \
        .map(lambda loc: imread(loc)) \
        .map(lambda img: skimage.img_as_float(img)) \
        .map(lambda img: skimage.transform.resize(img, image_size)) \
        .map(lambda img: preprocess_image(img)) \
        .reduce(lambda img0, img1: np.concatenate(img0[None, :], img1[None, :]))
  label = label[0]
  return np.array(imgs, dtype=np.float32), label

def _preprocess_tf_function(filename, label, image_size=[128, 320]):
  """Read in.
  No preprocess, just convert to float32
  """
  image_string = tf.read_file(filename)
  img = tf.image.decode_bmp(image_string, channels=3)
  img = tf.image.resize_images(img, image_size)
  img = tf.cast(img, tf.float32)
  return img, label

def _normalize_function(img, label):
  img = img / 255.0
  img = img - tf.reduce_mean(img)
  return img, label

def _distort_tf_function(image, label):
  """Perform random distortions on an image.
  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
  """
  with tf.name_scope("distort_color", values=[image]):
    image = tf.image.random_brightness(image, max_delta=10.0)
    image = tf.image.random_saturation(image, lower=0.15, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.01)
    image = tf.image.random_contrast(image, lower=0.15, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
  '''
  with tf.name_scope("random_flip", values=[image]):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
  '''

  return image, label


def test_dataset(ds):
  iterator = ds.make_one_shot_iterator()
  next_element = iterator.get_next()
  sess = tf.Session()
  print(sess.run(next_element))

def _read_fn(filenames, labels, config, train_val_test):
  """Read fn.
  Will raise tf.errors.OutOfRangeError every epoch.
  You can use try, except, to collect some statistics
  (e.g. the validation error) for the epoch. Like:
          # Compute for 100 epochs.
          for _ in range(100):
            sess.run(iterator.initializer)
            while True:
              try:
                sess.run(next_element)
              except tf.errors.OutOfRangeError:
                break

            # [Perform end-of-epoch calculations here.]
  Args:
    filenames: Tensor
    labels: Tensor
  Return:
    dataset: return a dataset, with element: ((15, 128, 320, 3), (1, 10))
            ((max_sequence_length, h, w ,c), (1, num_annotation))
  """
  filenames = tf.constant(filenames)
  labels = tf.constant(labels)
  dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
  # read in, convert to float32
  dataset = dataset.map(lambda f, l: _preprocess_tf_function(f, l, config.image_size),
      num_threads=64)
  # distore for train
  if train_val_test == 'train':
    dataset = dataset.map(lambda img, l: _distort_tf_function(img, l),
                          num_threads=32)
  # divide 255 and minus mean
  dataset = dataset.map(lambda img, l: _normalize_function(img, l), num_threads=16)
  dataset = dataset.batch(config.max_sequence_length)
  dataset = dataset.map(lambda img, l: (img, tf.expand_dims(l[0], 0)))
  
  return dataset

def return_dataset(config, train_val_test):
  """Given config, 'train' or 'val' or 'test' return corresponding
  dataset.
  dataset element: shape((15, 128, 320, 3), (1, 10)) which means:
            ((max_sequence_length, h, w ,c), (1, num_annotation))
  """
  ds = Dataset_0(config)
  train_ds = ds.raw_dataset
  val_ds = ds.valid_dataset
  test_ds = ds.test_dataset
  vocab = ds.vocab
  binarizer = MultiLabelBinarizer(classes=vocab)

  if train_val_test == 'train':
    dataset = train_ds
  elif train_val_test == 'val':
    dataset = val_ds
  elif train_val_test == 'test':
    dataset = test_ds
  else:
    raise ValueError("train_val_test should be 'train', 'val' or 'test'")
  images = []
  labels = []
  for ele in dataset:
    urls = _cut_to_max_length(ele['urls'], config.max_sequence_length)
    images += list(urls)
    label_index = binarizer.fit_transform([ele['annot']]).tolist()
    labels += (label_index * config.max_sequence_length)
  tf_dataset = _read_fn(images, labels, config, train_val_test)
  return tf_dataset, len(dataset), vocab

def test_ds():
  config = ModelConfig()
  config.finish()
  ds = Dataset(config)
  train_ds = ds.raw_dataset
  vocab = ds.vocab
  binarizer = MultiLabelBinarizer(classes=vocab)
  dataset = train_ds
  images = []
  labels = []
  for ele in dataset:
    # urls = _cut_to_max_length(ele['urls'], config.max_sequence_length)
    # # change it to string
    # urls_string = ''
    # for idx in range(config.annot_min_per_group):
    #   urls_string += (urls[idx] + ' ')
    # images.append(urls_string)
    # label_index = binarizer.fit_transform([ele['annot']])
    # labels.append(label_index[0].tobytes())
    urls = _cut_to_max_length(ele['urls'], config.max_sequence_length)
    images += list(urls)
    label_index = binarizer.fit_transform([ele['annot']]).tolist()
    labels += (label_index * config.max_sequence_length)
  tf_dataset = _read_fn(images[:500], labels[:500], config)
  test_dataset(tf_dataset)

def input_fn(train_val_test, config):
  # proprecess data here
  ds, samples = return_dataset(config, train_val_test)
  iterator = ds.make_one_shot_iterator()
  features, labels = iterator.get_next()

  # A dict containing key/value pairs that map feature column names
  # to Tensors containing the corresponding feature data.
  feature_cols = {'images': features}

  # return 1) a mapping of feature columns to Tensors with
  #           the corresponding feature data
  #        2) a Tensor containing labels
  return feature_cols, labels

def _model_fn(features, labels, mode, params):
  """
  Args:
    params: weight_decay
  """
  # Logic to do the following:
  # 1. Configure the model via TensorFlow operations
  # 2. Define the loss function for training/evaluation
  # 3. Define the training operation/optimizer
  # 4. Generate predictions
  # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

  # features: A dict containing the features passed to the model via input_fn.
  # labels: A Tensor containing the labels passed to the model via input_fn.
  #          Will be empty for predict() calls, as these are the values the model will infer.
  # mode: One of the following tf.estimator.ModeKeys
  # params: argument containing a dict of hyperparameters used for training

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  config = params['config']
  # If features contains an n-dimensional Tensor with all your feature data,
  # then it can serve as the input layer. If features contains a dict of feature columns
  # passed to the model via an input function, you can convert it to an input-layer Tensor
  # with the tf.feature_column.input_layer function.
  images = features['images']
  model = MaxModel(images, labels, config, is_training)
  # define loss
  loss = model.cost
  # define prediction
  predictions = model.prediction


  # define train op
  boundaries = [
    1000 * x
    for x in np.array([10, 20, 80], dtype=np.int64)
  ]
  staged_lr = [x for x in [1e-2, 1e-3, 1e-4, 1e-5]]
  learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                              boundaries, staged_lr)
  # Create a nicely-named tensor for logging
  learning_rate = tf.identity(learning_rate, name='learning_rate')
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(
    loss=loss, global_step=tf.train.get_global_step())

  # define metrics
  metrics = {
    'FN': tf.metrics.false_negatives(labels, predictions),
    'auc': tf.contrib.metrics.streaming_auc(predictions, labels),
    'accuracy': tf.contrib.metrics.streaming_accuracy(predictions, labels),
    'wrong number': tf.contrib.metrics.streaming_mean(model.wrong_number)
  }

  return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op=train_op, eval_metric_ops=metrics)

# create experiment
def get_experiment_fn(train_input_fn, eval_input_fn, train_steps, eval_steps,
                      train_hooks, params):
  """Returns an Experiment function.
  Experiments perform training on several workers in parallel,
  in other words experiments know how to invoke train and eval in a sensible
  fashion for distributed training.
  """
  def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    classifier = tf.estimator.Estimator(model_fn=_model_fn,
                                        params=params,
                                        config=run_config)
    # Create experiment.
    experiment = tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps
        # eval_hooks= )
    )
    # Adding hooks to be used by the estimator on training mode.
    experiment.extend_train_hooks(train_hooks)
    return experiment
  return _experiment_fn

class VggInit(session_run_hook.SessionRunHook):
  """Hook to restore pre-trained vgg."""
  def __init__(self, ckpt_path):
    self.ckpt_path = ckpt_path



class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """
  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None):
    """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)


