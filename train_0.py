import tensorflow as tf
import functools

from config import ModelConfig

from train_ops import input_fn, ExamplesPerSecondHook, \
    get_experiment_fn, _model_fn


def main():
  """Main func, run train, eval."""

  # Session configuration.
  sess_config = tf.ConfigProto()
  sess_config.allow_soft_placement = True

  # Model config.
  model_config = ModelConfig()
  model_config.finish()

  model_params = {'config': model_config}
  train_input_fn = functools.partial(input_fn, 'train', model_config)
  eval_input_fn = functools.partial(input_fn, 'val', model_config)
  test_input_fn = functools.partial(input_fn, 'test', model_config)

  # Hook.
  tensors_to_log = {'learning_rate': 'learning_rate',
                    'loss': 'loss',
                    'cross_entropy': 'cross_entropy',
                    'training_auc': 'training_auc',
                    'wrong_number': 'wrong_number'}

  validation_metrics = {
    "accuracy":
      tf.contrib.learn.MetricSpec(
        metric_fn=tf.contrib.metrics.streaming_accuracy,
        prediction_key="classes"),
    "precision":
      tf.contrib.learn.MetricSpec(
        metric_fn=tf.contrib.metrics.streaming_precision,
        prediction_key="classes"),
    "recall":
      tf.contrib.learn.MetricSpec(
        metric_fn=tf.contrib.metrics.streaming_recall,
        prediction_key="classes")
  }
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # build classifier
  classifier = tf.estimator.Estimator(
    model_fn=_model_fn, params=model_params)

  # Fit model.
  classifier.fit(input_fn=train_input_fn
                 steps=2000,
                 monitors=[validation_monitor])

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(
    x=test_set.data, y=test_set.target)["accuracy"]
  print("Accuracy: {0:f}".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
  y = list(classifier.predict(new_samples))
  print("Predictions: {}".format(str(y)))










