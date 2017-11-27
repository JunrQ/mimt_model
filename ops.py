import numpy as np
from tensorflow.python.training import device_setter
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
import collections
import six

def _cut_to_max_length(url_list, max_len):
  url_list = np.tile(url_list, int(max_len / len(url_list)) + 1)[:max_len]
  np.random.shuffle(url_list)
  return url_list

def fancy_vocab(vocab_list, num_a_line=3):
  """Given a list of words, return a str."""
  tmp_count = 1
  s = ''
  for word in vocab_list:
    s += "%-35s "%(word)
    tmp_count += 1
    if (tmp_count>1) and (tmp_count % num_a_line == 0):
      s += "\n"
  return s

def annot2vec(label, vocab):
  """
  Given label ,vocab, return one-hot
  """
  label_list = np.zeros(len(vocab), dtype=np.int)
  for w in label:
    try:
      idx = vocab.index(w)
      label_list[idx] = 1
    except ValueError:
      continue
  return label_list

class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.

    Args:
      worker_device: the device to use for computation Ops.
      ps_devices: a list of devices to use for Variable Ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name

def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser

def normalize(imgs):
  min = np.min(imgs)
  max = np.max(imgs)
  return (imgs - min) / (max - min)

def concat_dim_minus1(dim4):
  """Concatenate along first and last dim."""
  if dim4.ndim != 4:
    raise ValueError("Input should has ndim %d but get ndim %d"%(4, dim4.ndim))
  else:
    tmp = dim4[0, :, :, :]
    for idx in range(1, dim4.shape[0]):
      tmp = np.concatenate((tmp, dim4[idx, :, :, :]), axis=0)
    tmp_ = tmp[:, :, 0]
    for idx in range(1, dim4.shape[-1]):
      tmp_ = np.concatenate((tmp_, tmp[:, :, idx]), axis=-1)
  return tmp_