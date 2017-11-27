import numpy as np
import math
import skimage.transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import sys

from functional import seq

def filter2img(last_images, input_h, input_w, filters, start_h, start_w, sensitivity, tmp_arg):
  """
  Args:
    start_h, start_w: left top coordinate
    tmp_arg: only cal output filters in tmp_arg, others set to 0
  """
  filter_h, filter_w, input_filters, output_filters = filters.shape
  img_h, img_w, _ = last_images[0].shape
  img_to_return = np.zeros((img_h+filter_h-1, img_w+filter_w-1, output_filters))
  tmp_image = np.zeros((img_h+filter_h-1, img_w+filter_w-1, output_filters))
  for tmp_h in range(filter_h):
    for tmp_w in range(filter_w):
      if (start_h + tmp_h < 0) or (start_h + tmp_h >= input_h):
        continue
      if (start_w + tmp_w < 0) or (start_w + tmp_w >= input_w):
        continue
      multiply_image = last_images[(start_h+tmp_h)*input_w+start_w + tmp_w]
      if np.sum(multiply_image) == 0:
        continue
      for tmp_filter in tmp_arg:
        if sensitivity[tmp_filter] == 0:
          continue
        tmp_conv = filters[tmp_h, tmp_w, :, tmp_filter] * multiply_image
        try:
          tmp_image[tmp_h:(tmp_h+img_h), tmp_w:(tmp_w+img_w), tmp_filter] = np.sum(tmp_conv, axis=-1)
        except ValueError:
          print(tmp_conv.shape)
          print(tmp_image[tmp_h:(tmp_h+img_h), tmp_w:(tmp_w+img_w), tmp_filter].shape)
          print(tmp_image.shape)
          print(img_h, img_w)
          print(tmp_h, tmp_w)
          sys.exit(1)
        img_to_return = np.add(img_to_return, tmp_image)
  return img_to_return

def img2img_list(img):
  """
  Args:
    img: shape [h, w, 3]
  """
  img_list = []
  h, w, _ = img.shape
  for tmp_h in range(h):
    for tmp_w in range(w):
      img_list.append(img[tmp_h, tmp_w, :][None, None, ...])
  return img_list

# def ones_img_list(shape):
#   return np.ones((1, 1, shape), dtype=np.float32)

def return_image(inputs, last_images, outputs, filters, strides, tmp_arg):
  """
  Only support "SAME" padding.
  Args:
    inputs: ndarray with shape (heigth, width, channels)
    last_images: list whose element is ndarray of images return by last layer,
                 element shape: [h, w, c]
    filters: list, e.g. [3, 3, input_filters, output_filters]
    strides: list, e.g. [1, 1]
  Returns:
    images
  """
  images = []
  h, w, c = inputs.shape
  if len(last_images) != h*w:
    raise ValueError("last_images should have length equal to size of inputs, but %d != %d*%d"%(len(last_images), h, w))
  if (h % strides[0] + w % strides[1]) != 0:
    raise ValueError("shape: (%d, %d) can not be divided exactly by strides: (%d, %d)"%(h, w, strides[0], strides[1]))

  # first: padding
  filters_shape = filters.shape
  new_height = math.ceil(h / strides[0])
  new_width = math.ceil(w / strides[1])
  pad_needed_height = (new_height - 1) * strides[0] + filters_shape[0] - h
  pad_needed_width = (new_width - 1) * strides[1] + filters_shape[1] - w
  pad_top = pad_needed_height // 2
  # pad_bottom = pad_needed_height - pad_top
  pad_left = pad_needed_width // 2
  # pad_right = pad_needed_width - pad_left
  padded_image = np.zeros((new_height+pad_needed_height, new_width+pad_needed_width, c))
  padded_image[pad_top:pad_top+h, pad_left:pad_left+w, :] = inputs

  # second: get receptive field and do conv2d
  receptive_field_list = []
  for tmp_h in range(0, new_height):
    for tmp_w in range(0, new_width):
      sensitivity = outputs[tmp_h, tmp_w, :]
      tmp_image = filter2img(last_images, h, w, filters, tmp_h*strides[0]-pad_top, tmp_w*strides[1]-pad_left, sensitivity, tmp_arg)
      # receptive_field_list.append({'img': tmp_image,
      #                              'sensitivity': sensitivity})
      images.append(tmp_image)
  # third:
  return images

def img_to_mask(img):
  """Given a 2 dim img, return 2x2 pooling mask."""
  for i in [0, 1]:
    if img.shape[i] % 2 != 0:
      raise ValueError("shape[%d] = %d is not an even" % (i, img.shape[i]))
  mask = np.zeros_like(img)
  for idx_0 in range(0, img.shape[0], 2):
    for idx_1 in range(0, img.shape[1], 2):
      max_idx = np.argmax(img[idx_0:idx_0+2, idx_1:idx_1+2].flatten())
      mask[idx_0 + max_idx//2, idx_1 + max_idx%2] = 1
  return mask

def pool_images_list(img, images_list):
  """Given img(shape: h, w, c), corrsponding images list, return images list after mask."""
  def _func(img, m):
    """Given img list and mask, return masked img.
    e.g.: img is a list: [array(2, 2), array(2, 2)]
          maks: array([1, 0])
          return [array(2, 2)]
    """
    m_f = m.flatten()
    return [img[idx] for idx in range(len(img)) if m_f[idx] == 1]

  imgs = [img[:, :, c] for c in range(img.shape[-1])]
  imgs_list = [list(map(lambda x: x[:, :, idx], images_list)) for idx in range(images_list[0].shape[-1])]
  imgs_ = zip(imgs, imgs_list)
  img_list_return = seq(imgs_) \
                    .map(lambda x: [x[0], x[1], img_to_mask(x[0])]) \
                    .map(lambda x: _func(x[1], x[2])) \
                    .to_list()
  img_list_return = [np.concatenate([img_list_return[idx][idx_1][..., None] for idx in range(len(img_list_return))], axis=-1)
                                       for idx_1 in range(len(img_list_return[0]))]
  return img_list_return


def flatten_image_channels(img):
  img_return = [img[..., tmp_idx] for tmp_idx in range(img.shape[-1])]
  return img_return

def my_imshow(images, top_k=30):
  """Return top_k images, ordered by number of nonzeros"""
  imgs = seq(images) \
      .map(lambda x: flatten_image_channels(x)) \
      .flat_map(lambda x: x) \
      .map(lambda x: [x, np.sum(x>0)]) \
      .filter(lambda x: x[1] > 0) \
      .to_list()
  arg_idx = np.argsort([imgs[idx][1] for idx in range(len(imgs))])
  imgs_return = [imgs[idx][0] for idx in arg_idx[-top_k:]]
  return np.concatenate(imgs_return, axis=1)

def get_weight_from_pytorch(conv1_w):
  """Make shape [output_filters, input_filters, h, w] to [h, w, input_filters, output_filters]"""
  conv1_weight_np = list(map(lambda x: x.data.numpy(), list(conv1_w)))
  change_dim_weight_np = []
  for tmp_weight in conv1_weight_np:
    dim3_weight = [tmp_weight[tmp_idx, ...][..., None] for tmp_idx in range(tmp_weight.shape[0])]
    change_dim_weight_np.append(np.concatenate(dim3_weight, axis=-1)[..., None])
  weight = np.concatenate(change_dim_weight_np, axis=-1)
  return weight

def cf2cl(img):
  """Channel first to channel last."""
  img_to_return = [img[tmp_idx, ...][..., None] for tmp_idx in range(img.shape[0])]
  return np.concatenate(img_to_return, axis=-1)

def cl2cf(img):
  """Channel last to channel first."""
  img_to_return = [img[..., tmp_idx][None, ...] for tmp_idx in range(img.shape[-1])]
  return np.concatenate(img_to_return, axis=0)

def normalize_img(img):
  min = np.min(img)
  max = np.max(img)
  return (img - min) / (max - min)

# def conv_part(last_images, img, w, )

def test():
  img_path = 'cat.jpeg'
  img = plt.imread(img_path)
  img = skimage.transform.resize(img, [256, 256])
  # last_images = img2img_list(img)
  last_images = [np.ones((1, 1, 3), dtype=np.float32)] * (256**2)

  vgg16 = models.vgg16(pretrained=True)
  conv1_w = vgg16.features[0].weight

  for param in vgg16.parameters():
    param.requires_grad = False

  weight_conv1_np = get_weight_from_pytorch(conv1_w)
  # bias_conv1_np = np.array(list(map(lambda x: x.data.numpy(), list(conv1_b))))

  class test_model(nn.Module):
    def __init__(self):
      super(test_model, self).__init__()
      self.vgg16 = nn.Sequential(
        vgg16.features[0],
        vgg16.features[1]
      )
    def forward(self, x):
      x = self.vgg16(x)
      return x

  inputs = cl2cf(img)
  inputs = inputs[None, ...]
  model_cnn = test_model().cuda()
  inputs = Variable(torch.from_numpy(inputs), requires_grad=True).cuda()
  outputs = model_cnn(inputs)

  outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
  tmp_sum = np.sum(outputs_, axis=(0, 1))
  tmp_arg = np.argsort(tmp_sum)[-10:]
  images = return_image(img, last_images, outputs_, weight_conv1_np, [1, 1], tmp_arg)
  tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]

  plt.imshow(my_imshow(tmp_images))






