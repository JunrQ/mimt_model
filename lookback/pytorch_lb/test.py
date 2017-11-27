from skimage.io import imsave
from .ops import *
# %matplotlib inline
img_path = 'cat.jpeg'
img = plt.imread(img_path)
img = skimage.transform.resize(img, [256, 256])

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.parameters():
  param.requires_grad = False

# conv1 part
last_images = img2img_list(img)
conv1_w = vgg16.features[0].weight
weight_conv1_np = get_weight_from_pytorch(conv1_w)
class test_model_conv1(nn.Module):
  def __init__(self):
    super(test_model_conv1, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[0],
    vgg16.features[1]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs.astype(np.float32)
inputs = inputs[None, ...]
model_cnn = test_model_conv1()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv1_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv1 = normalize_img(my_imshow(tmp_images))

# conv2 part
last_images = images
img = outputs_
conv2_w = vgg16.features[2].weight
weight_conv2_np = get_weight_from_pytorch(conv2_w)
class test_model_conv2(nn.Module):
  def __init__(self):
    super(test_model_conv2, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[2],
    vgg16.features[3]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv2()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv2_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv2 = normalize_img(my_imshow(tmp_images))


# pooling 1 part
img = outputs_
class test_model_pool1(nn.Module):
  def __init__(self):
    super(test_model_pool1, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[4]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_pool1()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
images = pool_images_list(img, images)

# conv3 part
last_images = images
img = outputs_
conv3_w = vgg16.features[5].weight
weight_conv3_np = get_weight_from_pytorch(conv3_w)
class test_model_conv3(nn.Module):
  def __init__(self):
    super(test_model_conv3, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[5],
    vgg16.features[6]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv3()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv3_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv3 = normalize_img(my_imshow(tmp_images))

# conv4 part
last_images = images
img = outputs_
conv4_w = vgg16.features[7].weight
weight_conv4_np = get_weight_from_pytorch(conv4_w)
class test_model_conv4(nn.Module):
  def __init__(self):
    super(test_model_conv4, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[7],
    vgg16.features[8]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv4()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv4_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv4 = normalize_img(my_imshow(tmp_images))

# pooling 2 part
img = outputs_
class test_model_pool2(nn.Module):
  def __init__(self):
    super(test_model_pool2, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[9]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_pool2()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
images = pool_images_list(img, images)

# conv5 part
last_images = images
img = outputs_
conv5_w = vgg16.features[10].weight
weight_conv5_np = get_weight_from_pytorch(conv5_w)
class test_model_conv5(nn.Module):
  def __init__(self):
    super(test_model_conv5, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[10],
    vgg16.features[11]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv5()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv5_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv5 = normalize_img(my_imshow(tmp_images))

# conv6 part
last_images = images
img = outputs_
conv6_w = vgg16.features[12].weight
weight_conv6_np = get_weight_from_pytorch(conv6_w)
class test_model_conv6(nn.Module):
  def __init__(self):
    super(test_model_conv6, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[12],
    vgg16.features[13]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv6()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv6_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv6 = normalize_img(my_imshow(tmp_images))
imsave('conv6.bmp', img_to_show_conv6)

# conv7 part
last_images = images
img = outputs_
conv7_w = vgg16.features[14].weight
weight_conv7_np = get_weight_from_pytorch(conv7_w)
class test_model_conv7(nn.Module):
  def __init__(self):
    super(test_model_conv7, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[14],
    vgg16.features[15]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv7()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv7_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv7 = normalize_img(my_imshow(tmp_images))
imsave('conv7.bmp', img_to_show_conv7)

# pool 3 part
img = outputs_
class test_model_pool3(nn.Module):
  def __init__(self):
    super(test_model_pool3, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[16]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_pool3()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
images = pool_images_list(img, images)

# conv8 part
last_images = images
img = outputs_
conv8_w = vgg16.features[17].weight
weight_conv8_np = get_weight_from_pytorch(conv8_w)
class test_model_conv8(nn.Module):
  def __init__(self):
    super(test_model_conv8, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[17],
    vgg16.features[18]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv8()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv8_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv8 = normalize_img(my_imshow(tmp_images))
imsave('conv8.bmp', img_to_show_conv8)

# conv9 part
last_images = images
img = outputs_
conv9_w = vgg16.features[19].weight
weight_conv9_np = get_weight_from_pytorch(conv9_w)
class test_model_conv9(nn.Module):
  def __init__(self):
    super(test_model_conv9, self).__init__()
    self.vgg16 = nn.Sequential(
    vgg16.features[19],
    vgg16.features[20]
    )
  def forward(self, x):
    x = self.vgg16(x)
    return x
inputs = cl2cf(img)
inputs = inputs[None, ...]
model_cnn = test_model_conv9()
inputs = Variable(torch.from_numpy(inputs))
outputs = model_cnn(inputs)
outputs_ = cf2cl(outputs.data.cpu().numpy()[0])
tmp_sum = np.sum(outputs_, axis=(0, 1))
tmp_arg = np.argsort(tmp_sum)[-10:]
images = return_image(img, last_images, outputs_, weight_conv9_np, [1, 1], tmp_arg)
tmp_images = [images[tmp_idx][:, :, tmp_arg] for tmp_idx in range(len(images))]
img_to_show_conv9 = normalize_img(my_imshow(tmp_images))
imsave('conv9.bmp', img_to_show_conv9)








