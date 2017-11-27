import os
import pickle


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self,
               vgg_trainable=False,
               stage=6,
               annotation_number=10,
               max_sequence_length=10,
               loss_ratio=5.0,
               neg_threshold=0.1,
               pos_threshold=0.99,
               image_size=(128, 320),
               batch_size=10,
               threshold=0.5,
               buffer_size=512,
               save_per_epoch=20,
               weight_decay=1e-5,
               SAVE_PATH='model.ckpt',
               save_max_metrics='micro_f1',
               MAX_SAVE_PATH='max.ckpt',
               GRAND_PARENT_PATH='/data/vgg_max_model',
               IMAGE_PARENT_PATH='/home/litiange/pic_data'
               ):
    """Model hyperparameters and configuration."""

    # loss parameters
    self.loss_ratio = loss_ratio
    self.pos_threshold = pos_threshold
    self.neg_threshold = neg_threshold
    self.weight_decay = weight_decay
    self.vgg_trainable = vgg_trainable

    self.stage = stage

    # only the most k lables will be considered
    self.annotation_number = annotation_number

    # max image num in a group, also batch size
    self.max_sequence_length = max_sequence_length

    self.image_size = image_size
    self.height = image_size[0]
    self.width = image_size[1]

    self.batch_size = batch_size

    self.threshold = threshold
    # used in dataset = dataset.shuffle(buffer_size=config.buffer_size)
    self.buffer_size = buffer_size

    self.save_per_epoch = save_per_epoch
    self.SAVE_PATH = SAVE_PATH
    self.save_max_metrics = save_max_metrics

    self.MAX_SAVE_PATH = MAX_SAVE_PATH
    self.GRAND_PARENT_PATH = GRAND_PARENT_PATH

    # original image parent path
    self.IMAGE_PARENT_PATH = IMAGE_PARENT_PATH

  def finish(self):
    self.PKL_PATH = os.path.join(self.GRAND_PARENT_PATH, 'pkl')

    self.PARENT_PATH = os.path.join(self.GRAND_PARENT_PATH,
                                    'model/mimt/' + str(self.stage) + '-' + str(self.annotation_number))
    self.PARENT_PATH += '/'
    if not os.path.isdir(self.PARENT_PATH):
      os.mkdir(self.PARENT_PATH)

    # csvfile path
    self.CSVFILE_PATH = os.path.join('/home/litiange/prp_file', 'csvfile.csv')

    # tensorflow ckpt file, trained weight file
    self.CKPT_PATH = os.path.join('/home/litiange/prp_file', 'vgg_16.ckpt')
    self.RESNET_CKPT_PATH = '/home/fuxiaofeng/Documents/flyexpress/data/resnet_v2_101/resnet_v2_101.ckpt'

    self.SAVE_PATH = os.path.join(self.PARENT_PATH, self.SAVE_PATH)
    self.MAX_SAVE_PATH = os.path.join(self.PARENT_PATH, self.MAX_SAVE_PATH)

