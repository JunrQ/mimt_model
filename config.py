import os
import pickle

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self,
               adaption_layer_filters=[1024, 1024, 512],
               adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
               adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
               net_global_stride=[2],
               net_global_kernel_size=[[3,3]],
               weight_decay=1e-3,
               plus_global_feature=True,
               net_global_dim=[128, 256],
               net_max_features_nums=[512],
               adaption_fc_layers_num=None,
               adaption_fc_filters=None,
               stages=[6],
               annotation_number=10,
               max_sequence_length=15,
               annot_min_per_group=0,
               only_word=None,
               deprecated_word=None,
               min_annot_num=0,
               vgg_trainable=False,
               vgg_output_layer='conv4/conv4_2',
               loss_ratio=5.0,
               neg_threshold=0.2,
               pos_threshold=0.9,
               proportion={'train': 0.6, 'val': 0.2, 'test': 0.2},
               image_size=(128, 320),
               batch_size=10,
               threshold=0.5,
               buffer_size=256,
               summary_frequency=200,
               save_per_epoch=20,
               SAVE_PATH='model.ckpt',
               save_max_metrics='micro_f1',
               MAX_SAVE_PATH='max.ckpt',
               GRAND_PARENT_PATH='/data/vgg_max_model',
               IMAGE_PARENT_PATH='/home/litiange/pic_data',
               global_conv2d=True,
               vgg_or_resnet='vgg',
               net_max_features_size=[[3,3]],
               net_max_features_stride=[[2,2]],
               output_units=[]
               ):
    """Model hyperparameters and configuration."""

    # model arch parameters
    self.adaption_layer_filters = adaption_layer_filters
    self.adaption_kernels_size = adaption_kernels_size
    self.adaption_layer_strides = adaption_layer_strides
    self.adaption_fc_layers_num = adaption_fc_layers_num
    self.adaption_fc_filters = adaption_fc_filters
    self.weight_decay = weight_decay
    self.net_global_stride = net_global_stride
    self.net_global_kernel_size = net_global_kernel_size
    self.global_conv2d = global_conv2d
    self.net_max_features_size = net_max_features_size
    self.net_max_features_stride = net_max_features_stride
    self.output_units = output_units

    self.vgg_or_resnet = vgg_or_resnet

    self.plus_global_feature = plus_global_feature
    self.net_global_dim = net_global_dim
    self.net_max_features_nums = net_max_features_nums

    # vgg parameters
    self.vgg_trainable = vgg_trainable
    self.vgg_output_layer = vgg_output_layer

    # loss parameters
    self.loss_ratio = loss_ratio
    self.pos_threshold = pos_threshold
    self.neg_threshold = neg_threshold

    # choose stage
    self.stages = stages

    # only the most k lables will be considered
    self.annotation_number = annotation_number

    # max image num in a group, also batch size
    self.max_sequence_length = max_sequence_length

    # min annot number in a group
    self.annot_min_per_group = annot_min_per_group

    # only the annot in only_word will be considered
    self.only_word = only_word

    # remove the specified word in deprecated_word
    self.deprecated_word = deprecated_word

    # if top_k_labels is not None, then the labels less than min_annot_num
    # will not be considered, only work when top_k_labels is None
    self.min_annot_num = min_annot_num

    self.proportion = proportion

    self.image_size = image_size

    self.batch_size = batch_size

    self.threshold = threshold
    # used in dataset = dataset.shuffle(buffer_size=config.buffer_size)
    self.buffer_size = buffer_size

    self.summary_frequency = summary_frequency
    self.save_per_epoch = save_per_epoch
    self.SAVE_PATH = SAVE_PATH
    self.save_max_metrics = save_max_metrics

    self.MAX_SAVE_PATH = MAX_SAVE_PATH
    self.GRAND_PARENT_PATH = GRAND_PARENT_PATH

    # original image parent path
    self.IMAGE_PARENT_PATH = IMAGE_PARENT_PATH

  def finish(self):

    # where to save .pkl, after concatnate images
    # PKL_PATH = r'E:\zcq\codes\pkl'
    self.PKL_PATH = os.path.join(self.GRAND_PARENT_PATH, 'pkl')

    stage_str = ''
    for _s in self.stages:
      stage_str += str(_s)

    self.PARENT_PATH = os.path.join(self.GRAND_PARENT_PATH, 'model/all_stage/' + stage_str + '-' + str(self.annotation_number))
    if not os.path.isdir(self.PARENT_PATH):
      os.mkdir(self.PARENT_PATH)

    # where to save dataset: self.raw_dataset, self.vocab, after load_data
    self.RAW_DATASET_PATH = os.path.join(self.PARENT_PATH, 'raw_dataset.pkl')

    self.VALID_DATASET_PATH = os.path.join(self.PARENT_PATH, 'valid_dataset.pkl')

    self.TEST_DATASET_PATH = os.path.join(self.PARENT_PATH, 'test_dataset.pkl')

    # csvfile path
    self.CSVFILE_PATH = os.path.join('/home/litiange/prp_file', 'csvfile.csv')

    # tensorflow ckpt file, trained weight file
    self.CKPT_PATH = os.path.join('/home/litiange/prp_file', 'vgg_16.ckpt')
    self.RESNET_CKPT_PATH = '/home/fuxiaofeng/Documents/flyexpress/data/resnet_v2_101/resnet_v2_101.ckpt'

    self.SAVE_PATH = os.path.join(self.PARENT_PATH, self.SAVE_PATH)
    self.MAX_SAVE_PATH = os.path.join(self.PARENT_PATH, self.MAX_SAVE_PATH)

    model_config = {'adaption_layer_filters': self.adaption_layer_filters,
                    'net_global_dim': self.net_global_dim,
                    'net_max_features_nums': self.net_max_features_nums,
                    'adaption_kernels_size': self.adaption_kernels_size,
                    'adaption_layer_strides': self.adaption_layer_strides,
                    'net_global_stride': self.net_global_stride,
                    'net_global_kernel_size': self.net_global_kernel_size,
                    'global_conv2d': self.global_conv2d}
    with open(self.PARENT_PATH + '/model_config.pkl', 'wb') as f:
      pickle.dump(model_config, f)

    print("Config: ", model_config)
