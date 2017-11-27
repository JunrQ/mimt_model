# this change avgpool to conv2d
import tensorflow as tf
import sys
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

from train_tf import simple_train

result_path = '/home/litiange/prp_file/model/result/'

gpu = 0
stage = 6

for annotation_number in [30]:
  print("Traing with gpu: %d, stage: %d, annotation_number: %d"%(gpu, stage, annotation_number))
  with tf.device("/gpu:%d"%gpu):
    result = simple_train(stage=[stage],
                 annotation_number=annotation_number,
                 net_max_features_nums=[256, 256],
                 net_max_features_stride=[[5, 5], [5, 5]],
                 net_max_features_size=[[2, 2], [2, 2]],
                 net_global_dim=[256, 128, 128, 256],
                 net_global_kernel_size=[[4, 4], [4, 4], [4, 4]],
                 net_global_stride=[(2, 2), (2, 2), (2, 2)],
                 global_conv2d=True,
                 adaption_layer_filters=[1024, 1024, 512],
                 adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
                 adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
                 epoch=100,
                 summary_frequency=500,
                 save_per_epoch=33,
                 SAVE_PATH='model.ckpt',
                 MAX_SAVE_PATH='max.ckpt')

  with open(result_path+'%s-%s'%(str(stage), str(annotation_number))+'.pkl', 'wb') as f:
        pickle.dump(result, f)
  tf.reset_default_graph()