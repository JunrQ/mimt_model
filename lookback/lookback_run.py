f = 0

if f == 0:
  from lookback.lookback_ops import *

  config = return_a_config()
  output_layer = 'conv1'
  batch_img_list, result, labels, stage = test_lb_vgg(config, output_layer, top_k_activation=5)
  simple_save_result_from_test_lb(batch_img_list, result, labels, stage, config, output_layer)

elif f == 1:
  from lookback.lookback_ops import *
  config = return_a_config()
  output_layer='adaption_conv'
  batch_img_list, result, labels, stage = test_lb_adaption_conv(config, top_k_activation=5)
  simple_save_result_from_test_lb(batch_img_list, result, labels, stage, config, output_layer)

elif f == 2:
  from lookback.lookback_ops import *
  config = return_a_config()
  batch_img_list, i, result, labels, stage = test_lb_whole(config)
  simple_save_result_from_test_lb_whole(batch_img_list, i, result, labels, stage, config)

