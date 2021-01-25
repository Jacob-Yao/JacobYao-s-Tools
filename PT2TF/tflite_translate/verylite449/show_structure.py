import torch
import torch.nn as nn
from networks import ResnetConditionHRVeryLite as NetTorch
from network_tf import ResnetConditionHRVeryLite as NetTf

import tensorflow as tf

model = NetTorch(input_nc=(3,3,1,4), ngf=8, nf_part=8, output_nc=4, n_blocks1=1, n_blocks2=1, norm_layer=nn.BatchNorm2d)
if 0:
    print(model)
    pth = torch.load('netG_epoch_14.pth', map_location=torch.device('cpu'))
    print(type(pth))
    for item in pth.keys():
        print('name: '+item+' '*(60-len(item))+'\t size:'+str(pth[item].shape))



model_tf = NetTf(input_nc=(3,3,1,4), ngf=8, nf_part=8, output_nc=4, n_blocks1=1, n_blocks2=1)
if 1:
    x1, x2, x3 = tf.random.uniform((1,448,448,3)), \
                    tf.random.uniform((1,448,448,3)), \
                    tf.random.uniform((1,448,448,1))
    out = model_tf(x1, x2, x3)
    # model_tf.build([tf.TensorShape([1,448,448,3]), 
    #                 tf.TensorShape([1,448,448,3]), 
    #                 tf.TensorShape([1,448,448,1])])
    # print(model_tf.to_json())
    print(model_tf.summary())
    for l in model_tf.layers:
        print()
