import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import Input, Model
from resnet_tf import *
import resnet
from resnet import resnet18
import cv2

print("START TENSORFLOWING")

model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=17)
x = tf.random.uniform((1,224,224,3))
model.build((1,224,224,3))
y = model(x)

print("START PYTORCHING")

pt_device = torch.device('cpu')
model_pt = resnet.resnet18(num_classes=17)
checkpoint = torch.load('./resnet18_latest.pth.tar', map_location=lambda storage, loc: storage)
state_dict = {k.replace('module.',''): v for k,v in checkpoint['state_dict'].items()}
model_pt.load_state_dict(state_dict,strict=True)
model_pt.eval()
model_pt = model_pt.to(pt_device)

weights_pt = []
for module in model_pt.modules():
    print(module)
    weights = []
    if isinstance(module, nn.Conv2d): # conv
        if module.groups == 1: # what is 2310 and 2301, and why detach, as explained in doc, weights layout for depthwise conv and conv are different in tf
            weights.append(module.weight.permute(2,3,1,0).detach().numpy()) #depthwise
        else:
            weights.append(module.weight.permute(2,3,0,1).detach().numpy())
        if hasattr(module, 'bias') and module.bias is not None:
            weights.append(module.bias.detach().numpy())
        weights_pt.append(weights)

    elif isinstance(module, nn.Linear):
        weights.append(module.weight.permute(1,0).detach().numpy())
        if hasattr(module, 'bias') and module.bias is not None:
            weights.append(module.bias.detach().numpy())
        weights_pt.append(weights)

    elif isinstance(module, nn.BatchNorm2d):
        weights.append(module.weight.detach().numpy())
        if hasattr(module, 'bias') and module.bias is not None:
            weights.append(module.bias.detach().numpy())

        weights.append(module.running_mean.detach().numpy())
        weights.append(module.running_var.detach().numpy())
        weights_pt.append(weights)

i = -1
tf_layers = []

#RESNET
for l in model.layers:
    if isinstance(l, Flatten) or isinstance(l,ReLU) or isinstance(l,MaxPool2D):
        continue
    if not isinstance(l,Sequential):
        # print('\tl',l)
        tf_layers.append(l)
        i += 1
        l.set_weights(weights_pt[i])
    elif isinstance(l,Sequential):
        for m in l.layers:
            # print('xxm',m)
            if isinstance(m, BasicBlock):
                for n in m.conv.layers:
                    if not isinstance(n,ReLU) and not isinstance(n,ZeroPadding2D):
                        # print('\tn',n)
                        tf_layers.append(n)
                        i += 1
                        n.set_weights(weights_pt[i])
                if hasattr(m, 'downsample') and m.downsample is not None:
                    for n in m.downsample.layers:
                        if not isinstance(n,ReLU) and not isinstance(n,ZeroPadding2D):
                            # print('\td',n)
                            tf_layers.append(n)
                            i += 1
                            n.set_weights(weights_pt[i])
    else:
        raise NotImplementedError


print(len(tf_layers)) # debug purpose, make sure weight length same
# for i in range(len(tf_layers)):
#     # print(tf_layers[i].weights)
#     # print(tf_layers[i])
#     tf_layers[i].set_weights(weights_pt[i])
# print(len(weights_pt))

def resizeSubtractMean(image, size=224, bgr_mean=(104, 117, 123),pil_std=[0.229, 0.224, 0.225],model='mbv2'):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image -= bgr_mean
    if model == 'resnet18':
        image /= 255
        image = np.ascontiguousarray(image[:,:,::-1]) #cv2pil
        image /= pil_std
    return image

def centerCrop(image):
    height, width, _ = image.shape
    output_size = min(height,width)
    # cropside = 'height' if output_size == width else 'width'
    crop_top = int(round((height - output_size) / 2.))
    crop_left = int(round((width - output_size) / 2.))
    image_t = image[crop_top:crop_top+output_size,crop_left:crop_left+output_size]
    # print(image.shape[:2],image_t.shape[:2],cropside)
    return image_t

x = tf.random.uniform((1, 224,224,3))
x = cv2.imread('image.bmp')
x = centerCrop(x)
xprocessed = resizeSubtractMean(x,model='resnet18')

x = tf.convert_to_tensor(xprocessed, dtype=tf.float32)
x = tf.expand_dims(x , 0)
y_tf = model(x).numpy()

x_pt = torch.from_numpy(x.numpy()).permute(0,3,1,2)
x_pt = x_pt.to(pt_device)
model_pt = model_pt.to(pt_device)

for param in model_pt.parameters():
    if param.is_cuda == True:
        print('here')
y_pt = model_pt(x_pt).detach().numpy()
# y_pt = model_pt(x_pt).detach().permute(0, 2,3,1).numpy()
# print(y_pt.shape)

# print(y_tf[0][0][0][1], y_pt[0][0][0][1])
print('tensorflow: ',y_tf)
print('pytorch: ',y_pt)
print('pytorch - tf: ',np.mean((y_pt-y_tf) ** 2))
# Vgcprint(y.shape)

# model._set_input((1,224,224,3))
# x = tf.random.uniform((1,224,224,3))
# x = tf.random.uniform((1, 224,224,3))
# x = cv2.imread('image.bmp')
x = tf.convert_to_tensor(xprocessed, dtype=tf.float32)
x = tf.expand_dims(x , 0)


tfout = model.predict(x)
print('tflite: ' ,tfout)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('RESNET.tflite', "wb").write(tflite_model)

# print(weights_pt)
    # shapes = []
    # curr_layers = []
    # if hasattr(layer, 'weight'):
    #     shapes.append(layer.weight.shape)
    #     curr_layers.append(layer.weight)

    # if hasattr(layer, 'bias') and layer.bias is not None:
    #     shapes.append(layer.bias.shape)
    #     curr_layers.append(layer.bias)

    # if shapes:
    #     pt_shapes.append(shapes)
    #     pt_layers.append(curr_layers)
    #     print(layer, shapes)

# for module in pt_layers:
#     print(module)
    # if isinstance(module, nn.Conv2d):
    #     weights_pt.append(module.weight.permute(2,3,1,0).detach().numpy())
    #     # weights_pt.append(module.weight.permute(2,3,1,0).detach().numpy())
    #     biases_pt.append(module.bias.detach().numpy())
    #     # print(module.weight.permute(2,3,1,0).shape)
    #     # print(module.bias.shape)
    #     # print(module.weight.shape)
    # if isinstance(module, nn.Linear):
    #     # print(module.weight.permute(1,0).shape)
    #     # print(module.bias.shape)
    #     weights_pt.append(module.weight.permute(1,0).detach().numpy())
    #     biases_pt.append(module.bias.detach().numpy())

# print(weights_pt)
# for tf_layer, pt_layer in zip(tf_layers, pt_layers):
#     tf_layer
#     print(len(tf_layers), len(pt_layers))
# print(len(pt_shapes))
