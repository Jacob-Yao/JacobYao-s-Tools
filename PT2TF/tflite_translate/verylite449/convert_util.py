import os
import numpy as np
import time
from itertools import islice

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Add, BatchNormalization, Dropout, Flatten, Layer, Dense, InputLayer
from tensorflow.keras.layers import AvgPool2D, Conv2D, DepthwiseConv2D, ReLU, ZeroPadding2D, Softmax
from tensorflow.keras.layers import ReLU, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.activations import tanh

from networks import ResnetConditionHRVeryLite as NetPt
from networks import ResnetBlock as ResnetBlock_Pt
from network_tf2 import ResnetConditionHRVeryLite as NetTf
from network_tf2 import ReflectionPadding2D, ReplicationPadding2D, Tanh, ResnetBlock
import cv2

PT_ENTERYS = (nn.Sequential, ResnetBlock_Pt) 
TF_IGNORES = (Flatten, ReLU, MaxPool2D, ReflectionPadding2D, ReplicationPadding2D, \
            Tanh, UpSampling2D, ZeroPadding2D, InputLayer, Concatenate)
TF_ENTERYS = (Sequential, ResnetBlock)
TF_PARAM_LAYERS = (Conv2D, BatchNormalization)

# for module_name in islice(model._modules, 1, None):


def crawl_pt(model, indent=''):
    weights_pt = []
    force_print = 0
    for module_name in model._modules:
        module = model._modules[module_name]
        weights = []
        param_print = 0
        if isinstance(module, nn.Conv2d):
            # Conv2d weights conversion
            # 2310/2301/detach：weights layout for depthwise conv and conv are different in tf
            if module.groups == 1: 
                weights.append(module.weight.permute(2,3,1,0).detach().numpy()) #depthwise
            else:
                weights.append(module.weight.permute(2,3,0,1).detach().numpy())
            if hasattr(module, 'bias') and module.bias is not None:
                weights.append(module.bias.detach().numpy())
            weights_pt.append(weights)
            param_print = 1
        elif isinstance(module, nn.Linear):
            # Linear weights conversion
            weights.append(module.weight.permute(1,0).detach().numpy())
            if hasattr(module, 'bias') and module.bias is not None:
                weights.append(module.bias.detach().numpy())
            weights_pt.append(weights)
            param_print = 1
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm weights conversion
            weights.append(module.weight.detach().numpy())
            if hasattr(module, 'bias') and module.bias is not None:
                weights.append(module.bias.detach().numpy())
            weights.append(module.running_mean.detach().numpy())
            weights.append(module.running_var.detach().numpy())
            weights_pt.append(weights)
            param_print = 1
        elif isinstance(module, PT_ENTERYS):
            # enter and iterate
            param_print = 1
            weights = crawl_pt(module, indent='\t'+indent)
            weights_pt += weights
        else:
            param_print = 0
            continue
        if force_print or param_print:
            print(indent+str(type(module)))
    return weights_pt


def crawl_tf(model, weights_pt, idx=0, indent=''):
    force_print = 0
    for layer in model.layers:
        param_print = 0
        if isinstance(layer, TF_IGNORES):
            # layers with no param
            param_print = 0
            continue
        elif isinstance(layer, TF_ENTERYS):
            param_print = 1
            # enter and search
            new_idx = crawl_tf(layer, weights_pt, idx=idx, indent='\t'+indent)
            idx = new_idx
        elif isinstance(layer, TF_PARAM_LAYERS):
            layer.set_weights(weights_pt[idx])
            idx += 1
            param_print = 1
        else:
            raise NotImplementedError('TF layer \"{}\" not classified.'.format(str(type(layer))))
        if force_print or param_print:
            print(indent+str(type(layer)))
    return idx


