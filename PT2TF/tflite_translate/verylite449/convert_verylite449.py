import os
import numpy as np
import cv2

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Add, BatchNormalization, Dropout, Flatten, Layer, Dense, InputLayer
from tensorflow.keras.layers import AvgPool2D, Conv2D, DepthwiseConv2D, ReLU, ZeroPadding2D, Softmax
from tensorflow.keras.layers import ReLU, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.activations import tanh

from networks import ResnetConditionHRVeryLite_zeropad as NetPt
from network_tf2 import ResnetConditionHRVeryLite_rewrite as NetTf
from network_tf2 import ReflectionPadding2D, ReplicationPadding2D, Tanh, ResnetBlock

from convert_util import crawl_pt, crawl_tf


# -----------------------------------------------------------
# --------------------- model preparing ---------------------
# -----------------------------------------------------------
print("START PYTORCHING")
pt_device = torch.device('cpu')
model_pt = NetPt(input_nc=(3,3,1,4), ngf=8, nf_part=8, output_nc=4, n_blocks1=1, n_blocks2=1)
# checkpoint = torch.load('./netG_epoch_14.pth', map_location=lambda storage, loc: storage)
# state_dict = {k.replace('module.',''): v for k,v in checkpoint['state_dict'].items()}
# model_pt.load_state_dict(state_dict,strict=True)
# model_pt.load_state_dict(torch.load('./netG_epoch_14.pth', map_location=torch.device('cpu')))
checkpoint = torch.load('./netG_epoch_8.pth', map_location=pt_device)
state_dict = {k.replace('module.',''): v for k,v in checkpoint.items()}
model_pt.load_state_dict(state_dict,strict=False)
model_pt.eval()
model_pt = model_pt.to(pt_device)

print("START TENSORFLOWING")
model_tf_main = NetTf(input_nc=(3,3,1,4), ngf=8, nf_part=8, output_nc=4, n_blocks1=1, n_blocks2=1)
model_tf = model_tf_main.build()
x1, x2, x3 = tf.random.uniform((1,448,448,3)), \
            tf.random.uniform((1,448,448,3)), \
            tf.random.uniform((1,448,448,1))
# model.build((1,448,448,3))
# y = model_tf(tuple([x1, x2, x3]))


# -----------------------------------------------------------
# --------------------- model crawling ----------------------
# -----------------------------------------------------------
print('-'*50, 'Start crawling PT', '-'*50)
weights_pt = crawl_pt(model_pt)

print('-'*50, 'Start crawling TF', '-'*50)
idx_tf = crawl_tf(model_tf, weights_pt, idx=0)

print('Converted {}/{} weight params from PT to TF.'.format(idx_tf, len(weights_pt)))
if not idx_tf == len(weights_pt):
    raise Exception('Conversion not completed')
else:
    print('-'*50, 'Conversion Finished', '-'*50)




# -----------------------------------------------------------
# --------------------- model testing -----------------------
# -----------------------------------------------------------
from functions import *
from skimage.measure import label
x1 = cv2.resize(cv2.cvtColor(cv2.imread('imgs/0348_img.png'), cv2.COLOR_BGR2RGB), (448,448))[np.newaxis, ...]
x1 = 2 * x1.astype(float)/(255) - 1
x2 = cv2.resize(cv2.cvtColor(cv2.imread('imgs/bg.png'), cv2.COLOR_BGR2RGB), (448,448))[np.newaxis, ...]
x2 = 2 * x2.astype(float)/(255) - 1
x3 = cv2.resize(cv2.imread('imgs/0348_masksDL.png'), (448,448))[np.newaxis, ...][:,:,:,0:1]
x3 = 2 * x3.astype(float)/(255) - 1

reso=(448,448)

bgr_img = cv2.imread('imgs/0348_img.png'); bgr_img=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB);
rcnn = cv2.imread('imgs/0348_masksDL.png',0); 
back_img10=cv2.imread('imgs/bg.png'); back_img10=cv2.cvtColor(back_img10,cv2.COLOR_BGR2RGB);#Green-screen background
back_img20=np.zeros(back_img10.shape); back_img20[...,0]=120; back_img20[...,1]=255; back_img20[...,2]=155;
bg_im0=cv2.imread('imgs/bg.png'); bg_im0=cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB);
        

bgr_img0=bgr_img;
bbox=get_bbox(rcnn,R=bgr_img0.shape[0],C=bgr_img0.shape[1])

crop_list=[bgr_img,bg_im0,rcnn,back_img10,back_img20]
crop_list=crop_images(crop_list,reso,bbox)
bgr_img=crop_list[0]; bg_im=crop_list[1]; rcnn=crop_list[2]; back_img1=crop_list[3]; back_img2=crop_list[4]

#process segmentation mask
kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1;
K=25
cv2.imwrite('before.png', rcnn*255)
zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
del_id=zero_id[0][zero_id[0]>250]
if len(del_id)>0:
    del_id=[del_id[0]-2,del_id[0]-1,*del_id]
    rcnn=np.delete(rcnn,del_id,0)
cv2.imwrite('after.png', rcnn*255)
rcnn = cv2.copyMakeBorder( rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
rcnn =cv2.GaussianBlur(rcnn.astype(np.float32),(31,31),0)
rcnn=(255*rcnn).astype(np.uint8)
rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)

img=torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0); img=2*img.float().div(255)-1
bg=torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0); bg=2*bg.float().div(255)-1
rcnn_al=torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0); rcnn_al=2*rcnn_al.float().div(255)-1

with torch.no_grad():
    print(img.min(), img.max())
    print(bg.min(), bg.max())
    print(np.unique(rcnn_al.clone().detach().numpy()))
    rcnn_al[:] = -1
    img,bg,rcnn_al=torch.autograd.Variable(img.cpu()),  torch.autograd.Variable(bg.cpu()), torch.autograd.Variable(rcnn_al.cpu())
    alpha_pred,fg_pred_tmp=model_pt(img,bg,rcnn_al)
    print(alpha_pred.min(), alpha_pred.max())
    al_mask=(alpha_pred>0.95).type(torch.FloatTensor)
    fg_pred=img*al_mask + fg_pred_tmp*(1-al_mask)
    alpha_out=to_image(alpha_pred[0,...])
    #refine alpha with connected component
    labels=label((alpha_out>0.05).astype(int))
    try:
        assert( labels.max() != 0 )
    except:
        raise Exception
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    alpha_out=alpha_out*largestCC

    alpha_out=(255*alpha_out[...,0]).astype(np.uint8)				

    fg_out=to_image(fg_pred[0,...]); fg_out=fg_out*np.expand_dims((alpha_out.astype(float)/255>0.01).astype(float),axis=2); fg_out=(255*fg_out).astype(np.uint8)

    #Uncrop
    R0=bgr_img0.shape[0];C0=bgr_img0.shape[1]
    alpha_out0=uncrop(alpha_out,bbox,R0,C0)
    fg_out0=uncrop(fg_out,bbox,R0,C0)

back_img10=cv2.resize(back_img10,(C0,R0)); back_img20=cv2.resize(back_img20,(C0,R0))
comp_im_tr1=composite4(fg_out0,back_img10,alpha_out0)
comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

cv2.imwrite('pt'+'_out_95mask.png', al_mask.detach().data.numpy()*255)
cv2.imwrite('pt'+'_out_original.png', alpha_out)
cv2.imwrite('pt'+'_out.png', alpha_out0)
cv2.imwrite('pt'+'_fg.png', cv2.cvtColor(fg_out0,cv2.COLOR_BGR2RGB))
cv2.imwrite('pt'+'_compose.png', cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB))
cv2.imwrite('pt'+'_matte.png', cv2.cvtColor(comp_im_tr2,cv2.COLOR_BGR2RGB))


with torch.no_grad():
        img,bg,rcnn_al = tf.convert_to_tensor(img.detach().numpy().transpose(0,2,3,1), dtype=tf.float32),\
                        tf.convert_to_tensor(bg.detach().numpy().transpose(0,2,3,1), dtype=tf.float32),\
                        tf.convert_to_tensor(rcnn_al.detach().numpy().transpose(0,2,3,1), dtype=tf.float32)


        
        alpha_pred,fg_pred_tmp=model_tf(tuple([img,bg,rcnn_al]))
        img = torch.from_numpy(img.numpy()).permute(0,3,1,2).type(torch.FloatTensor)

        alpha_pred = torch.from_numpy(alpha_pred.numpy()).permute(0,3,1,2).type(torch.FloatTensor)
        fg_pred_tmp = torch.from_numpy(fg_pred_tmp.numpy()).permute(0,3,1,2).type(torch.FloatTensor)
        al_mask=(alpha_pred>0.95).type(torch.FloatTensor)
        fg_pred=img*al_mask + fg_pred_tmp*(1-al_mask)
        alpha_out=to_image(alpha_pred[0,...])
		#refine alpha with connected component
        labels=label((alpha_out>0.05).astype(int))
        try:
            assert( labels.max() != 0 )
        except:
            raise Exception
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        alpha_out=alpha_out*largestCC

        alpha_out=(255*alpha_out[...,0]).astype(np.uint8)
        fg_out=to_image(fg_pred[0,...]); fg_out=fg_out*np.expand_dims((alpha_out.astype(float)/255>0.01).astype(float),axis=2); fg_out=(255*fg_out).astype(np.uint8)

        #Uncrop
        R0=bgr_img0.shape[0];C0=bgr_img0.shape[1]
        alpha_out0=uncrop(alpha_out,bbox,R0,C0)
        fg_out0=uncrop(fg_out,bbox,R0,C0)

back_img10=cv2.resize(back_img10,(C0,R0)); back_img20=cv2.resize(back_img20,(C0,R0))
comp_im_tr1=composite4(fg_out0,back_img10,alpha_out0)
comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

cv2.imwrite('tf'+'_out_95mask.png', al_mask.detach().data.numpy()*255)
cv2.imwrite('tf'+'_out_original.png', alpha_out)
cv2.imwrite('tf'+'_out.png', alpha_out0)
cv2.imwrite('tf'+'_fg.png', cv2.cvtColor(fg_out0,cv2.COLOR_BGR2RGB))
cv2.imwrite('tf'+'_compose.png', cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB))
cv2.imwrite('tf'+'_matte.png', cv2.cvtColor(comp_im_tr2,cv2.COLOR_BGR2RGB))



# -----------------------------------------------------------
# --------------------- to tflite ---------------------------
# -----------------------------------------------------------
# i1, i2, i3 = tf.random.uniform((1, 448, 448, 3)),\
#             tf.random.uniform((1, 448, 448, 3)),\
#             tf.random.uniform((1, 448, 448, 1))
# model_tf.predict(x={'image':i1, 'back':i2, 'seg':i3})
# model_tf.compile(optimizer='sgd', loss='mean_squared_error')
converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
tflite_model = converter.convert()
open('verylite448.tflite', "wb").write(tflite_model)