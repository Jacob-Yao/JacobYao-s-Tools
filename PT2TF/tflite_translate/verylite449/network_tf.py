import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Add, BatchNormalization, Dropout, Flatten, Layer, Dense
from tensorflow.keras.layers import AvgPool2D, Conv2D, DepthwiseConv2D, ReLU, ZeroPadding2D, Softmax
from tensorflow.keras.layers import ReLU, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.activations import tanh

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


class ReplicationPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')


class Tanh(Layer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)
    def call(self, input_tensor, mask=None):
        return tanh(input_tensor)


# netG=ResnetConditionHR(input_nc=(3,3,1,4),
#                        output_nc=4,
#                       n_blocks1=args.n_blocks1,
#                       n_blocks2=args.n_blocks2)

class ResnetConditionHRVeryLite(Model):

    def __init__(self, input_nc, output_nc, \
                ngf=64, nf_part=64, \
                norm_layer=BatchNormalization, use_dropout=False, \
                n_blocks1=7, n_blocks2=3, padding_type='reflect'):
        super(ResnetConditionHRVeryLite, self).__init__()

        assert(n_blocks1 >= 0); assert(n_blocks2 >= 0)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True

        # main encoder output 256xW/4xH/4
        model_enc1 = [ReflectionPadding2D(padding=(1,1)), \
                        Conv2D(ngf, 3, input_shape=[448,448,input_nc[0]], padding='valid', use_bias=use_bias), \
                        norm_layer(momentum=0.1, epsilon=1e-05), \
                        ReLU()]
        model_enc1 += [Conv2D(ngf * 2, 3, strides=2, padding='same', use_bias=use_bias), \
                        norm_layer(momentum=0.1, epsilon=1e-05), \
                        ReLU()]

        # back encoder output 256xW/4xH/4
        model_enc_back = [ReflectionPadding2D(padding=(1,1)), \
                        Conv2D(ngf, 3, input_shape=[448,448,input_nc[1]], padding='valid', use_bias=use_bias), \
                        norm_layer(momentum=0.1, epsilon=1e-05), \
                        ReLU()]
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**i
            model_enc_back += [Conv2D(ngf * mult * 2, 3, strides=2, padding='same', use_bias=use_bias), \
                                norm_layer(momentum=0.1, epsilon=1e-05), \
                                ReLU()]

        # seg encoder output 256xW/4xH/4
        model_enc_seg = [ReflectionPadding2D(padding=(1,1)), \
                        Conv2D(ngf, 3, input_shape=[448,448,input_nc[2]], padding='valid', use_bias=use_bias), \
                        norm_layer(momentum=0.1, epsilon=1e-05), \
                        ReLU()]
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**i
            model_enc_seg += [Conv2D(ngf * mult * 2, 3, strides=2, padding='same', use_bias=use_bias), \
                            norm_layer(momentum=0.1, epsilon=1e-05), \
                            ReLU()]
        mult = 2**n_downsampling

        self.model_enc1 = Sequential(model_enc1)
        self.model_enc2 = Sequential(model_enc1)
        # self.model_enc2 = self.model_enc1
        self.model_enc_back = Sequential(model_enc_back)
        self.model_enc_seg = Sequential(model_enc_seg)

        mult = 2 ** n_downsampling
        self.comb_back = Sequential([
                        Conv2D(nf_part, 1, strides=1, padding='valid', use_bias=False),
                        norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU()
                        ])
        self.comb_seg = Sequential([
                        Conv2D(nf_part, 1, strides=1, padding='valid', use_bias=False),
                        norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU()
                        ])

        # decoder
        model_res_dec = [Conv2D(ngf*mult, 1, strides=1, padding='valid', use_bias=False),
                        norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU()]
        for i in range(n_blocks1):
            model_res_dec += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_res_dec_al=[]
        for i in range(n_blocks2):
            model_res_dec_al += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_res_dec_fg=[]
        for i in range(n_blocks2):
            model_res_dec_fg += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_dec_al=[]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_dec_al += [UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear'),
                            Conv2D(int(ngf * mult / 2), 3, strides=1, padding='same'),
                            norm_layer(momentum=0.1, epsilon=1e-05),
                            ReLU()]
        model_dec_al += [ReflectionPadding2D(padding=(1,1)),
                        Conv2D(1, 3, padding='valid'),
                        Tanh()]

        model_dec_fg1 = [UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear'),
                        Conv2D(ngf, 3, strides=1, padding='same'),
                        norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU(),
                        ReflectionPadding2D(padding=(1,1)),
                        Conv2D(output_nc-1, 3, padding='valid')]
        model_dec_fg2 = [UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear'),
                        Conv2D(ngf, 3, strides=1, padding='same'),
                        norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU(),
                        ReflectionPadding2D(padding=(1,1)),
                        Conv2D(output_nc-1, 3, padding='valid')]

        self.model_res_dec = Sequential(model_res_dec)
        self.model_res_dec_al = Sequential(model_res_dec_al)
        self.model_res_dec_fg = Sequential(model_res_dec_fg)
        self.model_al_out = Sequential(model_dec_al)
        self.model_dec_fg1 = Sequential(model_dec_fg1)
        self.model_fg_out = Sequential(model_dec_fg2)
    
    # image, back, seg
    def call(self, input_tensors, training=False):
        image, back, seg = input_tensors[0], input_tensors[1], input_tensors[2]
        img_feat1 = self.model_enc1(image)
        img_feat = img_feat1
        
        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)
        
        # TODO axis?
        # oth_feat=torch.cat([self.comb_back(torch.cat([img_feat,back_feat],dim=1)),
        #                     self.comb_seg(torch.cat([img_feat,seg_feat],dim=1))],dim=1)
        # out_dec=self.model_res_dec(torch.cat([img_feat,oth_feat],dim=1))
        oth_feat = Concatenate(axis=-1)([self.comb_back(Concatenate(axis=-1)([img_feat, back_feat])),
                            self.comb_seg(Concatenate(axis=-1)([img_feat, seg_feat]))])
              
        out_dec = self.model_res_dec(Concatenate(axis=-1)([img_feat, oth_feat]))
        
        out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_dec_al)
        
        out_dec_fg = self.model_res_dec_fg(out_dec)
        out_dec_fg1 = self.model_dec_fg1(out_dec_fg)
        
        _ = self.model_fg_out(out_dec_fg)
        fg_out = out_dec_fg1
        
        return al_out, fg_out


# Define a resnet block
class ResnetBlock(Model):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        if padding_type == 'reflect':
            conv_block += [ReflectionPadding2D(padding=(1,1)),
                            Conv2D(dim, 3, padding='valid', use_bias=use_bias)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPadding2D(padding=(1,1)),
                            Conv2D(dim, 3, padding='valid', use_bias=use_bias)]
        elif padding_type == 'zero':
            conv_block += [Conv2D(dim, 3, padding='same', use_bias=use_bias)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [norm_layer(momentum=0.1, epsilon=1e-05),
                        ReLU()]
        if use_dropout:
            conv_block += [Dropout(0.5)]

        if padding_type == 'reflect':
            conv_block += [ReflectionPadding2D(padding=(1,1)),
                            Conv2D(dim, 3, padding='valid', use_bias=use_bias)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPadding2D(padding=(1,1)),
                            Conv2D(dim, 3, padding='valid', use_bias=use_bias)]
        elif padding_type == 'zero':
            conv_block += [Conv2D(dim, 3, padding='same', use_bias=use_bias)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [norm_layer(momentum=0.1, epsilon=1e-05)]

        return Sequential(conv_block)

    def call(self, x):
        out = x + self.conv_block(x)
        return out