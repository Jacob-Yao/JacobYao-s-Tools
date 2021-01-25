import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Add, BatchNormalization, Dropout, Flatten, Layer, Dense
from tensorflow.keras.layers import AvgPool2D, Conv2D, DepthwiseConv2D, ReLU, ZeroPadding2D, Softmax
from tensorflow.keras.layers import ReLU, MaxPool2D, ZeroPadding2D

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2D(out_planes, 3, stride, use_bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2D(out_planes, 1, stride, use_bias=False)

class BasicBlock(Layer):
    expansion = 1

    def __init__(self, inplanes, out_planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNormalization(epsilon=1e-5)
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        layers = []
        layers.append(ZeroPadding2D(padding=1))
        layers.append(conv3x3(inplanes, out_planes,stride))
        layers.append(BatchNormalization(epsilon=1e-5))
        layers.append(ReLU())
        layers.append(ZeroPadding2D(padding=1))
        layers.append(conv3x3(inplanes, out_planes))
        layers.append(BatchNormalization(epsilon=1e-5))

        self.downsample = downsample

        self.conv = Sequential(layers)


    def call(self, x):
        identity = x
        out = self.conv(x)
        # x = tf.nn.relu6(x)
        # x = tf.pad(x, [[0,0], [self.padding, self.padding],[self.padding, self.padding], [0,0]])
        # x = self.conv2(x)
        # x = tf.nn.relu6(x)

        # x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(x)
            # print('DOWNSAMPLE',identity)
        out += identity
        out = tf.nn.relu(out)

        return out

####

class ResNet(Model):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2D(self.inplanes, 7, 2, use_bias=False)
        self.bn1 = BatchNormalization(epsilon=1e-5)
        self.relu = ReLU()
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.flatten = Flatten()
        self.fc = (Dense(num_classes, activation=tf.nn.softmax))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv1x1(self.inplanes, planes* block.expansion, stride)
            downsample = Sequential([
                conv1x1(self.inplanes, planes* block.expansion, stride),
                norm_layer(epsilon=1e-5)]
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return Sequential(layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = tf.pad(x, [[0,0], [3, 3],[3, 3], [0,0]])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = tf.pad(x, [[0,0], [1, 1],[1, 1], [0,0]])
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = tf.nn.avg_pool2d(x, 7, 1, 'VALID')

        x = self.flatten(x)
        x = self.fc(x)

        return x

    def call(self, x):
        return self._forward_impl(x)



if __name__ == '__main__':

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=17)
    x = tf.random.uniform((1,224,224,3))
    model.build((1,224,224,3))
    y = model(x)
    # print(y.shape)
    tf_layers = []
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    for l in model.layers:
        if isinstance(l, Flatten) or isinstance(l,ReLU) or isinstance(l,MaxPool2D):
            continue
        if not isinstance(l,Sequential):
            # print('\tl',l)
            tf_layers.append(l)
        elif isinstance(l,Sequential):
            for m in l.layers:
                # print('xxm',m)
                if isinstance(m, BasicBlock):
                    for n in m.conv.layers:
                        if not isinstance(n,ReLU) and not isinstance(n,ZeroPadding2D):
                            # print('\tn',n)
                            tf_layers.append(n)
                    if hasattr(m, 'downsample') and m.downsample is not None:
                        for n in m.downsample.layers:
                            if not isinstance(n,ReLU) and not isinstance(n,ZeroPadding2D):
                                # print('\td',n)
                                tf_layers.append(n)

                else:
                    raise ValueError
        else:
            raise NotImplementedError

    print(len(tf_layers))
