"""
time: 2022/03/22
author: cong
object_detection的yolov4模型文件
网络结构如图所示：
"""



from functools import wraps, reduce
from keras import backend as k
from keras.initializers import random_normal
from keras.layers import (Add, Concatenate, Conv2D, Layer,
                          Input,MaxPooling2D,UpSampling2D,
                          LeakyReLU, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model


class Mish(Layer):
    def __init__(self, **kwargs):  # 初始化实例属性,*args 表示任何多个无名参数，它是一个tuple；**kwargs 表示关键字参数，它是一个dict。
        super(Mish, self).__init__(**kwargs)#super(Mish, self)：Mish继承父类Layer的属性，然后再调用__init__()方法传参：**kwargs
        self.supports_masking = True    #当前类的supports_masking属性为True,masking的作用是屏蔽padding值。

    def call(self, inputs):
        return inputs * k.tanh(k.softplus(inputs))
    
    def get_config(self):
        config = super(Mish, self).get_config()  # 继承父类Layer的get_config()方法
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)#https://www.jianshu.com/p/b1f3a26d5746
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer': random_normal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def make_five_convs(x, num_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x= DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x= DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x



#  cspdarknet53残差结构
#  首先利用ZeroPadding2D和一个步长为2*2的卷积块进行宽和高的压缩
#  然后建立一个残差边shortconv，这个大残差边绕过内部的残差架构
#  主干部分会对num_blocks进行循环，循环内部就是残差结构
#  对于整个CSPdarknet的结构块，就是一个大残差边+内部小残差块


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    # 首先利用ZeroPadding2D和一个步长为2*2的卷积块进行宽和高的压缩
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)
    # 然后建立一个残差边shortconv，这个大残差边绕过内部的残差架构
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    # 建立内部的残差结构
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
            DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    # 将大残差边堆叠回来
    route = Concatenate()([postconv,shortconv])
    # 对通道数进行整合
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)


# CSPdarknet53的主体部分
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3


#---------------------以上为CSPdarknet53的主体部分-------------------#
#-----开始构建整体的网络-----#
def yolo_body(input_shape, anchors_mask, num_classes):
    inputs = Input(input_shape)
    #加载CSPdarknet53结构
    feat1, feat2, feat3 = darknet_body(inputs)

    # feat3进行三次卷积
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)

    # 使用SPP结构，不用尺度的最大池化后堆叠
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    #SPP之后， Concat
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    #conv*3
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)
#----上采样过程-----#
    # P5开始上采样:先卷积，再上采样
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(P5)
    # feat2进行卷积得到P4
    P4 = DarknetConv2D_BN_Leaky(256, (1,1))(feat2)
    # concat:P4和P5_upsample
    P4 = Concatenate()([P4, P5_upsample])
    # P4进行5次卷积
    P4 = make_five_convs(P4, 256)

    #P4开始上采样：先卷积，再上采样
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(P4)
    #feat1进行卷积得到P3
    P3 = DarknetConv2D_BN_Leaky(128, (1,1))(feat1)
    # concat:P3和P4_upsample
    P3 = Concatenate()([P3, P4_upsample])
    # P3进行5次卷积
    P3 = make_five_convs(P3, 128)
#-----逐步输出(Conv_Batchnorm_Leaky + conv) 下采样过程—---#
    # 输出P3_output
    P3_output = DarknetConv2D_BN_Leaky(256, (3,3))(P3)
    P3_output = DarknetConv2D(len(anchors_mask[0])*(num_classes+5), (1,1))(P3_output)

    # 输出P4_output
    P3_downsample = ZeroPadding2D(((1,0),(1,0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(P3_downsample)
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4, 256)
    P4_output = DarknetConv2D_BN_Leaky(512, (3,3))(P4)
    P4_output = DarknetConv2D(len(anchors_mask[1])*(num_classes+5), (1,1))(P4_output)

    # 输出P5_output
    P4_downsample = ZeroPadding2D(((1,0),(1,0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(P4_downsample)
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5, 512)
    P5_output = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5_output = DarknetConv2D(len(anchors_mask[2])*(num_classes+5), (1,1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])





















