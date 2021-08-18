# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import paddle
# from paddle import ParamAttr
# import paddle.nn as nn
# import paddle.nn.functional as F

import tensorflow as tf

from tensorflow.keras import Model

__all__ = ["ResNet_SAST", "tf_ResNet_SAST"]


'''
class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
            )

        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False
            )

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance'
            )

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y
'''

class tf_ConvBNLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, 
            ):
        super(tf_ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode

        # self._pool2d_avg = nn.AvgPool2D(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        if self.is_vd_mode:
            self._pool2d_avg = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_first')

        self._conv = tf.keras.layers.Conv2D(
            filters = out_channels,
            kernel_size = kernel_size,
            strides=stride,
            padding='same',
            groups = groups,
            use_bias=False
            # kernel_initializer=initializer
        )

        # self._batch_norm = nn.BatchNorm(out_channels, act=act)
        if tf.keras.backend.image_data_format() == 'channels_last':
           bn_axis = 3
        else:
           bn_axis = 1
        self._batch_norm = tf.keras.layers.BatchNormalization(
                                            axis=bn_axis,
                                            momentum=0.9, 
                                            epsilon=0.00001,
                                            scale=False,
                                            # moving_variance_initializer=tf.keras.initializers.GlorotUniform(),
                                            # beta_initializer=initializer,
                                            gamma_initializer=tf.keras.initializers.GlorotUniform())

        self.act = act
        if self.act is not None:
            self.relu = tf.keras.layers.Activation('relu')

    # def _bn(inputs, is_training):
    #     bn = tf.layers.batch_normalization(
    #         inputs=inputs,
    #         training=is_training,
    #         momentum = 0.99)
    #     return bn

    # @tf.function
    def call(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self.relu(y)
        return y

'''
class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y
'''

class tf_BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None,
                 ):
        super(tf_BottleneckBlock, self).__init__()

        self.conv0 = tf_ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            )

        self.conv1 = tf_ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            )

        self.conv2 = tf_ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            )

        if not shortcut:
            self.short = tf_ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                )

        self.shortcut = shortcut
        
        self.relu = tf.keras.layers.Activation('relu')
        #self.add = tf.keras.layers.add([])


    # @tf.function
    def call(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        # y = paddle.add(x=short, y=conv2)
        y = tf.keras.layers.add([short, conv2])
        y = self.relu(y)
        return y

'''
class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y
'''

class tf_BasicBlock(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None,
                 ):
        super(tf_BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = tf_ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            )

        self.conv1 = tf_ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            )

        if not shortcut:
            self.short = tf_ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                )

        self.shortcut = shortcut

        self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = tf.keras.layers.add([short, conv1])
        y = self.relu(y)
        return y


'''
class ResNet_SAST(nn.Layer):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet_SAST, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            # depth = [3, 4, 6, 3]
            depth = [3, 4, 6, 3, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        # num_channels = [64, 256, 512,
        #                 1024] if layers >= 50 else [64, 64, 128, 256]
        # num_filters = [64, 128, 256, 512]
        num_channels = [64, 256, 512,
                        1024, 2048] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.stages = []
        self.out_channels = [3, 64]
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    block_list.append(bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    block_list.append(basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        out = [inputs]
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        out.append(y)
        y = self.pool2d_max(y)
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out

'''


class tf_ResNet_SAST(tf.keras.Model):
    def __init__(self, in_channels=3, ResNet_Layer_num=50, **kwargs):
        super(tf_ResNet_SAST, self).__init__()

        self.ResNet_Layer_num = ResNet_Layer_num # resnet 层数=50

        supported_layers = [18, 34, 50, 101, 152, 200]
        assert ResNet_Layer_num in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, ResNet_Layer_num)

        if ResNet_Layer_num == 18:
            depth = [2, 2, 2, 2]
        elif ResNet_Layer_num == 34 or ResNet_Layer_num == 50:
            # depth = [3, 4, 6, 3]
            depth = [3, 4, 6, 3, 3]
        elif ResNet_Layer_num == 101:
            depth = [3, 4, 23, 3]
        elif ResNet_Layer_num == 152:
            depth = [3, 8, 36, 3]
        elif ResNet_Layer_num == 200:
            depth = [3, 12, 48, 3]
        # num_channels = [64, 256, 512,
        #                 1024] if ResNet_Layer_num >= 50 else [64, 64, 128, 256]
        # num_filters = [64, 128, 256, 512]

        # 卷积核个数
        num_channels = [64, 256, 512, 1024, 2048] if ResNet_Layer_num >= 50 else [64, 64, 128, 256]
        # 卷积核大小
        num_filters = [64, 128, 256, 512, 512]

        self.conv1_1 = tf_ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            )

        self.conv1_2 = tf_ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            )

        self.conv1_3 = tf_ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            )

        # self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.pool2d_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.stages = []
        self.out_channels = [3, 64]
        if ResNet_Layer_num >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False

                # res_block = keras.Sequential()
                for i in range(depth[block]):
                    
                    bottleneck_block = tf_BottleneckBlock(
                                            in_channels=num_channels[block] if i == 0 else num_filters[block] * 4,
                                            out_channels=num_filters[block],
                                            stride=2 if i == 0 and block != 0 else 1,
                                            shortcut=shortcut,
                                            if_first=block == i == 0
                                            )                        

                    shortcut = True
                    block_list.append(bottleneck_block)
                    # res_block.add(bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                self.stages.append(tf.keras.models.Sequential(block_list))
                # self.stages.append(res_block)

        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    basic_block = tf_BasicBlock(
                                    in_channels=num_channels[block] if i == 0 else num_filters[block],
                                    out_channels=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    shortcut=shortcut,
                                    if_first=block == i == 0                                    
                                    )
                    shortcut = True
                    block_list.append(basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(tf.keras.models.Sequential(block_list))
    # @tf.function
    def call(self, inputs):
        out = [inputs]
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        out.append(y)
        y = self.pool2d_max(y)
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out        