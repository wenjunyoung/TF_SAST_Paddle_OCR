# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
# from paddle import nn
# import paddle.nn.functional as F
# from paddle import ParamAttr
import tensorflow as tf


from tensorflow.keras import Model

'''
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
  
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
'''

class tf_ConvBNLayer(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None,
                 ):
        super(tf_ConvBNLayer, self).__init__()

        self.if_act = if_act
        self.act = act
        

        initializer = tf.keras.initializers.GlorotUniform()

        self.conv = tf.keras.layers.Conv2D(
            filters = out_channels,
            kernel_size = kernel_size,
            strides=stride,
            padding='same',
            groups = groups,
            use_bias=False
        )

        if tf.keras.backend.image_data_format() == 'channels_last':
           bn_axis = 3
        else:
           bn_axis = 1
        self.bn = tf.keras.layers.BatchNormalization(
                                            axis=bn_axis,
                                            momentum=0.9, 
                                            epsilon=0.00001,
                                            scale=False,
                                            # moving_variance_initializer=tf.keras.initializers.GlorotUniform(),
                                            # beta_initializer=initializer,
                                            gamma_initializer=tf.keras.initializers.GlorotUniform())

        if self.act is not None:
            self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.relu(x)
        return x

'''
class DeConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(DeConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x
'''

class tf_DeConvBNLayer(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None,
                 ):
        super(tf_DeConvBNLayer, self).__init__()

        self.if_act = if_act
        self.act = act
        

        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            use_bias=False
            )

        if tf.keras.backend.image_data_format() == 'channels_last':
           bn_axis = 3
        else:
           bn_axis = 1
        self.bn = tf.keras.layers.BatchNormalization(
                                            axis=bn_axis,
                                            momentum=0.9, 
                                            epsilon=0.00001,
                                            scale=False,
                                            # moving_variance_initializer=tf.keras.initializers.GlorotUniform(),
                                            # beta_initializer=initializer,
                                            gamma_initializer=tf.keras.initializers.GlorotUniform())

        if self.act is not None:
            self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.act is not None:
            x =self.relu(x)
        return x


'''
class FPN_Up_Fusion(nn.Layer):
    def __init__(self, in_channels):
        super(FPN_Up_Fusion, self).__init__()
        in_channels = in_channels[::-1]
        out_channels = [256, 256, 192, 192, 128]
                
        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 1, 1, act=None, name='fpn_up_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 1, 1, act=None, name='fpn_up_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 1, 1, act=None, name='fpn_up_h2')
        self.h3_conv = ConvBNLayer(in_channels[3], out_channels[3], 1, 1, act=None, name='fpn_up_h3')
        self.h4_conv = ConvBNLayer(in_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_h4')

        self.g0_conv = DeConvBNLayer(out_channels[0], out_channels[1], 4, 2, act=None, name='fpn_up_g0')

        self.g1_conv = nn.Sequential(
            ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_up_g1_1'),
            DeConvBNLayer(out_channels[1], out_channels[2], 4, 2, act=None, name='fpn_up_g1_2')
        )
        self.g2_conv = nn.Sequential(
            ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_up_g2_1'),
            DeConvBNLayer(out_channels[2], out_channels[3], 4, 2, act=None, name='fpn_up_g2_2')
        )
        self.g3_conv = nn.Sequential(
            ConvBNLayer(out_channels[3], out_channels[3], 3, 1, act='relu', name='fpn_up_g3_1'),
            DeConvBNLayer(out_channels[3], out_channels[4], 4, 2, act=None, name='fpn_up_g3_2')
        )

        self.g4_conv = nn.Sequential(
            ConvBNLayer(out_channels[4], out_channels[4], 3, 1, act='relu', name='fpn_up_fusion_1'),
            ConvBNLayer(out_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_fusion_2')
        )

    def _add_relu(self, x1, x2):
        x = paddle.add(x=x1, y=x2)
        x = F.relu(x)
        return x

    def forward(self, x):
        f = x[2:][::-1]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        h3 = self.h3_conv(f[3])
        h4 = self.h4_conv(f[4])

        g0 = self.g0_conv(h0)
        g1 = self._add_relu(g0, h1)
        g1 = self.g1_conv(g1)
        g2 = self.g2_conv(self._add_relu(g1, h2))
        g3 = self.g3_conv(self._add_relu(g2, h3))
        g4 = self.g4_conv(self._add_relu(g3, h4))

        return g4

'''

class tf_FPN_Up_Fusion(tf.keras.layers.Layer):
    def __init__(self, in_channels, ):
        super(tf_FPN_Up_Fusion, self).__init__()
        in_channels = in_channels[::-1]
        out_channels = [256, 256, 192, 192, 128]
                
        self.h0_conv = tf_ConvBNLayer(in_channels[0], out_channels[0], 1, 1, act=None, name='fpn_up_h0')
        self.h1_conv = tf_ConvBNLayer(in_channels[1], out_channels[1], 1, 1, act=None, name='fpn_up_h1')
        self.h2_conv = tf_ConvBNLayer(in_channels[2], out_channels[2], 1, 1, act=None, name='fpn_up_h2')
        self.h3_conv = tf_ConvBNLayer(in_channels[3], out_channels[3], 1, 1, act=None, name='fpn_up_h3')
        self.h4_conv = tf_ConvBNLayer(in_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_h4')

        self.g0_conv = tf_DeConvBNLayer(out_channels[0], out_channels[1], 4, 2, act=None, name='fpn_up_g0')

        self.g1_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_up_g1_1'),
            tf_DeConvBNLayer(out_channels[1], out_channels[2], 4, 2, act=None, name='fpn_up_g1_2')
        ]
        )
        self.g2_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_up_g2_1'),
            tf_DeConvBNLayer(out_channels[2], out_channels[3], 4, 2, act=None, name='fpn_up_g2_2')
            ]
        )
        self.g3_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[3], out_channels[3], 3, 1, act='relu', name='fpn_up_g3_1'),
            tf_DeConvBNLayer(out_channels[3], out_channels[4], 4, 2, act=None, name='fpn_up_g3_2')
        ]
        )

        self.g4_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[4], out_channels[4], 3, 1, act='relu', name='fpn_up_fusion_1'),
            tf_ConvBNLayer(out_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_fusion_2')
        ]
        )

        self.relu = tf.keras.layers.Activation('relu')

    def _add_relu(self, x1, x2):
        x = tf.keras.layers.add([x1, x2])
        x = self.relu(x)
        return x

    # @tf.function
    def call(self, x):
        f = x[2:][::-1]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        h3 = self.h3_conv(f[3])
        h4 = self.h4_conv(f[4])

        g0 = self.g0_conv(h0)
        g1 = self._add_relu(g0, h1)
        g1 = self.g1_conv(g1)
        g2 = self.g2_conv(self._add_relu(g1, h2))
        g3 = self.g3_conv(self._add_relu(g2, h3))
        g4 = self.g4_conv(self._add_relu(g3, h4))

        return g4

'''
class FPN_Down_Fusion(nn.Layer):
    def __init__(self, in_channels):
        super(FPN_Down_Fusion, self).__init__()
        out_channels = [32, 64, 128]

        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 3, 1, act=None, name='fpn_down_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 3, 1, act=None, name='fpn_down_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 3, 1, act=None, name='fpn_down_h2')

        self.g0_conv = ConvBNLayer(out_channels[0], out_channels[1], 3, 2, act=None, name='fpn_down_g0')

        self.g1_conv = nn.Sequential(
            ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_down_g1_1'),
            ConvBNLayer(out_channels[1], out_channels[2], 3, 2, act=None, name='fpn_down_g1_2')            
        )

        self.g2_conv = nn.Sequential(
            ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_down_fusion_1'),
            ConvBNLayer(out_channels[2], out_channels[2], 1, 1, act=None, name='fpn_down_fusion_2')            
        )

    def forward(self, x):
        f = x[:3]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        g0 = self.g0_conv(h0)
        g1 = paddle.add(x=g0, y=h1)
        g1 = F.relu(g1)
        g1 = self.g1_conv(g1)
        g2 = paddle.add(x=g1, y=h2)
        g2 = F.relu(g2)
        g2 = self.g2_conv(g2)
        return g2
'''

class tf_FPN_Down_Fusion(tf.keras.layers.Layer):
    def __init__(self, in_channels, ):
        super(tf_FPN_Down_Fusion, self).__init__()
        out_channels = [32, 64, 128]

        self.h0_conv = tf_ConvBNLayer(in_channels[0], out_channels[0], 3, 1, act=None, name='fpn_down_h0')
        self.h1_conv = tf_ConvBNLayer(in_channels[1], out_channels[1], 3, 1, act=None, name='fpn_down_h1')
        self.h2_conv = tf_ConvBNLayer(in_channels[2], out_channels[2], 3, 1, act=None, name='fpn_down_h2')

        self.g0_conv = tf_ConvBNLayer(out_channels[0], out_channels[1], 3, 2, act=None, name='fpn_down_g0')

        self.g1_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_down_g1_1'),
            tf_ConvBNLayer(out_channels[1], out_channels[2], 3, 2, act=None, name='fpn_down_g1_2')
        ]           
        )

        self.g2_conv = tf.keras.Sequential([
            tf_ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_down_fusion_1'),
            tf_ConvBNLayer(out_channels[2], out_channels[2], 1, 1, act=None, name='fpn_down_fusion_2')     
        ]       
        )

        self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, x):
        f = x[:3]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        g0 = self.g0_conv(h0)
        # g1 = paddle.add(x=g0, y=h1)
        g1 = tf.keras.layers.add([g0, h1])
        g1 = self.relu(g1)
        g1 = self.g1_conv(g1)
        # g2 = paddle.add(x=g1, y=h2)
        g2 = tf.keras.layers.add([g1, h2])
        g2 = self.relu(g2)
        g2 = self.g2_conv(g2)
        return g2

'''
class Cross_Attention(nn.Layer):
    def __init__(self, in_channels):
        super(Cross_Attention, self).__init__()
        self.theta_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_theta')
        self.phi_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_phi')
        self.g_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_g')

        self.fh_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_weight')
        self.fh_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_sc')

        self.fv_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_weight')
        self.fv_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_sc')

        self.f_attn_conv = ConvBNLayer(in_channels * 2, in_channels, 1, 1, act='relu', name='f_attn')

    def _cal_fweight(self, f, shape):
        f_theta, f_phi, f_g = f
        #flatten
        f_theta = paddle.transpose(f_theta, [0, 2, 3, 1])
        f_theta = paddle.reshape(f_theta, [shape[0] * shape[1], shape[2], 128])
        f_phi = paddle.transpose(f_phi, [0, 2, 3, 1])
        f_phi = paddle.reshape(f_phi, [shape[0] * shape[1], shape[2], 128])
        f_g = paddle.transpose(f_g, [0, 2, 3, 1])
        f_g = paddle.reshape(f_g, [shape[0] * shape[1], shape[2], 128])
        #correlation
        f_attn = paddle.matmul(f_theta, paddle.transpose(f_phi, [0, 2, 1]))
        #scale
        f_attn = f_attn / (128**0.5)
        f_attn = F.softmax(f_attn)
        #weighted sum
        f_weight = paddle.matmul(f_attn, f_g)
        f_weight = paddle.reshape(
            f_weight, [shape[0], shape[1], shape[2], 128])
        return f_weight

    def forward(self, f_common):
        f_shape = paddle.shape(f_common)
        # print('f_shape: ', f_shape)

        f_theta = self.theta_conv(f_common)
        f_phi = self.phi_conv(f_common)
        f_g = self.g_conv(f_common)

        ######## horizon ########
        fh_weight = self._cal_fweight([f_theta, f_phi, f_g], 
                                        [f_shape[0], f_shape[2], f_shape[3]])
        fh_weight = paddle.transpose(fh_weight, [0, 3, 1, 2])
        fh_weight = self.fh_weight_conv(fh_weight)
        #short cut
        fh_sc = self.fh_sc_conv(f_common)
        f_h = F.relu(fh_weight + fh_sc)

        ######## vertical ########
        fv_theta = paddle.transpose(f_theta, [0, 1, 3, 2])
        fv_phi = paddle.transpose(f_phi, [0, 1, 3, 2])
        fv_g = paddle.transpose(f_g, [0, 1, 3, 2])
        fv_weight = self._cal_fweight([fv_theta, fv_phi, fv_g], 
                                        [f_shape[0], f_shape[3], f_shape[2]])
        fv_weight = paddle.transpose(fv_weight, [0, 3, 2, 1])
        fv_weight = self.fv_weight_conv(fv_weight)
        #short cut
        fv_sc = self.fv_sc_conv(f_common)
        f_v = F.relu(fv_weight + fv_sc)

        ######## merge ########
        f_attn = paddle.concat([f_h, f_v], axis=1)
        f_attn = self.f_attn_conv(f_attn)
        return f_attn
'''

class tf_Cross_Attention(tf.keras.layers.Layer):
    def __init__(self, in_channels, ):
        super(tf_Cross_Attention, self).__init__()

        self.theta_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_theta')
        self.phi_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_phi')
        self.g_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_g')

        self.fh_weight_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_weight')
        self.fh_sc_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_sc')

        self.fv_weight_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_weight')
        self.fv_sc_conv = tf_ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_sc')

        self.f_attn_conv = tf_ConvBNLayer(in_channels * 2, in_channels, 1, 1, act='relu', name='f_attn')

        self.relu = tf.keras.layers.Activation('relu')
        

    def _cal_fweight(self, f, shape):
        f_theta, f_phi, f_g = f
        #flatten
        f_theta = tf.transpose(f_theta, [0, 2, 3, 1])
        f_theta = tf.reshape(f_theta, [shape[0] * shape[1], shape[2], 128])
        f_phi = tf.transpose(f_phi, [0, 2, 3, 1])
        f_phi = tf.reshape(f_phi, [shape[0] * shape[1], shape[2], 128])
        f_g = tf.transpose(f_g, [0, 2, 3, 1])
        f_g = tf.reshape(f_g, [shape[0] * shape[1], shape[2], 128])

        #correlation
        # f_attn = tf.linalg.matmul(f_theta, tf.transpose(f_phi, [0, 2, 1]))
        f_attn = tf.matmul(f_theta, tf.transpose(f_phi, [0, 2, 1]))

        #scale
        f_attn = f_attn / (128**0.5)
        f_attn = tf.nn.softmax(f_attn)

        #weighted sum
        # f_weight = tf.linalg.matmul(f_attn, f_g)
        f_weight = tf.matmul(f_attn, f_g)
        f_weight = tf.reshape(f_weight, [shape[0], shape[1], shape[2], 128])
        return f_weight

    # @tf.function
    def call(self, f_common):
        f_shape = tf.shape(f_common)
        # print('f_shape: ', f_shape)

        f_theta = self.theta_conv(f_common)
        f_phi = self.phi_conv(f_common)
        f_g = self.g_conv(f_common)

        ######## horizon ########
        fh_weight = self._cal_fweight([f_theta, f_phi, f_g], [f_shape[0], f_shape[2], f_shape[3]])
        fh_weight = tf.transpose(fh_weight, [0, 3, 1, 2])
        fh_weight = self.fh_weight_conv(fh_weight)
        #short cut
        fh_sc = self.fh_sc_conv(f_common)
        f_h = self.relu(fh_weight + fh_sc)

        ######## vertical ########
        fv_theta = tf.transpose(f_theta, [0, 1, 3, 2])
        fv_phi = tf.transpose(f_phi, [0, 1, 3, 2])
        fv_g = tf.transpose(f_g, [0, 1, 3, 2])
        fv_weight = self._cal_fweight([fv_theta, fv_phi, fv_g], [f_shape[0], f_shape[3], f_shape[2]])
        fv_weight = tf.transpose(fv_weight, [0, 3, 2, 1])
        fv_weight = self.fv_weight_conv(fv_weight)
        #short cut
        fv_sc = self.fv_sc_conv(f_common)
        f_v = self.relu(fv_weight + fv_sc)

        ######## merge ########
        f_attn = tf.concat([f_h, f_v], axis=1)
        f_attn = self.f_attn_conv(f_attn)
        return f_attn

'''
class SASTFPN(nn.Layer):
    def __init__(self, in_channels, with_cab=False, **kwargs):
        super(SASTFPN, self).__init__()
        self.in_channels = in_channels
        self.with_cab = with_cab
        self.FPN_Down_Fusion = FPN_Down_Fusion(self.in_channels)
        self.FPN_Up_Fusion = FPN_Up_Fusion(self.in_channels)
        self.out_channels = 128
        self.cross_attention = Cross_Attention(self.out_channels)

    def forward(self, x):
        #down fpn
        f_down = self.FPN_Down_Fusion(x)

        #up fpn
        f_up = self.FPN_Up_Fusion(x)

        #fusion
        f_common = paddle.add(x=f_down, y=f_up)
        f_common = F.relu(f_common)

        if self.with_cab:
            # print('enhence f_common with CAB.')
            f_common = self.cross_attention(f_common)

        return f_common
'''


class tf_SASTFPN(tf.keras.Model):
    def __init__(self, in_channels, with_cab=False, **kwargs):
        super(tf_SASTFPN, self).__init__()
        self.in_channels = in_channels
        self.with_cab = with_cab
        self.tf_FPN_Down_Fusion = tf_FPN_Down_Fusion(self.in_channels)
        self.tf_FPN_Up_Fusion = tf_FPN_Up_Fusion(self.in_channels)
        self.out_channels = 128
        self.tf_cross_attention = tf_Cross_Attention(self.out_channels)
        self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, x):
        #down fpn
        f_down = self.tf_FPN_Down_Fusion(x)

        #up fpn
        f_up = self.tf_FPN_Up_Fusion(x)

        #fusion
        f_common = f_down + f_up
        f_common = self.relu(f_common)

        if self.with_cab:
            # print('enhence f_common with CAB.')
            f_common = self.tf_cross_attention(f_common)

        return f_common

