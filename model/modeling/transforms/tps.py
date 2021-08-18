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

import math
# import paddle
# from paddle import nn, ParamAttr
# from paddle.nn import functional as F
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model

'''
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = "bn_" + name
        self.bn = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

'''

class tf_ConvBNLayer(tf.keras.Model):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None
                 training = True):
        super(tf_ConvBNLayer, self).__init__()

        self.if_act = if_act
        self.act = act
        self.training = training

        self.conv = tf.keras.layers.Conv2D(
            filters = out_channels,
            kernel_size = kernel_size,
            stride=stride,
            padding='same',
            groups = groups,
            use_bias=False
        )
  
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
class LocalizationNetwork(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        self.block_list = []
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            conv = self.add_sublayer(
                name,
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act='relu',
                    name=name))
            self.block_list.append(conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2D(1)
            else:
                pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.append(pool)
        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(
            in_channels,
            fc_dim,
            weight_attr=ParamAttr(
                learning_rate=loc_lr,
                name=name + "_w",
                initializer=nn.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name=name + '.b_0'),
            name=name)

        # Init fc2 in LocalizationNetwork
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"

        param_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(np.zeros([fc_dim, F * 2])),
            name=name + "_w")

        bias_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(initial_bias),
            name=name + "_b")

        self.fc2 = nn.Linear(
            fc_dim,
            F * 2,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

        self.out_channels = F * 2

    def forward(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(axis=2).squeeze(axis=2)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias
'''


class tf_AdaptiveAvgPool2D(tf.keras.Model):
    def __init__(self, outputsize = 1):
        super(tf_AdaptiveAvgPool2D, self).__init__()
        '''
        tensoeflow 实现自适应平均池化
        
        stride = floor ( (input_size / (output_size) )
        kernel_size = input_size − (output_size−1) * stride
        '''

        self.outputsize = np.array(outputsize)

    # @tf.function
    def call(self, x):

        inputsz = np.array(x.shape[2:])
        outputsz = np.array(self.outputsize)
        stridesz = np.floor(inputsz/outputsz).astype(np.int32)
        kernelsz = inputsz-(outputsz-1)*stridesz
        avg = tf.keras.layers.AveragePooling2D(kernel_size=[1,kernelsz,kernelsz,1],stride=[1,stridesz,stridesz,1])
        return avg


class tf_LocalizationNetwork(tf.keras.Model):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name, training=True):
        super(tf_LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        self.block_list = []
        self.bl = keras.Sequential()
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            conv = tf_ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act='relu',
                    name=name,
                    training=training)
                    
            self.block_list.append(conv)
            self.bl.add(conv)
            if fno == len(num_filters_list) - 1:
                pool = tf_AdaptiveAvgPool2D(1)
            else:
                pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
            in_channels = num_filters
            self.block_list.append(pool)
            self.bl.add(pool)
        self.blSequential = keras.Sequential(self.block_list)

        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)

        self.fc1 = tf.keras.layers.Dense(
            units=fc_dim,
            kernel_initializer=tf.random.uniform(fc_dim, -stdv, stdv),
            )

        # Init fc2 in LocalizationNetwork
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)

        name = "loc_fc2"
        self.fc2 = tf.keras.layers.Dense(
            F * 2,
            kernel_initializer=tf.zeros_initializer(shape=[fc_dim, F * 2]),
            bias_initializer=tf.constant_initializer(initial_bias)
            )

        self.out_channels = F * 2
        self.relu = tf.keras.layers.Activation('relu')

    # @tf.function
    def call(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        # for block in self.block_list:
        #     x = block(x)
        x = self.blSequential(x)
        x = tf.squeeze(x, axis=2)
        x = tf.squeeze(x, axis=2)
        x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)
        x = tf.reshape(x, shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias

'''
class GridGenerator(nn.Layer):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        initializer = nn.initializer.Constant(value=0.0)
        param_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_w")
        bias_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_b")
        self.fc = nn.Linear(
            in_channels,
            6,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)

        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype('float32')
        P_hat_tensor = self.build_P_hat_paddle(
            C, paddle.to_tensor(P)).astype('float32')

        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor.stop_gradient = True

        batch_C_prime_with_zeros = paddle.concat(
            [batch_C_prime, batch_C_ex_part_tensor], axis=1)
        batch_T = paddle.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = paddle.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = paddle.linspace(-1.0, 1.0, int(F / 2), dtype='float64')
        ctrl_pts_y_top = -1 * paddle.ones([int(F / 2)], dtype='float64')
        ctrl_pts_y_bottom = paddle.ones([int(F / 2)], dtype='float64')
        ctrl_pts_top = paddle.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = paddle.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = paddle.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (paddle.arange(
            -I_r_width, I_r_width, 2, dtype='float64') + 1.0
                      ) / paddle.to_tensor(np.array([I_r_width]))

        I_r_grid_y = (paddle.arange(
            -I_r_height, I_r_height, 2, dtype='float64') + 1.0
                      ) / paddle.to_tensor(np.array([I_r_height]))

        # P: self.I_r_width x self.I_r_height x 2
        P = paddle.stack(paddle.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        P = paddle.transpose(P, perm=[1, 0, 2])
        # n (= self.I_r_width x self.I_r_height) x 2
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = paddle.zeros((F, F), dtype='float64')  # F x F
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = paddle.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = (hat_C**2) * paddle.log(hat_C)
        delta_C = paddle.concat(  # F+3 x F+3
            [
                paddle.concat(
                    [paddle.ones(
                        (F, 1), dtype='float64'), C, hat_C], axis=1),  # F x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (2, 3), dtype='float64'), paddle.transpose(
                                C, perm=[1, 0])
                    ],
                    axis=1),  # 2 x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (1, 3), dtype='float64'), paddle.ones(
                                (1, F), dtype='float64')
                    ],
                    axis=1)  # 1 x F+3
            ],
            axis=0)
        inv_delta_C = paddle.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        C_tile = paddle.unsqueeze(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = paddle.norm(P_diff, p=2, axis=2, keepdim=False)

        # rbf: n x F
        rbf = paddle.multiply(
            paddle.square(rbf_norm), paddle.log(rbf_norm + eps))
        P_hat = paddle.concat(
            [paddle.ones(
                (n, 1), dtype='float64'), P, rbf], axis=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor
'''

class tf_GridGenerator(tf.keras.Model):
    def __init__(self, in_channels, num_fiducial):
        super(tf_GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        initializer = tf.constant_initializer(value=0.0)
        self.fc = tf.keras.layers.Dense(
            6,
            kernel_initializer=initializer,
            bias_initializer=initializer
            )

    # @tf.function
    def call(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)

        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype('float32')
        P_hat_tensor = self.build_P_hat_paddle(C, tf.convert_to_tensor(P)).astype('float32')

        inv_delta_C_tensor = tf.stop_gradient(inv_delta_C_tensor)
        P_hat_tensor = tf.stop_gradient(P_hat_tensor)

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor = tf.stop_gradient(batch_C_ex_part_tensor)

        batch_C_prime_with_zeros = tf.concat([batch_C_prime, batch_C_ex_part_tensor], axis=1)

        batch_T = tf.math.multiply(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = tf.math.multiply(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = tf.linspace(-1.0, 1.0, int(F / 2), dtype=tf.dtypes.float64)
        ctrl_pts_y_top = -1 * tf.ones([int(F / 2)], dtype=tf.dtypes.float64)
        ctrl_pts_y_bottom = tf.ones([int(F / 2)], dtype=tf.dtypes.float64)
        ctrl_pts_top = tf.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = tf.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = tf.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (tf.experimental.numpy.arange-I_r_width, I_r_width, 2, dtype=tf.dtypes.float64) + 1.0) /  \
                     tf.convert_to_tensor(np.array([I_r_width]))

        I_r_grid_y = (tf.experimental.numpy.arange(-I_r_height, I_r_height, 2, dtype=tf.dtypes.float64) + 1.0) / \
                     tf.convert_to_tensor(np.array([I_r_height]))

        # P: self.I_r_width x self.I_r_height x 2
        P = tf.stack(tf.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        P = tf.transpose(P, perm=[1, 0, 2])
        # n (= self.I_r_width x self.I_r_height) x 2
        P = tf.reshape(P, [-1, 2])
        return P

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = tf.zeros((F, F), dtype=tf.dtypes.float64)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = tf.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = (hat_C**2) * tf.math.log(hat_C)
        delta_C = tf.concat(  # F+3 x F+3
            [
                tf.concat(
                    [tf.ones(
                        (F, 1), dtype=tf.dtypes.float64), C, hat_C], axis=1),  # F x F+3
                tf.concat(
                    [
                        tf.zeros(
                            (2, 3), dtype=tf.dtypes.float64), tf.transpose(
                                C, perm=[1, 0])
                    ],
                    axis=1),  # 2 x F+3
                tf.concat(
                    [
                        tf.zeros(
                            (1, 3), dtype=tf.dtypes.float64), tf.ones(
                                (1, F), dtype=tf.dtypes.float64)
                    ],
                    axis=1)  # 1 x F+3
            ],
            axis=0)
        inv_delta_C = tf.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        P_tile = tf.tile(tf.expand_dims(P, axis=1), multiples=(1, F, 1))
        C_tile = tf.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = tf.norm(P_diff, ord=2, axis=2, keepdim=False)

        # rbf: n x F
        rbf = tf.math.multiply(tf.math.square(rbf_norm), tf.math.log(rbf_norm + eps))

        P_hat = tf.concat([tf.ones((n, 1), dtype=tf.dtypes.float64), P, rbf], axis=1)

        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = tf.reshape(batch_C_prime, [B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = tf.reshape(batch_C_ex_part_tensor, [-1, 3, 2])
        return batch_C_ex_part_tensor


'''
class TPS(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr,model_name)

        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)

        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape([-1, image.shape[2], image.shape[3], 2])

        batch_I_r = F.grid_sample(x=image, grid=batch_P_prime)
        return batch_I_r
'''

class tf_TPS(tf.keras.Model):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(tf_TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr,model_name)

        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)

        self.out_channels = in_channels
        self.resampler = tf_grid_sampe()

    # @tf.function
    def call(self, image):
        image = tf.stop_gradient(image)
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = tf.reshape(batch_P_prime, [-1, image.shape[2], image.shape[3], 2])
        
        # batch_I_r = F.grid_sample(x=image, grid=batch_P_prime)
        batch_I_r = self.resampler(image, x=batch_P_prime[:, :, :, 0], x=batch_P_prime[:, :, :, 1])
        
        return batch_I_r


class tf_grid_sampe(tf.keras.Model):
    '''
    见网页：https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
    '''
    def __init__(self, img=None, x=None, y=None):
        super(tf_grid_sampe, self).__init__()

    self.grid_sample = self.bilinear_sampler(img, x, y)

    # @tf.function
    def call(self, img, x, y):
        return self.grid_sample(img, x, y)


    def spatial_transformer_network(self, input_fmap, theta, out_dims=None, **kwargs):
        """
        Spatial Transformer Network layer implementation as described in [1].
        The layer is composed of 3 elements:
        - localization_net: takes the original image as input and outputs
        the parameters of the affine transformation that should be applied
        to the input image.
        - affine_grid_generator: generates a grid of (x,y) coordinates that
        correspond to a set of points where the input should be sampled
        to produce the transformed output.
        - bilinear_sampler: takes as input the original image and the grid
        and produces the output image using bilinear interpolation.
        Input
        -----
        - input_fmap: output of the previous layer. Can be input if spatial
        transformer layer is at the beginning of architecture. Should be
        a tensor of shape (B, H, W, C).
        - theta: affine transform tensor of shape (B, 6). Permits cropping,
        translation and isotropic scaling. Initialize to identity matrix.
        It is the output of the localization network.
        Returns
        -------
        - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
        Notes
        -----
        [1]: 'Spatial Transformer Networks', Jaderberg et. al,
            (https://arxiv.org/abs/1506.02025)
        """
        # grab input dimensions
        B = tf.shape(input_fmap)[0]
        H = tf.shape(input_fmap)[1]
        W = tf.shape(input_fmap)[2]

        # reshape theta to (B, 2, 3)
        theta = tf.reshape(theta, [B, 2, 3])

        # generate grids of same size or upsample/downsample if specified
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = affine_grid_generator(out_H, out_W, theta)
        else:
            batch_grids = affine_grid_generator(H, W, theta)

        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]

        # sample input with grid to get output
        out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

        return out_fmap


    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)


    def affine_grid_generator(self, height, width, theta):
        """
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.
        Input
        -----
        - height: desired height of grid/output. Used
        to downsample or upsample.
        - width: desired width of grid/output. Used
        to downsample or upsample.
        - theta: affine transform matrices of shape (num_batch, 2, 3).
        For each image in the batch, we have 6 theta parameters of
        the form (2x3) that define the affine transformation T.
        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
        The 2nd dimension has 2 components: (x, y) which are the
        sampling points of the original image for each point in the
        target image.
        Note
        ----
        [1]: the affine transformation allows cropping, translation,
            and isotropic scaling.
        """
        num_batch = tf.shape(theta)[0]

        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

        return batch_grids


    def bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.
        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.
        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = get_pixel_value(img, x0, y0)
        Ib = get_pixel_value(img, x0, y1)
        Ic = get_pixel_value(img, x1, y0)
        Id = get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return out