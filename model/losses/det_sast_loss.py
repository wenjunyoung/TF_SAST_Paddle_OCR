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
# from .det_basic_loss import DiceLoss
from .det_basic_loss import tf_DiceLoss
import numpy as np

import tensorflow as tf


'''
class SASTLoss(nn.Layer):
    """
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(SASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor

        tf - reshape ---> expand
        tf - reduce_sum ---> sum
        tf - split -----> split
        tf - cast ------> cast
        tf - math.cast ---> abs
        """

        f_score = predicts['f_score']
        f_border = predicts['f_border']
        f_tvo = predicts['f_tvo']
        f_tco = predicts['f_tco']

        # print("f_score shape:", f_score.shape)
        # print("f_border shape:", f_border.shape)
        # print("f_tvo shape:", f_tvo.shape)
        # print("f_tco shape:", f_tco.shape)

        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]

        #score_loss
        intersection = paddle.sum(f_score * l_score * l_mask)
        union = paddle.sum(f_score * l_mask) + paddle.sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)
        #print('score_loss=', score_loss)

        #border loss
        l_border_split, l_border_norm = paddle.split(l_border, num_or_sections=[4, 1], axis=1)
        f_border_split = f_border

        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = paddle.expand(x=l_border_norm, shape=border_ex_shape)  

        # l_border_norm:[4, 1, 128, 128] 
        # l_border_split:[4, 4, 128, 128]  
        # border_ex_shape:(4,) 
        # l_border_norm_split:[4, 4, 128, 128]

        l_border_score = paddle.expand(x=l_score, shape=border_ex_shape)
        l_border_mask = paddle.expand(x=l_mask, shape=border_ex_shape)

        border_diff = l_border_split - f_border_split
        abs_border_diff = paddle.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = paddle.cast(border_sign, dtype='float32')
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = paddle.sum(border_out_loss * l_border_score * l_border_mask) / (paddle.sum(l_border_score * l_border_mask) + 1e-5)
        #print('border_loss=', border_loss)

        #tvo_loss
        l_tvo_split, l_tvo_norm = paddle.split(l_tvo, num_or_sections=[8, 1], axis=1)
        f_tvo_split = f_tvo

        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        l_tvo_norm_split = paddle.expand(x=l_tvo_norm, shape=tvo_ex_shape)
        l_tvo_score = paddle.expand(x=l_score, shape=tvo_ex_shape)
        l_tvo_mask = paddle.expand(x=l_mask, shape=tvo_ex_shape)
        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = paddle.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = paddle.cast(tvo_sign, dtype='float32')
        tvo_sign.stop_gradient = True
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = paddle.sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / (paddle.sum(l_tvo_score * l_tvo_mask) + 1e-5)
        # print('tvo_loss=', tvo_loss)

        #tco_loss
        l_tco_split, l_tco_norm = paddle.split(
            l_tco, num_or_sections=[2, 1], axis=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])
        l_tco_norm_split = paddle.expand(x=l_tco_norm, shape=tco_ex_shape)
        l_tco_score = paddle.expand(x=l_score, shape=tco_ex_shape)
        l_tco_mask = paddle.expand(x=l_mask, shape=tco_ex_shape)

        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = paddle.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = paddle.cast(tco_sign, dtype='float32')
        tco_sign.stop_gradient = True
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = paddle.sum(tco_out_loss * l_tco_score * l_tco_mask) / (paddle.sum(l_tco_score * l_tco_mask) + 1e-5)
        # print('tco_loss=', tco_loss)

        # total loss
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = score_loss * score_lw + border_loss * border_lw + tvo_loss * tvo_lw + tco_loss * tco_lw
        
        losses = {'loss':total_loss, "score_loss":score_loss, "border_loss":border_loss, 'tvo_loss':tvo_loss, 'tco_loss':tco_loss}
        return losses

'''

'''
class tf_SASTLoss(tf.keras.layers.Layer):
    """
    tensorflow 版本的SAST loss 函数
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(tf_SASTLoss, self).__init__()
        # self.dice_loss = DiceLoss(eps=eps)

    def call(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor

        tf - reshape ---> expand
        tf - reduce_sum ---> sum
        tf - split -----> split
        tf - cast ------> cast
        tf - math.cast ---> abs
        """
        # paddle Tensor转numpy
        f_score = predicts['f_score'].numpy()
        f_border = predicts['f_border'].numpy()
        f_tvo = predicts['f_tvo'].numpy()
        f_tco = predicts['f_tco'].numpy()

        p_f_score = predicts['f_score']
        p_f_border = predicts['f_border']
        p_f_tvo = predicts['f_tvo']
        p_f_tco = predicts['f_tco']
                
        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]
        p_l_score, p_l_border, p_l_mask, p_l_tvo, p_l_tco = l_score, l_border, l_mask, l_tvo, l_tco
        l_score, l_border, l_mask, l_tvo, l_tco =l_score.numpy(), l_border.numpy(), l_mask.numpy(), l_tvo.numpy(), l_tco.numpy()

        # numpy 转 tf Tensor
        f_score = tf.convert_to_tensor(f_score)
        f_border = tf.convert_to_tensor(f_border)
        f_tvo = tf.convert_to_tensor(f_tvo)
        f_tco = tf.convert_to_tensor(f_tco)

        l_score = tf.convert_to_tensor(l_score)
        l_border = tf.convert_to_tensor(l_border)
        l_mask = tf.convert_to_tensor(l_mask)
        l_tvo = tf.convert_to_tensor(l_tvo)
        l_tco = tf.convert_to_tensor(l_tco)

        #score_loss
        intersection =tf.math.reduce_sum(f_score * l_score * l_mask)
        union = tf.math.reduce_sum(f_score * l_mask) + tf.math.reduce_sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)

        # border loss
        # l_border_norm:[4, 1, 128, 128] 
        # l_border_split:[4, 4, 128, 128]
        # border_ex_shape:(4,)         
        l_border_split, l_border_norm = tf.split(l_border, num_or_size_splits=[4, 1], axis=1)
        f_border_split = f_border
        p_f_border_split = p_f_border

        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = tf.tile(l_border_norm, multiples=(1, 4, 1, 1))
        l_border_score = tf.tile(l_score, multiples=(1, 4, 1, 1))
        l_border_mask = tf.tile(l_mask, multiples=(1, 4, 1, 1))

        border_diff = l_border_split - f_border_split
        abs_border_diff = tf.math.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = tf.cast(border_sign, dtype='float32')
        border_sign = tf.stop_gradient(border_sign)
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = tf.math.reduce_sum(border_out_loss * l_border_score * l_border_mask) / \
                                        (tf.math.reduce_sum(l_border_score * l_border_mask) + 1e-5)

        #tvo_loss
        l_tvo_split, l_tvo_norm = tf.split(l_tvo, num_or_size_splits=[8, 1], axis=1)
        f_tvo_split = f_tvo

        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        # l_tvo_norm_split = tf.reshape(l_tvo_norm, shape=tvo_ex_shape)
        # l_tvo_score = tf.reshape(l_score, shape=tvo_ex_shape)
        # l_tvo_mask = tf.reshape(l_mask, shape=tvo_ex_shape)

        l_tvo_norm_split = tf.tile(l_tvo_norm, multiples=(1, 8, 1, 1))
        l_tvo_score = tf.tile(l_score, multiples=(1, 8, 1, 1))
        l_tvo_mask = tf.tile(l_mask, multiples=(1, 8, 1, 1))

        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = tf.math.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = tf.cast(tvo_sign, dtype='float32')
        tvo_sign = tf.stop_gradient(tvo_sign)
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = tf.math.reduce_sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / \
                             (tf.math.reduce_sum(l_tvo_score * l_tvo_mask) + 1e-5)

        #tco_loss
        l_tco_split, l_tco_norm = tf.split(l_tco, num_or_size_splits=[2, 1], axis=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])

        # l_tco_norm_split = tf.reshape(l_tco_norm, shape=tco_ex_shape)
        # l_tco_score = tf.reshape(l_score, shape=tco_ex_shape)
        # l_tco_mask = tf.reshape(l_mask, shape=tco_ex_shape)

        l_tco_norm_split = tf.tile(l_tco_norm, multiples=(1, 2, 1, 1))
        l_tco_score = tf.tile(l_score, multiples=(1, 2, 1, 1))
        l_tco_mask = tf.tile(l_mask, multiples=(1, 2, 1, 1))

        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = tf.math.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = tf.cast(tco_sign, dtype='float32')
        tco_sign = tf.stop_gradient(tco_sign)
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = tf.math.reduce_sum(tco_out_loss * l_tco_score * l_tco_mask) / \
                             (tf.math.reduce_sum(l_tco_score * l_tco_mask) + 1e-5)

        # total loss
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = score_loss * score_lw + border_loss * border_lw + tvo_loss * tvo_lw + tco_loss * tco_lw

        # tf Tensor 转 numpy
        total_loss = total_loss.numpy()
        score_loss = score_loss.numpy()
        border_loss = border_loss.numpy()
        tvo_loss = tvo_loss.numpy()
        tco_loss = tco_loss.numpy()

        # numpy 转 tf paddle Tensor 
        total_loss = paddle.to_tensor(total_loss)
        score_loss = paddle.to_tensor(score_loss)
        border_loss = paddle.to_tensor(border_loss)
        tvo_loss = paddle.to_tensor(tvo_loss)
        tco_loss = paddle.to_tensor(tco_loss)

        losses = {'loss':total_loss, "score_loss":score_loss, "border_loss":border_loss, 'tvo_loss':tvo_loss, 'tco_loss':tco_loss}
        return losses
'''


class tf_SASTLoss(tf.keras.layers.Layer):
    """
    tensorflow 版本的SAST loss 函数
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(tf_SASTLoss, self).__init__()
        # self.dice_loss = DiceLoss(eps=eps)

    def call(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor

        tf - reshape ---> expand
        tf - reduce_sum ---> sum
        tf - split -----> split
        tf - cast ------> cast
        tf - math.cast ---> abs
        """
        f_score = predicts['f_score']
        f_border = predicts['f_border']
        f_tvo = predicts['f_tvo']
        f_tco = predicts['f_tco']

        # print("f_score shape:", f_score.shape)
        # print("f_border shape:", f_border.shape)
        # print("f_tvo shape:", f_tvo.shape)
        # print("f_tco shape:", f_tco.shape)
                
        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]
        # print("l_score shape:", l_score.shape)
        # print("l_border shape:", l_border.shape)
        # print("l_mask shape:", l_mask.shape)
        # print("l_tvo shape:", l_tvo.shape)
        # print("l_tco shape:", l_tco.shape)

        #score_loss
        intersection =tf.math.reduce_sum(f_score * l_score * l_mask)
        union = tf.math.reduce_sum(f_score * l_mask) + tf.math.reduce_sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)

        # border loss
        # l_border_norm:[4, 1, 128, 128] 
        # l_border_split:[4, 4, 128, 128]
        # border_ex_shape:(4,)         
        l_border_split, l_border_norm = tf.split(l_border, num_or_size_splits=[4, 1], axis=1)
        f_border_split = f_border

        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = tf.tile(l_border_norm, multiples=(1, 4, 1, 1))
        l_border_score = tf.tile(l_score, multiples=(1, 4, 1, 1))
        l_border_mask = tf.tile(l_mask, multiples=(1, 4, 1, 1))

        # print("l_border_split shape:", l_border_split.shape)
        # print("f_border_split shape:", f_border_split.shape)
        border_diff = l_border_split - f_border_split
        # print("border_diff shape:", border_diff.shape)

        abs_border_diff = tf.math.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = tf.cast(border_sign, dtype='float32')
        border_sign = tf.stop_gradient(border_sign)
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = tf.math.reduce_sum(border_out_loss * l_border_score * l_border_mask) / \
                                        (tf.math.reduce_sum(l_border_score * l_border_mask) + 1e-5)

        #tvo_loss
        l_tvo_split, l_tvo_norm = tf.split(l_tvo, num_or_size_splits=[8, 1], axis=1)
        f_tvo_split = f_tvo

        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        # l_tvo_norm_split = tf.reshape(l_tvo_norm, shape=tvo_ex_shape)
        # l_tvo_score = tf.reshape(l_score, shape=tvo_ex_shape)
        # l_tvo_mask = tf.reshape(l_mask, shape=tvo_ex_shape)

        l_tvo_norm_split = tf.tile(l_tvo_norm, multiples=(1, 8, 1, 1))
        l_tvo_score = tf.tile(l_score, multiples=(1, 8, 1, 1))
        l_tvo_mask = tf.tile(l_mask, multiples=(1, 8, 1, 1))

        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = tf.math.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = tf.cast(tvo_sign, dtype='float32')
        tvo_sign = tf.stop_gradient(tvo_sign)
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = tf.math.reduce_sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / \
                             (tf.math.reduce_sum(l_tvo_score * l_tvo_mask) + 1e-5)

        #tco_loss
        l_tco_split, l_tco_norm = tf.split(l_tco, num_or_size_splits=[2, 1], axis=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])

        # l_tco_norm_split = tf.reshape(l_tco_norm, shape=tco_ex_shape)
        # l_tco_score = tf.reshape(l_score, shape=tco_ex_shape)
        # l_tco_mask = tf.reshape(l_mask, shape=tco_ex_shape)

        l_tco_norm_split = tf.tile(l_tco_norm, multiples=(1, 2, 1, 1))
        l_tco_score = tf.tile(l_score, multiples=(1, 2, 1, 1))
        l_tco_mask = tf.tile(l_mask, multiples=(1, 2, 1, 1))

        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = tf.math.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = tf.cast(tco_sign, dtype='float32')
        tco_sign = tf.stop_gradient(tco_sign)
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = tf.math.reduce_sum(tco_out_loss * l_tco_score * l_tco_mask) / \
                             (tf.math.reduce_sum(l_tco_score * l_tco_mask) + 1e-5)

        # total loss
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = score_loss * score_lw + border_loss * border_lw + tvo_loss * tvo_lw + tco_loss * tco_lw

        losses = {'loss':total_loss, "score_loss":score_loss, "border_loss":border_loss, 'tvo_loss':tvo_loss, 'tco_loss':tco_loss}
        return losses



