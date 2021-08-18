# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import errno
import os
import pickle
import six

# import paddle
import tensorflow as tf

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))

'''
def load_dygraph_pretrain(model, logger, path=None, load_static_weights=False):
    print('path===>', path)
    path = path[:-1]   # for res50_vd_ssld
    print('after path===>', path)
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    if load_static_weights:
        pre_state_dict = paddle.static.load_program_state(path)
        param_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            weight_name = weight_name.replace('binarize', '').replace(
                'thresh', '')  # for DB
            if weight_name in pre_state_dict.keys():
                # logger.info('Load weight: {}, shape: {}'.format(
                #     weight_name, pre_state_dict[weight_name].shape))
                if 'encoder_rnn' in key:
                    # delete axis which is 1
                    pre_state_dict[weight_name] = pre_state_dict[
                        weight_name].squeeze()
                    # change axis
                    if len(pre_state_dict[weight_name].shape) > 1:
                        pre_state_dict[weight_name] = pre_state_dict[
                            weight_name].transpose((1, 0))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                param_state_dict[key] = model_dict[key]
        model.set_state_dict(param_state_dict)
        return

    param_state_dict = paddle.load(path + '.pdparams')
    model.set_state_dict(param_state_dict)
    return


def init_model(config, model, logger, optimizer=None, lr_scheduler=None):
    """
    load model from checkpoint or pretrained_model
    """
    gloabl_config = config['Global']
    checkpoints = gloabl_config.get('checkpoints')
    pretrained_model = gloabl_config.get('pretrained_model')
    best_model_dict = {}
    if checkpoints:
        assert os.path.exists(checkpoints + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        para_dict = paddle.load(checkpoints + '.pdparams')
        opti_dict = paddle.load(checkpoints + '.pdopt')
        model.set_state_dict(para_dict)
        if optimizer is not None:
            optimizer.set_state_dict(opti_dict)

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1

        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        load_static_weights = gloabl_config.get('load_static_weights', False)
        if not isinstance(pretrained_model, list):
            pretrained_model = [pretrained_model]
        if not isinstance(load_static_weights, list):
            load_static_weights = [load_static_weights] * len(pretrained_model)
        for idx, pretrained in enumerate(pretrained_model):
            load_static = load_static_weights[idx]
            load_dygraph_pretrain(
                model, logger, path=pretrained, load_static_weights=load_static)
            logger.info("load pretrained model from {}".format(
                pretrained_model))
    else:
        logger.info('train from scratch')
    return best_model_dict
'''

def tf_init_model(config, model, logger, optimizer=None, lr_scheduler=None):
    """
    load model from checkpoint or pretrained_model
    """
    gloabl_config = config['Global']
    checkpoints_prefix = gloabl_config.get('checkpoints')
    pretrained_model = gloabl_config.get('pretrained_model')
    best_model_dict = {}

    if checkpoints_prefix:
        # assert os.path.exists(checkpoints_prefix + "model_latest-800.index"), "Given dir {} checkpoint not exist.".format(checkpoints_prefix)

        # 载入模型
        checkpoint = tf.train.Checkpoint(tf_model=model, tf_optimizer=optimizer)
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoints))
        '''
        save_path_with_prefix_and_index 是之前保存的文件的目录 + 前缀 + 编号。
        例如，调用 checkpoint.restore('./save/model.ckpt-1') 就可以载入前缀为 model.ckpt ，序号为 1 的文件来恢复模型
        '''
        # checkpoint.restore(checkpoints_prefix+'model_latest' + '-310') # save_path_with_prefix_and_index
        checkpoint.restore(tf.train.latest_checkpoint(checkpoints_prefix))

        # 保存模型的状态信息，包括best model
        if os.path.exists(checkpoints_prefix + '.states'):
            with open(checkpoints_prefix + '.states', 'rb') as f:
                states_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1

        logger.info("resume from {}".format(checkpoints_prefix))

    elif pretrained_model:
        # model = tf.saved_model.load(pretrained_model)
        model = tf.keras.models.load_model(pretrained_model, compile=False)
        logger.info("load pretrained model from {}".format(pretrained_model))
    else:
        logger.info('train from scratch')
    return best_model_dict

'''
def save_model(model,
               optimizer,
               model_path,
               logger,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(model.state_dict(), model_prefix + '.pdparams')
    paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')

    # save metric and config
    with open(model_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
'''

def tf_save_model(model,
               optimizer,
               model_path, # save_model_dir
               logger,
               is_best=False,
               is_checkpoint=False,
               epoch=0,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)

    _mkdir_if_not_exist(model_path+'checkpoint/', logger)
    model_checkpoint = os.path.join(model_path+'checkpoint/', prefix)

    _mkdir_if_not_exist(model_path+'latest/', logger)
    model_latest = os.path.join(model_path+'latest/', prefix)

    # 保存 best mdoel
    if is_best:
        # tf.saved_model.save(model, model_prefix)
        tf.keras.models.save_model(model, model_prefix)

    # 保存 checkpoint
    if is_checkpoint:

        # 保存 pb model
        # tf.saved_model.save(model, model_latest)
        tf.keras.models.save_model(model, model_latest)

        # 保存 checkpoint model
        checkpoint = tf.train.Checkpoint(tf_model=model, tf_optimizer=optimizer)
        # checkpoint.save(model_path+'model_latest')
        manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint, checkpoint_name='model_latest', max_to_keep=5)
        manager.save(checkpoint_number=epoch)    

    # save metric and config
    with open(model_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    elif is_checkpoint:
        logger.info('save is_checkpoint is to {}'.format(model_latest))
    else:
        logger.info("save model in {}".format(model_prefix))        
