# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import yaml
# import paddle
# import paddle.distributed as dist
# dist.get_world_size()
# paddle.seed(2)

# from ppocr.data import build_dataloader # 
from data_lib.data import tf_build_dataloader # 

# from ppocr.modeling.architectures import build_model #
from model.modeling.architectures import tf_build_model #

from model.losses import build_loss # 
from model.optimizer import build_optimizer #
from model.postprocess import build_post_process  #
from model.metrics import build_metric            #

# from ppocr.utils.save_load import init_model   # 
from model.utils.save_load import tf_init_model   # 
import tools.program as program
import tensorflow as tf
tf.compat.v1.set_random_seed(2)

'''
def main(config, device, logger, vdl_writer):
    # init dist environment
    # if config['Global']['distributed']:
    #     dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])
    # if config['Global']['distributed']:
    #     model = paddle.DataParallel(model)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)

'''

def tf_main(config, device, logger, vdl_writer):

    global_config = config['Global']

    # build dataloader
    train_dataloader = tf_build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = tf_build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = tf_build_model(config['Architecture'])

    model.build(input_shape=(4,3,512,512))
    model.summary()
    #print(model.backbone())

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader)
        # parameters=model.parameters()
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,                        
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-08                                            
                                            )

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = tf_init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train
    program.tf_train(config, 
                  train_dataloader, 
                  valid_dataloader, 
                  device, 
                  model,
                  loss_class, 
                  optimizer, 
                  lr_scheduler, 
                  post_process_class,
                  eval_class, 
                  pre_best_model_dict, 
                  logger, 
                  vdl_writer)


def test_reader(config, device, logger):
    loader = tf_build_dataloader(config, 'Train', device, logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':

    # paddle
    #config, device, logger, vdl_writer = program.preprocess(is_train=True)
    #main(config, device, logger, vdl_writer)

    # tf
    config, device, logger, vdl_writer = program.tf_preprocess(is_train=True)
    tf_main(config, device, logger, vdl_writer)
    # test_reader(config, device, logger)
