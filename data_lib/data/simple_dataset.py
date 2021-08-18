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
import numpy as np
import os
import random
# from paddle.io import Dataset

from .imaug import transform, create_operators
import tensorflow as tf
import numpy as np

'''
class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.outs_list = []

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            
            data_line = data_line.decode('utf-8')
            
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
                
            outs = transform(data, self.ops)
    
        except Exception as e:
            self.logger.error("When parsing line {}, error happened with msg: {}".format(data_line, e))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__()) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        # if self.mode != "train":
        #     print("outs shape:",len(outs))
        return outs

    def __len__(self):
        return  len(self.data_idx_order_list)

'''

class tf_SimpleDataSet(tf.keras.utils.Sequence):
    def __init__(self, config, mode, logger, seed=None):
        super(tf_SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        self.batch_size = loader_config['batch_size_per_card']
        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        idx_list = []
        idx = random.randint(0, len(self.data_idx_order_list)-1)
        while(idx in idx_list):
            idx = random.randint(0, len(self.data_idx_order_list)-1)
        idx_list.append(idx)
        out_list = []
        image = []
        score_map = []
        border_map = []
        training_mask = []
        tvo_map = []
        tco_map = []

        # outs, image, score_map, border_map, training_mask, tvo_map, tco_map <= self.data_generation(idx)
        for i in range(int(self.batch_size)):
            outs = self.data_generation(idx)                      
            
            image.append(outs[0])
            score_map.append(outs[1])
            border_map.append(outs[2])
            training_mask.append(outs[3])
            if self.mode == "train":
                tvo_map.append(outs[4])
                tco_map.append(outs[5])
        
        if self.mode == "train":
            image = tf.stack(image, axis=0)
            score_map = tf.stack(score_map, axis=0)
            border_map = tf.stack(border_map, axis=0)
            training_mask = tf.stack(training_mask, axis=0)
            tvo_map = tf.stack(tvo_map, axis=0)
            tco_map = tf.stack(tco_map, axis=0)

            # image = tf.transpose(image, perm=[0, 2, 3, 1])
            # score_map = tf.transpose(score_map, perm=[0, 2, 3, 1])
            # border_map = tf.transpose(border_map, perm=[0, 2, 3, 1])
            # training_mask = tf.transpose(training_mask, perm=[0, 2, 3, 1])
            # tvo_map = tf.transpose(tvo_map, perm=[0, 2, 3, 1])
            # tco_map = tf.transpose(tco_map, perm=[0, 2, 3, 1])

            # print("image shape:", image.shape)
            
            return [image, score_map, border_map, training_mask, tvo_map, tco_map]
        if self.mode == "eval":
            # print("===============================================================")  
            image = tf.stack(image, axis=0)
            score_map = tf.stack(score_map, axis=0)
            border_map = tf.stack(border_map, axis=0)
            training_mask = tf.stack(training_mask, axis=0)

            # print("images shape:", image.shape)
            # print("score_map shape:", score_map.shape)
            # print("border_map shape:", border_map.shape)
            # print("training_mask shape:", training_mask.shape)

            return [image, score_map, border_map, training_mask]

    def __len__(self):
        return  round(len(self.data_idx_order_list)/self.batch_size)

    def data_generation(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]        
        try:
        
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                
                data['image'] = img
                
            outs = transform(data, self.ops)
            
        except Exception as e:
            self.logger.error("When parsing line {}, error happened with msg: {}".format(data_line, e))
            outs = None
        if outs is None:
            length = int(self.__len__()*self.batch_size)
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(length) if self.mode == "train" else (idx + 1) % length
            return self.data_generation(rnd_idx)
        return outs

