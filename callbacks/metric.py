# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: metric.py

@time: 2020/5/21 7:33

@desc:

"""

import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

from data_loader import MultiTaskClsDataGenerator
from config import ModelConfig
from utils import precision_recall_fscore


class MultiTaskMetric(Callback):
    def __init__(self, valid_generator: MultiTaskClsDataGenerator, config: ModelConfig):
        self.valid_generator = valid_generator
        self.config = config
        super(MultiTaskMetric, self).__init__()

    def predict_multitask(self, x):
        pred_cate1, pred_cate2, pred_cate3 = self.model.predict(x)
        pred_cate1 = np.argmax(pred_cate1, axis=-1).tolist()
        pred_cate2 = np.argmax(pred_cate2, axis=-1).tolist()
        pred_cate3 = np.argmax(pred_cate3, axis=-1).tolist()
        return pred_cate1, pred_cate2, pred_cate3

    def predict_one_task(self, x):
        pred_all_cate = np.argmax(self.model.predict(x), axis=-1).tolist()
        pred_all_cate = map(self.config.idx2all_cate.get, pred_all_cate)
        pred_all_cate_split = map(lambda c: c.split('|'), pred_all_cate)
        pred_cate1, pred_cate2, pred_cate3 = zip(*list(pred_all_cate_split))

        pred_cate1 = list(map(self.config.cate1_vocab.get, pred_cate1))
        pred_cate2 = list(map(self.config.cate2_vocab.get, pred_cate2))
        pred_cate3 = list(map(self.config.cate3_vocab.get, pred_cate3))
        return pred_cate1, pred_cate2, pred_cate3

    def on_epoch_end(self, epoch, logs={}):
        pred_cate1_list, true_cate1_list = [], []
        pred_cate2_list, true_cate2_list = [], []
        pred_cate3_list, true_cate3_list = [], []
        for x_valid, y_valid in self.valid_generator:
            if self.config.use_multi_task:
                pred_cate1, pred_cate2, pred_cate3 = self.predict_multitask(x_valid)
            else:
                pred_cate1, pred_cate2, pred_cate3 = self.predict_one_task(x_valid)
            pred_cate1_list.extend(pred_cate1)
            pred_cate2_list.extend(pred_cate2)
            pred_cate3_list.extend(pred_cate3)

            true_cate1_list.extend(np.argmax(y_valid[0], axis=-1).tolist())
            true_cate2_list.extend(np.argmax(y_valid[1], axis=-1).tolist())
            true_cate3_list.extend(np.argmax(y_valid[2], axis=-1).tolist())

        print(f'Logging Info - Epoch: {epoch+1} evaluation:')
        eval_results = precision_recall_fscore(pred_cate1_list=pred_cate1_list, true_cate1_list=true_cate1_list,
                                               pred_cate2_list=pred_cate2_list, true_cate2_list=true_cate2_list,
                                               pred_cate3_list=pred_cate3_list, true_cate3_list=true_cate3_list)
        logs.update(eval_results)


class MaskedMultiTaskMetric(Callback):
    def __init__(self,
                 valid_generator: MultiTaskClsDataGenerator,
                 cate1_model,
                 cate2_model,
                 cate3_model,
                 config: ModelConfig):
        self.valid_generator = valid_generator
        self.cate1_model = cate1_model
        self.cate2_model = cate2_model
        self.cate3_model = cate3_model
        self.config = config
        assert self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3
        if self.config.use_mask_for_cate2:
            assert self.valid_generator.cate1_to_cate2_matrix is not None
        if self.config.use_mask_for_cate3:
            assert self.valid_generator.cate_to_cate3_matrix is not None
        super(MaskedMultiTaskMetric, self).__init__()

    def predict_masked_multitask(self, x):
        pred_cate1 = self.cate1_model.predict(x)
        pred_cate1 = np.argmax(pred_cate1, axis=-1)

        inputs_for_cate2_model = x
        if self.config.use_mask_for_cate2:
            input_cate2_mask = self.valid_generator.cate1_to_cate2_matrix[pred_cate1]
            if isinstance(inputs_for_cate2_model, list):
                inputs_for_cate2_model = inputs_for_cate2_model + [input_cate2_mask]
            else:
                inputs_for_cate2_model = [inputs_for_cate2_model, input_cate2_mask]
        pred_cate2 = self.cate2_model.predict(inputs_for_cate2_model)
        pred_cate2 = np.argmax(pred_cate2, axis=-1)

        inputs_for_cate3_model = x
        if self.config.use_mask_for_cate3:
            if self.config.cate3_mask_type == 'cate1':
                input_cate3_mask = self.valid_generator.cate_to_cate3_matrix[pred_cate1]
            elif self.config.cate3_mask_type == 'cate2':
                input_cate3_mask = self.valid_generator.cate_to_cate3_matrix[pred_cate2]
            else:
                raise ValueError(f'`cate3_mask_type` not understood: {self.config.cate3_mask_type}')
            if isinstance(inputs_for_cate3_model, list):
                inputs_for_cate3_model = inputs_for_cate3_model + [input_cate3_mask]
            else:
                inputs_for_cate3_model = [inputs_for_cate3_model, input_cate3_mask]
        pred_cate3 = self.cate3_model.predict(inputs_for_cate3_model)
        pred_cate3 = np.argmax(pred_cate3, axis=-1)

        return pred_cate1.tolist(), pred_cate2.tolist(), pred_cate3.tolist()

    def on_epoch_end(self, epoch, logs={}):
        pred_cate1_list, true_cate1_list = [], []
        pred_cate2_list, true_cate2_list = [], []
        pred_cate3_list, true_cate3_list = [], []
        for x_valid, y_valid in self.valid_generator:
            pred_cate1, pred_cate2, pred_cate3 = self.predict_masked_multitask(x_valid)

            pred_cate1_list.extend(pred_cate1)
            pred_cate2_list.extend(pred_cate2)
            pred_cate3_list.extend(pred_cate3)

            true_cate1_list.extend(np.argmax(y_valid[0], axis=-1).tolist())
            true_cate2_list.extend(np.argmax(y_valid[1], axis=-1).tolist())
            true_cate3_list.extend(np.argmax(y_valid[2], axis=-1).tolist())

        print(f'Logging Info - Epoch: {epoch + 1} evaluation:')
        eval_results = precision_recall_fscore(pred_cate1_list=pred_cate1_list, true_cate1_list=true_cate1_list,
                                               pred_cate2_list=pred_cate2_list, true_cate2_list=true_cate2_list,
                                               pred_cate3_list=pred_cate3_list, true_cate3_list=true_cate3_list)
        logs.update(eval_results)
