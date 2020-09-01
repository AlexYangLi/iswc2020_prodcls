# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: multitask_classify_loader.py

@time: 2020/5/21 21:12

@desc:

"""

import os
import nltk
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical

from config import NLTK_DATA
from utils import pad_sequences_1d, get_bert_tokenizer
from .base_loader import load_data


class MultiTaskClsDataGenerator(Sequence):
    def __init__(self,
                 data_type,
                 batch_size,
                 use_multi_task=True,
                 input_type='name_desc',
                 use_word_input=True,
                 word_vocab=None,
                 use_bert_input=False,
                 use_pair_input=False,
                 bert_model_type=None,
                 max_len=None,
                 cate1_vocab=None,
                 cate2_vocab=None,
                 cate3_vocab=None,
                 all_cate_vocab=None,
                 use_mask_for_cate2=False,
                 use_mask_for_cate3=False,
                 cate3_mask_type=None,
                 cate1_to_cate2=None,
                 cate_to_cate3=None,
                 train_on_cv=False,
                 cv_random_state=42,
                 cv_fold=5,
                 cv_index=0,
                 exchange_pair=False,
                 exchange_threshold=0.1,
                 cate3_count_dict=None,
                 use_pseudo_label=False,
                 pseudo_path=None,
                 pseudo_random_state=42,
                 pseudo_rate=0.1,
                 pseudo_index=0):
        self.data_type = data_type
        self.train_on_cv = train_on_cv
        self.data = load_data(data_type, train_on_cv=train_on_cv, cv_random_state=cv_random_state,
                              cv_fold=cv_fold, cv_index=cv_index)
        # data augmentation, only for training set
        if self.data_type == 'train':
            if use_pseudo_label:
                self.add_pseudo_label(pseudo_path, pseudo_random_state, pseudo_rate, pseudo_index)
            if exchange_pair:
                self.exchange_pair_data(cate3_count_dict, exchange_threshold)

        self.data_size = len(self.data['name'])
        self.indices = np.arange(self.data_size)
        if self.data_type == 'train':
            np.random.shuffle(self.indices)  # only shuffle for training set, we can't shuffle validation and test set!!
        self.batch_size = batch_size
        self.steps = int(np.ceil(self.data_size / self.batch_size))
        self.use_multi_task = use_multi_task
        self.input_type = input_type
        self.use_word_input = use_word_input
        self.word_vocab = word_vocab
        self.use_bert_input = use_bert_input

        if use_word_input:
            nltk.data.path.append(NLTK_DATA)
            assert word_vocab is not None
            self.word_vocab = word_vocab
        elif use_bert_input:
            assert bert_model_type is not None
            assert max_len is not None
            self.bert_model_type = bert_model_type
            self.bert_tokenizer = get_bert_tokenizer(bert_model_type)

        if input_type != 'name_desc':
            assert not use_pair_input
        self.use_pair_input = use_pair_input
        self.max_len = max_len

        self.cate1_vocab = cate1_vocab
        self.cate2_vocab = cate2_vocab
        self.cate3_vocab = cate3_vocab
        self.all_cate_vocab = all_cate_vocab
        if not use_multi_task:
            assert self.all_cate_vocab is not None
        else:
            assert self.cate1_vocab is not None and self.cate2_vocab is not None and self.cate3_vocab is not None

        self.use_mask_for_cate2 = use_mask_for_cate2
        self.use_mask_for_cate3 = use_mask_for_cate3
        self.cate3_mask_type = cate3_mask_type
        if self.use_mask_for_cate2:
            assert self.use_multi_task
            assert cate1_to_cate2 is not None
            self.cate1_to_cate2_matrix = self.create_mask_matrix(cate1_to_cate2, len(cate1_vocab), len(cate2_vocab))
        else:
            self.cate1_to_cate2_matrix = None
        if self.use_mask_for_cate3:
            assert self.use_multi_task
            assert self.cate3_mask_type in ['cate1', 'cate2']
            self.cate_to_cate3_matrix = self.create_mask_matrix(
                cate_to_cate3,
                len(cate1_vocab) if self.cate3_mask_type == 'cate1' else len(cate2_vocab),
                len(cate3_vocab)
            )
        else:
            self.cate_to_cate3_matrix = None

    def exchange_pair_data(self, cate3_count_dict, exchange_threshold):
        added_data = {
            'id': [], 'name': [], 'desc': [], 'cate1': [], 'cate2': [], 'cate3': []
        }
        for i in range(len(self.data['id'])):
            cate3 = self.data['cate3'][i]
            if cate3 in cate3_count_dict and cate3_count_dict[cate3] <= exchange_threshold:
                added_data['id'].append(self.data['id'][i])
                # exchange name and desc
                added_data['name'].append(self.data['desc'][i])
                added_data['desc'].append(self.data['name'][i])
                # keep the labels
                added_data['cate1'].append(self.data['cate1'][i])
                added_data['cate2'].append(self.data['cate2'][i])
                added_data['cate3'].append(self.data['cate3'][i])

        for key in added_data:
            self.data[key].extend(added_data[key])

    def add_pseudo_label(self, pseudo_path, pseudo_random_state=42, pseudo_rate=0.1, pseudo_index=0):
        pseudo_label_data = {
            'id': [], 'name': [], 'desc': [], 'cate1': [], 'cate2': [], 'cate3': []
        }
        with open(pseudo_path, 'r', encoding='utf8') as reader:
            lines = reader.readlines()
            pseudo_data_size = len(lines)
            if pseudo_rate < 1:
                np.random.seed(pseudo_random_state)
                sample_pseudo_size = int(pseudo_data_size * pseudo_rate)
                sample_indices = np.random.choice(pseudo_data_size, sample_pseudo_size,
                                                  replace=False)
            elif pseudo_rate == 1:
                sample_indices = range(pseudo_data_size)
            else:
                sample_pseudo_size = int(pseudo_data_size / pseudo_rate)
                start = pseudo_index * sample_pseudo_size
                end = (pseudo_index + 1) * sample_pseudo_size
                np.random.seed(pseudo_random_state)
                indices = np.random.permutation(pseudo_data_size)
                sample_indices = indices[start: end]
            for idx in sample_indices:
                line = lines[idx]
                text_id, name, desc, cate1, cate2, cate3 = line.strip().split('##')
                pseudo_label_data['id'].append(text_id)
                pseudo_label_data['name'].append(name)
                pseudo_label_data['desc'].append(desc)
                pseudo_label_data['cate1'].append(cate1)
                pseudo_label_data['cate2'].append(cate2)
                pseudo_label_data['cate3'].append(cate3)

        for key in pseudo_label_data:
            self.data[key].extend(pseudo_label_data[key])

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        batch_input_ids, batch_input_masks, batch_input_types = [], [], []
        batch_cate1_ids, batch_cate2_ids, batch_cate3_ids = [], [], []  # labels of multi task taining
        batch_all_cate_ids = []  # labels of single task taining
        for i in batch_index:
            text = self.prepare_text(self.data['name'][i], self.data['desc'][i])

            # prepare input
            if self.use_word_input:
                word_ids = [self.word_vocab.get(w, 1) for w in nltk.tokenize.word_tokenize(text)]
                batch_input_ids.append(word_ids)
            elif self.use_bert_input:
                if self.use_pair_input:
                    try:
                        inputs = self.bert_tokenizer.encode_plus(text=text[0], text_pair=text[1],
                                                                 max_length=self.max_len,
                                                                 pad_to_max_length=True,
                                                                 truncation_strategy='only_second')
                    except Exception:
                        inputs = self.bert_tokenizer.encode_plus(text=text[0], text_pair=text[1], 
                                                                 max_length=self.max_len,
                                                                 pad_to_max_length=True,
                                                                 truncation_strategy='longest_first')
                else:
                    inputs = self.bert_tokenizer.encode_plus(text=text, max_length=self.max_len, pad_to_max_length=True)
                batch_input_ids.append(inputs['input_ids'])
                batch_input_masks.append(inputs['attention_mask'])
                if 'token_type_ids' in inputs:
                    batch_input_types.append(inputs['token_type_ids'])
            else:
                raise ValueError('must use word or bert as input')

            if self.data_type == 'test':  # no labels for test data
                continue
            # prepare label for training or validation set
            if not self.use_multi_task:
                all_cate = f"{self.data['cate1'][i]}|{self.data['cate2'][i]}|{self.data['cate3'][i]}"
                if (self.data_type == 'dev' or self.train_on_cv) and all_cate not in self.all_cate_vocab:
                    batch_all_cate_ids.append(0)
                else:
                    batch_all_cate_ids.append(self.all_cate_vocab[all_cate])
            else:
                batch_cate1_ids.append(self.cate1_vocab[self.data['cate1'][i]])
                if (self.data_type == 'dev' or self.train_on_cv) and self.data['cate2'][i] not in self.cate2_vocab:
                    batch_cate2_ids.append(0)
                else:
                    batch_cate2_ids.append(self.cate2_vocab[self.data['cate2'][i]])
                if (self.data_type == 'dev' or self.train_on_cv) and self.data['cate3'][i] not in self.cate3_vocab:
                    batch_cate3_ids.append(0)
                else:
                    batch_cate3_ids.append(self.cate3_vocab[self.data['cate3'][i]])

        # feature input
        if self.use_word_input:
            batch_inputs = pad_sequences_1d(batch_input_ids, max_len=self.max_len)
        else:
            batch_inputs = [np.array(batch_input_ids), np.array(batch_input_masks)]
            if batch_input_types:
                batch_inputs.append(np.array(batch_input_types))

        if self.data_type == 'test':  # no labels for test data
            return batch_inputs

        # label masking (only for training dataset)
        if self.use_multi_task and self.data_type == 'train':
            if self.use_mask_for_cate2:
                if not isinstance(batch_inputs, list):
                    batch_inputs = [batch_inputs]
                batch_inputs.append(self.cate1_to_cate2_matrix[np.array(batch_cate1_ids)])
            if self.use_mask_for_cate3:
                if not isinstance(batch_inputs, list):
                    batch_inputs = [batch_inputs]
                if self.cate3_mask_type == 'cate1':
                    batch_inputs.append(self.cate_to_cate3_matrix[np.array(batch_cate1_ids)])
                elif self.cate3_mask_type == 'cate2':
                    batch_inputs.append(self.cate_to_cate3_matrix[np.array(batch_cate2_ids)])
                else:
                    raise ValueError(f'`cate3_mask_type` not understood')

        # ground truth labels
        if not self.use_multi_task:
            batch_labels = to_categorical(batch_all_cate_ids, num_classes=len(self.all_cate_vocab))
        else:
            batch_labels = [
                to_categorical(batch_cate1_ids, num_classes=len(self.cate1_vocab)),
                to_categorical(batch_cate2_ids, num_classes=len(self.cate2_vocab)),
                to_categorical(batch_cate3_ids, num_classes=len(self.cate3_vocab))
            ]

        return batch_inputs, batch_labels

    def prepare_text(self, name, desc):
        if self.input_type == 'name':
            return name
        elif self.input_type == 'desc':
            if not desc:
                return name
            else:
                return desc
        elif self.input_type == 'name_desc':
            if desc:
                if self.use_pair_input:
                    return name, desc
                else:
                    return f"{name} {desc}"
            else:
                if self.use_pair_input:
                    return name, name
                else:
                    return name
        else:
            raise ValueError(f'`input_type` not understood: {self.input_type}')

    @staticmethod
    def create_mask_matrix(cate1_to_cate2, n_cate1, n_cate2):
        mask_matrix = np.zeros(shape=(n_cate1, n_cate2))
        for cate1 in cate1_to_cate2:
            for cate2 in cate1_to_cate2[cate1]:
                mask_matrix[cate1][cate2] = 1
        return mask_matrix

