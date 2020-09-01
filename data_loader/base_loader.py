# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: base_loader.py

@time: 2020/5/21 21:01

@desc:

"""

from config import PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, TEST_DATA_FILENAME, \
    TRAIN_CV_DATA_TEMPLATE, DEV_CV_DATA_TEMPLATE
from utils import pickle_load, format_filename


def load_data(data_type, train_on_cv=False, cv_random_state=42, cv_fold=5, cv_index=0):
    if data_type == 'train':
        if train_on_cv:
            data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_CV_DATA_TEMPLATE,
                                               random=cv_random_state, fold=cv_fold, index=cv_index))
        else:
            data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME))
    elif data_type == 'dev':
        if train_on_cv:
            data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_CV_DATA_TEMPLATE,
                                               random=cv_random_state, fold=cv_fold, index=cv_index))
        else:
            data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME))
    elif data_type == 'test':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME))
    else:
        raise ValueError('data type not understood: {}'.format(data_type))
    return data
