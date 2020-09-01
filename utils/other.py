# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: other.py

@time: 2020/5/17 16:02

@desc:

"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def analyze_len(len_list):
    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    sort_len_list = sorted(len_list)
    print(f'max : {sort_len_list[-1]}')
    print(f'min : {sort_len_list[0]}')
    print(f'mean : {np.mean(sort_len_list)}')
    print(f'median : {np.median(sort_len_list)}')
    for i in frange(0.5, 1, 0.1):
        print(f'{i:.2f} : {sort_len_list[int(len(sort_len_list) * i)]}')


def pad_sequences_1d(sequences, max_len=None, padding='post', truncating='post', value=0.):
    """pad sequence for [[a, b, c, ...]]"""
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating, value=value)
