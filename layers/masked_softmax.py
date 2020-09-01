# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: masked_softmax.py

@time: 2020/5/30 22:18

@desc:

"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class MaskedSoftmax(Layer):
    def __init__(self, **kwargs):
        super(MaskedSoftmax, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        prob = inputs[0]
        mask = inputs[1]

        exp_prob = K.exp(prob) * mask + 1e-10
        return exp_prob / K.sum(exp_prob, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape
