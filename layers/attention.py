# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: attention.py

@time: 2020/6/5 21:57

@desc:

"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.initializers import zeros
from tensorflow.keras.layers import Activation, Layer


class HierarchicalAttentionRecurrent(Layer):
    """Hierarchical Attention-based Recurrent Layer"""
    def __init__(self, cate_hierarchy, cate_embed_dim=100, kernel_initializer=None, **kwargs):
        super(HierarchicalAttentionRecurrent, self).__init__(**kwargs)
        self.cate_hierarchy = cate_hierarchy
        self.n_hierarchy = len(self.cate_hierarchy)
        self.cate_embed_dim = cate_embed_dim
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.category_embeddings = []
        self.attend_weights = []
        self.cls_dense_weights = []
        self.cls_dense_biases = []
        self.cls_pred_weights = []
        self.cls_pred_biases = []
        for h in range(self.n_hierarchy):
            # category embedding
            self.category_embeddings.append(self.add_weight(name=f'category_embed_{h}',
                                                            shape=(self.cate_hierarchy[h], self.cate_embed_dim),
                                                            initializer=self.kernel_initializer))
            # attention weights to perform attention
            self.attend_weights.append(self.add_weight(name=f'attend_weight_{h}',
                                                       shape=(input_shape[2], self.cate_embed_dim),
                                                       initializer=self.kernel_initializer))
            # class prediction weights and biases
            self.cls_dense_weights.append(self.add_weight(name=f'cls_dense_weight_{h}',
                                                          shape=(input_shape[2] * 2, input_shape[2]),
                                                          initializer=self.kernel_initializer))
            self.cls_dense_biases.append(self.add_weight(name=f'cls_dense_bias_{h}',
                                                         shape=(input_shape[2]),
                                                         initializer=zeros))
            self.cls_pred_weights.append(self.add_weight(name=f'cls_pred_weight_{h}',
                                                         shape=(input_shape[2], self.cate_hierarchy[h]),
                                                         initializer=self.kernel_initializer))
            self.cls_pred_biases.append(self.add_weight(name=f'cls_pred_bias_{h}',
                                                        shape=(self.cate_hierarchy[h]),
                                                        initializer=zeros))

        super(HierarchicalAttentionRecurrent, self).build(input_shape)

    def call(self, inputs, mask=None):
        text_seq_embed = inputs  # hidden states of text sequence
        batch_size = K.shape(text_seq_embed)[0]
        max_len = K.shape(text_seq_embed)[1]
        prev_cate_info = None  # information of previous category level

        prob_outputs = []
        for h in range(self.n_hierarchy):
            '''1. Text-Category Attention TCA'''
            if h == 0:
                text_with_prev_cate_embed = text_seq_embed
            else:
                text_with_prev_cate_embed = tf.multiply(text_seq_embed, K.expand_dims(prev_cate_info,axis=2))
            attend_matrix = K.dot(K.tanh(K.dot(text_with_prev_cate_embed, self.attend_weights[h])),
                                  K.transpose(self.category_embeddings[h]))
            attend_matrix = K.softmax(attend_matrix, axis=1)  # attention score between text and categories at h level

            # associated text-category representation
            text_cate_attend_embed = K.mean(K.batch_dot(attend_matrix, text_with_prev_cate_embed, axes=1), axis=1)

            '''2. Class Prediction Module CPM'''
            cls_dense_embed = K.dot(K.concatenate([K.mean(text_seq_embed, axis=1), text_cate_attend_embed]),
                                    self.cls_dense_weights[h]) + self.cls_dense_biases[h]
            cls_dense_embed = K.relu(cls_dense_embed)

            cls_pred_embed = K.dot(cls_dense_embed, self.cls_pred_weights[h]) + self.cls_pred_biases[h]
            cls_pred_embed = K.softmax(cls_pred_embed, axis=1)
            prob_outputs.append(cls_pred_embed)

            '''3. Class Dependency Module CPM'''
            prev_cate_info = K.mean(attend_matrix * K.expand_dims(cls_pred_embed, axis=1), axis=2)

        return prob_outputs

    def compute_output_shape(self, input_shape):
        output_shape = []
        for h in range(self.n_hierarchy):
            output_shape.append((input_shape[0], self.cate_hierarchy[h]))
        return output_shape


class HierarchicalAttention(Layer):
    """Hierarchical Attention-based Layer"""
    def __init__(self, cate_hierarchy, cate_embed_dim=100, kernel_initializer=None, **kwargs):
        super(HierarchicalAttention, self).__init__(**kwargs)
        self.cate_hierarchy = cate_hierarchy
        self.n_hierarchy = len(self.cate_hierarchy)
        self.cate_embed_dim = cate_embed_dim
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.category_embeddings = []
        self.attend_weights_1 = []
        self.attend_weights_2 = []
        self.cls_dense_weights = []
        self.cls_dense_biases = []
        self.cls_pred_weights = []
        self.cls_pred_biases = []

        self.category_embeddings.append(self.add_weight(name=f'category_embed_0',
                                                        shape=(1, self.cate_embed_dim),
                                                        initializer=self.kernel_initializer))
        for h in range(self.n_hierarchy):
            # category embedding
            self.category_embeddings.append(self.add_weight(name=f'category_embed_{h+1}',
                                                            shape=(self.cate_hierarchy[h], self.cate_embed_dim),
                                                            initializer=self.kernel_initializer))
            # attention weights to perform attention
            self.attend_weights_1.append(self.add_weight(name=f'attend_weight_1_{h}',
                                                         shape=(input_shape[2] + self.cate_embed_dim,
                                                                input_shape[2] + self.cate_embed_dim),
                                                         initializer=self.kernel_initializer))
            self.attend_weights_2.append(self.add_weight(name=f'attend_weight_2_{h}',
                                                         shape=(input_shape[2] + self.cate_embed_dim, 1),
                                                         initializer=self.kernel_initializer))
            # class prediction weights and biases
            self.cls_dense_weights.append(self.add_weight(name=f'cls_dense_weight_{h}',
                                                          shape=(input_shape[2] + self.cate_embed_dim,
                                                                 input_shape[2] + self.cate_embed_dim),
                                                          initializer=self.kernel_initializer))
            self.cls_dense_biases.append(self.add_weight(name=f'cls_dense_bias_{h}',
                                                         shape=(input_shape[2] + self.cate_embed_dim),
                                                         initializer=zeros))
            self.cls_pred_weights.append(self.add_weight(name=f'cls_pred_weight_{h}',
                                                         shape=(input_shape[2] + self.cate_embed_dim,
                                                                self.cate_hierarchy[h]),
                                                         initializer=self.kernel_initializer))
            self.cls_pred_biases.append(self.add_weight(name=f'cls_pred_bias_{h}',
                                                        shape=(self.cate_hierarchy[h]),
                                                        initializer=zeros))

        super(HierarchicalAttention, self).build(input_shape)

    def call(self, inputs, mask=None):

        text_seq_embed = inputs  # hidden states of text sequence
        batch_size = K.shape(text_seq_embed)[0]
        max_len = K.shape(text_seq_embed)[1]
        prev_cate_embed = self.category_embeddings[0]
        prev_cate_embed = K.tile(K.expand_dims(prev_cate_embed, axis=0), [batch_size, 1, 1])

        prob_outputs = []
        for h in range(self.n_hierarchy):
            text_with_prev_cate_embed = K.concatenate([text_seq_embed, K.tile(prev_cate_embed, [1, max_len, 1])])

            attend_matrix = K.dot(K.tanh(K.dot(text_with_prev_cate_embed, self.attend_weights_1[h])),
                                  self.attend_weights_2[h])
            attend_matrix = K.softmax(attend_matrix, axis=1)

            text_cate_attend_embed = K.sum(attend_matrix * text_with_prev_cate_embed, axis=1)

            cls_dense_embed = K.dot(text_cate_attend_embed, self.cls_dense_weights[h]) + self.cls_dense_biases[h]
            cls_dense_embed = K.relu(cls_dense_embed)

            cls_pred_embed = K.dot(cls_dense_embed, self.cls_pred_weights[h]) + self.cls_pred_biases[h]
            prob_outputs.append(cls_pred_embed)

            prev_cate_embed = K.expand_dims(K.dot(cls_pred_embed, self.category_embeddings[h+1]), axis=1)

        return prob_outputs

    def compute_output_shape(self, input_shape):
        output_shape = []
        for h in range(self.n_hierarchy):
            output_shape.append((input_shape[0], self.cate_hierarchy[h]))
        return output_shape
