# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: base_classify_model.py

@time: 2020/5/21 8:47

@desc:

"""
import os
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from transformers.modeling_tf_utils import get_initializer
from transformers import TFSequenceSummary

from models.base_model import BaseModel
from losses import multi_category_focal_loss2
from layers import MaskedSoftmax, HierarchicalAttentionRecurrent, HierarchicalAttention
from utils import get_optimizer, get_transformer, pickle_dump, precision_recall_fscore, save_diff_to_file, \
    save_prob_to_file, submit_result
from callbacks import MultiTaskMetric, MaskedMultiTaskMetric


class MultiTaskClsModel(BaseModel):
    def __init__(self, config, mask_zero=False):
        super(MultiTaskClsModel, self).__init__(config)
        self.mask_zero = mask_zero
        # build model
        if self.config.use_multi_task and (self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3):
            self.model, self.cate1_model, self.cate2_model, self.cate3_model = self.build()
        else:
            self.model = self.build()
        if 'swa' in self.config.callbacks_to_add or 'swa_clr' in self.config.callbacks_to_add:
            if self.config.use_multi_task and (self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3):
                self.swa_model, _, _, _ = self.build()
            else:
                self.swa_model = self.build()

    def build(self):
        input_text = Input(shape=(self.config.max_len,))

        if self.config.word_embeddings is not None:
            assert self.config.word_vocab_size == self.config.word_embeddings.shape[0]
            assert self.config.word_embed_dim == self.config.word_embeddings.shape[1]
            embedding_layer = Embedding(input_dim=self.config.word_vocab_size,
                                        output_dim=self.config.word_embed_dim,
                                        weights=[self.config.word_embeddings],
                                        trainable=self.config.word_embed_trainable,
                                        name='word_embeddings', mask_zero=self.mask_zero)
        else:
            embedding_layer = Embedding(input_dim=self.config.word_vocab_size,
                                        output_dim=self.config.word_embed_dim,
                                        name='word_embeddings',
                                        mask_zero=self.mask_zero)
        text_embed = SpatialDropout1D(0.2)(embedding_layer(input_text))
        sentence_embed = self.encoder(text_embed, return_sequence=self.config.use_harl or self.config.use_hal)

        return self.build_model(input_text, sentence_embed)

    def encoder(self, text_embed, return_sequence):
        raise NotImplementedError

    def build_model(self, inputs, sentence_embed, kernel_initializer='glorot_uniform'):
        if self.config.use_focal_loss:
            loss_func = multi_category_focal_loss2(alpha=0.25, gamma=2)
        else:
            loss_func = 'categorical_crossentropy'
        if self.config.use_multi_task:
            if self.config.use_harl or self.config.use_hal:
                cate_hierarchy = [len(self.config.cate1_vocab),
                                  len(self.config.cate2_vocab),
                                  len(self.config.cate3_vocab)]

                attend_layer = HierarchicalAttentionRecurrent if self.config.use_harl else HierarchicalAttention
                cate1_output, cate2_output, cate3_output = attend_layer(
                    cate_hierarchy=cate_hierarchy,
                    cate_embed_dim=self.config.cate_embed_dim,
                    kernel_initializer=kernel_initializer,
                    name='cate'
                )(sentence_embed)
                output = [cate1_output, cate2_output, cate3_output]
                loss = [loss_func, loss_func, loss_func]
                loss_weights = [self.config.cate1_loss_weight,
                                self.config.cate2_loss_weight,
                                self.config.cate3_loss_weight]
                metrics = ['acc']
            else:
                cate1_output_unnormalize = Dense(self.config.n_cate1,
                                                 activation=None,
                                                 kernel_initializer=kernel_initializer,
                                                 name='cate1_unnormalize')(sentence_embed)
                cate1_output = Activation(activation='softmax', name='cate1')(cate1_output_unnormalize)

                if self.config.share_father_pred == 'before':
                    cate2_input = concatenate([sentence_embed, cate1_output_unnormalize])
                elif self.config.share_father_pred == 'after':
                    cate2_input = concatenate([sentence_embed, cate1_output])
                else:
                    cate2_input = sentence_embed
                cate2_output_unnormalize = Dense(self.config.n_cate2,
                                                 activation=None,
                                                 kernel_initializer=kernel_initializer,
                                                 name='cate2_unnormalize')(cate2_input)

                if self.config.use_mask_for_cate2:
                    input_cate2_mask = Input(shape=(self.config.n_cate2, ), name='input_cate2_mask')
                    cate2_output = MaskedSoftmax(name='cate2')([cate2_output_unnormalize, input_cate2_mask])
                else:
                    cate2_output = Activation(activation='softmax', name='cate2')(cate2_output_unnormalize)

                if self.config.share_father_pred == 'before':
                    cate3_input = concatenate([sentence_embed, cate1_output_unnormalize, cate2_output_unnormalize])
                elif self.config.share_father_pred == 'after':
                    cate3_input = concatenate([sentence_embed, cate1_output, cate2_output])
                else:
                    cate3_input = sentence_embed
                cate3_output_unnormalize = Dense(self.config.n_cate3,
                                                 activation=None,
                                                 kernel_initializer=kernel_initializer,
                                                 name='cate3_unnormalize')(cate3_input)
                if self.config.use_mask_for_cate3:
                    input_cate3_mask = Input(shape=(self.config.n_cate3, ), name='input_cate3_mask')
                    cate3_output = MaskedSoftmax(name='cate3')([cate3_output_unnormalize, input_cate3_mask])
                else:
                    cate3_output = Activation(activation='softmax', name='cate3')(cate3_output_unnormalize)

                output = [cate1_output, cate2_output, cate3_output]

                loss = {
                    'cate1': loss_func,
                    'cate2': loss_func,
                    'cate3': loss_func
                }
                loss_weights = {
                    'cate1': self.config.cate1_loss_weight,
                    'cate2': self.config.cate2_loss_weight,
                    'cate3': self.config.cate3_loss_weight
                }
                metrics = {
                    'cate1': 'acc',
                    'cate2': 'acc',
                    'cate3': 'acc'
                }

                if self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3:
                    inputs_for_model = inputs
                    inputs_for_cate1_model = inputs
                    inputs_for_cate2_model = inputs
                    inputs_for_cate3_model = inputs
                    if self.config.use_mask_for_cate2:
                        if isinstance(inputs_for_cate2_model, list):
                            inputs_for_cate2_model = inputs_for_cate2_model + [input_cate2_mask]
                        else:
                            inputs_for_cate2_model = [inputs_for_cate2_model, input_cate2_mask]
                        if isinstance(inputs_for_model, list):
                            inputs_for_model = inputs_for_model + [input_cate2_mask]
                        else:
                            inputs_for_model = [inputs_for_model, input_cate2_mask]
                    if self.config.use_mask_for_cate3:
                        if isinstance(inputs_for_cate3_model, list):
                            inputs_for_cate3_model = inputs_for_cate3_model + [input_cate3_mask]
                        else:
                            inputs_for_cate3_model = [inputs_for_cate3_model, input_cate3_mask]
                        if isinstance(inputs_for_model, list):
                            inputs_for_model = inputs_for_model + [input_cate3_mask]
                        else:
                            inputs_for_model = [inputs_for_model, input_cate3_mask]

                    model = Model(inputs_for_model, output)
                    cate1_model = Model(inputs_for_cate1_model, cate1_output)
                    cate2_model = Model(inputs_for_cate2_model, cate2_output)
                    cate3_model = Model(inputs_for_cate3_model, cate3_output)
                    model.compile(loss=loss, loss_weights=loss_weights, metrics=metrics,
                                  optimizer=get_optimizer(self.config.optimizer,
                                                          self.config.learning_rate))
                    return model, cate1_model, cate2_model, cate3_model
        else:
            output = Dense(self.config.n_all_cate, activation='softmax', kernel_initializer=kernel_initializer,
                           name='all_cate')(sentence_embed)
            loss = loss_func
            loss_weights = None
            metrics = ['acc']

        model = Model(inputs, output)
        model.compile(loss=loss, loss_weights=loss_weights, metrics=metrics,
                      optimizer=get_optimizer(self.config.optimizer,
                                              self.config.learning_rate))
        return model

    def predict_masked_multitask(self, x, cate1_to_cate2_matrix, cate_to_cate3_matrix, return_prob=False):
        pred_prob_cate1 = self.cate1_model.predict(x)
        pred_cate1 = np.argmax(pred_prob_cate1, axis=-1)

        inputs_for_cate2_model = x
        if self.config.use_mask_for_cate2:
            input_cate2_mask = cate1_to_cate2_matrix[pred_cate1]
            if isinstance(inputs_for_cate2_model, list):
                inputs_for_cate2_model = inputs_for_cate2_model + [input_cate2_mask]
            else:
                inputs_for_cate2_model = [inputs_for_cate2_model, input_cate2_mask]
        pred_prob_cate2 = self.cate2_model.predict(inputs_for_cate2_model)
        pred_cate2 = np.argmax(pred_prob_cate2, axis=-1)

        inputs_for_cate3_model = x
        if self.config.use_mask_for_cate3:
            if self.config.cate3_mask_type == 'cate1':
                input_cate3_mask = cate_to_cate3_matrix[pred_cate1]
            elif self.config.cate3_mask_type == 'cate2':
                input_cate3_mask = cate_to_cate3_matrix[pred_cate2]
            else:
                raise ValueError(f'`cate3_mask_type` not understood: {self.config.cate3_mask_type}')
            if isinstance(inputs_for_cate3_model, list):
                inputs_for_cate3_model = inputs_for_cate3_model + [input_cate3_mask]
            else:
                inputs_for_cate3_model = [inputs_for_cate3_model, input_cate3_mask]
        pred_prob_cate3 = self.cate3_model.predict(inputs_for_cate3_model)
        pred_cate3 = np.argmax(pred_prob_cate3, axis=-1)

        pred_cate1 = pred_cate1.tolist()
        pred_cate2 = pred_cate2.tolist()
        pred_cate3 = pred_cate3.tolist()

        if return_prob:
            return pred_cate1, pred_cate2, pred_cate3, pred_prob_cate1, pred_prob_cate2, pred_prob_cate3
        else:
            return pred_cate1, pred_cate2, pred_cate3

    def predict_multitask(self, x, return_prob=False):
        pred_prob_cate1, pred_prob_cate2, pred_prob_cate3 = self.model.predict(x)
        pred_cate1 = np.argmax(pred_prob_cate1, axis=-1).tolist()
        pred_cate2 = np.argmax(pred_prob_cate2, axis=-1).tolist()
        pred_cate3 = np.argmax(pred_prob_cate3, axis=-1).tolist()
        if return_prob:
            return pred_cate1, pred_cate2, pred_cate3, pred_prob_cate1, pred_prob_cate2, pred_prob_cate3
        else:
            return pred_cate1, pred_cate2, pred_cate3

    def predict_one_task(self, x, return_prob=False):
        pred_prob_all_cate = self.model.predict(x)
        pred_all_cate = np.argmax(pred_prob_all_cate, axis=-1).tolist()
        pred_all_cate = map(self.config.idx2all_cate.get, pred_all_cate)
        pred_all_cate_split = map(lambda c: c.split('|'), pred_all_cate)
        pred_cate1, pred_cate2, pred_cate3 = zip(*list(pred_all_cate_split))

        pred_cate1 = list(map(self.config.cate1_vocab.get, pred_cate1))
        pred_cate2 = list(map(self.config.cate2_vocab.get, pred_cate2))
        pred_cate3 = list(map(self.config.cate3_vocab.get, pred_cate3))
        if return_prob:
            return pred_cate1, pred_cate2, pred_cate3, pred_prob_all_cate
        else:
            return pred_cate1, pred_cate2, pred_cate3

    def predict(self, test_generator, save_prob=False, prob_file=None, submit=False, submit_file=None,
                submit_with_text=False):
        pred_prob_cate1_list, pred_prob_cate2_list, pred_prob_cate3_list = [], [], []
        pred_prob_all_cate_list = []
        pred_cate1_list, pred_cate2_list, pred_cate3_list = [], [], []
        for x in test_generator:
            if self.config.use_multi_task:
                if self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3:
                    pred_results = self.predict_masked_multitask(x,
                                                                 test_generator.cate1_to_cate2_matrix,
                                                                 test_generator.cate_to_cate3_matrix,
                                                                 return_prob=save_prob)
                else:
                    pred_results = self.predict_multitask(x, return_prob=save_prob)
            else:
                pred_results = self.predict_one_task(x, return_prob=save_prob)
            pred_cate1_list.extend(pred_results[0])
            pred_cate2_list.extend(pred_results[1])
            pred_cate3_list.extend(pred_results[2])
            if len(pred_results) == 6:
                pred_prob_cate1_list.append(pred_results[3])
                pred_prob_cate2_list.append(pred_results[4])
                pred_prob_cate3_list.append(pred_results[5])
            elif len(pred_results) == 4:
                pred_prob_all_cate_list.append(pred_results[4])
            else:
                raise ValueError(f'`pred_results` has wrong length: {len(pred_results)}')

        if save_prob:
            save_prob_to_file(pred_prob_cate1_list=pred_prob_cate1_list,
                              pred_prob_cate2_list=pred_prob_cate2_list,
                              pred_prob_cate3_list=pred_prob_cate3_list,
                              pred_prob_all_cate_list=pred_prob_all_cate_list,
                              prob_file=prob_file,
                              use_multi_task=self.config.use_multi_task)

        if submit:
            submit_result(test_data=test_generator.data,
                          pred_cate1_list=pred_cate1_list,
                          pred_cate2_list=pred_cate2_list,
                          pred_cate3_list=pred_cate3_list,
                          idx2cate1=self.config.idx2cate1,
                          idx2cate2=self.config.idx2cate2,
                          idx2cate3=self.config.idx2cate3,
                          submit_file=submit_file,
                          submit_with_text=submit_with_text)

        return pred_cate1_list, pred_cate2_list, pred_cate3_list

    def evaluate(self, valid_generator, save_prob=False, prob_file=None, save_diff=False, diff_file=None):
        pred_prob_cate1_list, pred_prob_cate2_list, pred_prob_cate3_list = [], [], []
        pred_prob_all_cate_list = []
        pred_cate1_list, pred_cate2_list, pred_cate3_list = [], [], []
        true_cate1_list, true_cate2_list, true_cate3_list = [], [], []

        for x_valid, y_valid in valid_generator:
            if self.config.use_multi_task:
                if self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3:
                    pred_results = self.predict_masked_multitask(x_valid,
                                                                 valid_generator.cate1_to_cate2_matrix,
                                                                 valid_generator.cate_to_cate3_matrix,
                                                                 return_prob=save_prob)
                else:
                    pred_results = self.predict_multitask(x_valid, return_prob=save_prob)
            else:
                pred_results = self.predict_one_task(x_valid, return_prob=save_prob)
            pred_cate1_list.extend(pred_results[0])
            pred_cate2_list.extend(pred_results[1])
            pred_cate3_list.extend(pred_results[2])
            if len(pred_results) == 6:
                pred_prob_cate1_list.append(pred_results[3])
                pred_prob_cate2_list.append(pred_results[4])
                pred_prob_cate3_list.append(pred_results[5])
            elif len(pred_results) == 4:
                pred_prob_all_cate_list.append(pred_results[4])
            else:
                raise ValueError(f'`pred_results` has wrong length: {len(pred_results)}')

            true_cate1_list.extend(np.argmax(y_valid[0], axis=-1).tolist())
            true_cate2_list.extend(np.argmax(y_valid[1], axis=-1).tolist())
            true_cate3_list.extend(np.argmax(y_valid[2], axis=-1).tolist())

        if save_prob:
            save_prob_to_file(pred_prob_cate1_list=pred_prob_cate1_list,
                              pred_prob_cate2_list=pred_prob_cate2_list,
                              pred_prob_cate3_list=pred_prob_cate3_list,
                              pred_prob_all_cate_list=pred_prob_all_cate_list,
                              prob_file=prob_file,
                              use_multi_task=self.config.use_multi_task)

        if save_diff:
            save_diff_to_file(data=valid_generator.data,
                              pred_cate1_list=pred_cate1_list,
                              pred_cate2_list=pred_cate2_list,
                              pred_cate3_list=pred_cate3_list,
                              true_cate1_list=true_cate1_list,
                              true_cate2_list=true_cate2_list,
                              true_cate3_list=true_cate3_list,
                              cate1_to_cate2=self.config.cate1_to_cate2,
                              cate2_to_cate3=self.config.cate2_to_cate3,
                              cate1_to_cate3=self.config.cate1_to_cate3,
                              idx2cate1=self.config.idx2cate1,
                              idx2cate2=self.config.idx2cate2,
                              idx2cate3=self.config.idx2cate3,
                              cate1_count_dict=self.config.cate1_count_dict,
                              cate2_count_dict=self.config.cate2_count_dict,
                              cate3_count_dict=self.config.cate3_count_dict,
                              diff_file=diff_file)

        eval_results = precision_recall_fscore(pred_cate1_list=pred_cate1_list,
                                               true_cate1_list=true_cate1_list,
                                               pred_cate2_list=pred_cate2_list,
                                               true_cate2_list=true_cate2_list,
                                               pred_cate3_list=pred_cate3_list,
                                               true_cate3_list=true_cate3_list)
        return eval_results

    def add_metric_callback(self, valid_generator):
        if self.config.use_multi_task and (self.config.use_mask_for_cate2 or self.config.use_mask_for_cate3):
            self.callbacks.append(MaskedMultiTaskMetric(valid_generator=valid_generator,
                                                        cate1_model=self.cate1_model,
                                                        cate2_model=self.cate2_model,
                                                        cate3_model=self.cate3_model,
                                                        config=self.config))
            print('Logging Info - Callback Added: MaskedMultiTaskMetric...')
        else:
            self.callbacks.append(MultiTaskMetric(valid_generator=valid_generator, config=self.config))
            print('Logging Info - save_diff_to_fileCallback Added: MultiTaskMetric...')


class BiLSTM(MultiTaskClsModel):
    def __init__(self, config):
        super(BiLSTM, self).__init__(config, mask_zero=True)

    def encoder(self, text_embed, return_sequence):
        hidden_states = Bidirectional(LSTM(units=300, return_sequences=True))(text_embed)
        if return_sequence:
            return hidden_states
        global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
        sentence_embed = global_max_pooling(hidden_states)

        sentence_embed = Dense(256, activation='relu')(sentence_embed)
        return sentence_embed


class CNNRNN(MultiTaskClsModel):
    def __init__(self, config):
        super(CNNRNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        conv_layer = Conv1D(300, kernel_size=3, padding="valid", activation='relu')(text_embed)
        conv_max_pool = MaxPooling1D(pool_size=2)(conv_layer)

        gru_layer = Bidirectional(GRU(300, return_sequences=True))(conv_max_pool)
        if return_sequence:
            return gru_layer

        sentence_embed = GlobalMaxPooling1D()(gru_layer)
        return sentence_embed


class DPCNN(MultiTaskClsModel):
    def __init__(self, config):
        super(DPCNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        repeat = 3
        region_x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(text_embed)
        x = Activation(activation='relu')(region_x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Add()([x, region_x])

        for _ in range(repeat):
            px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
            x = Activation(activation='relu')(px)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Activation(activation='relu')(x)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Add()([x, px])
        if return_sequence:
            return x
        sentence_embed = GlobalMaxPooling1D()(x)
        return sentence_embed


class MultiTextCNN(MultiTaskClsModel):
    def __init__(self, config):
        super(MultiTextCNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        filter_lengths = [2, 3, 4, 5]
        conv_sequences = []
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer_1 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(text_embed)
            bn_layer_1 = BatchNormalization()(conv_layer_1)
            conv_layer_2 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(bn_layer_1)
            bn_layer_2 = BatchNormalization()(conv_layer_2)
            conv_sequences.append(bn_layer_2)
            flatten = GlobalMaxPooling1D()(bn_layer_2)
            conv_layers.append(flatten)

        if return_sequence:
            return concatenate(conv_sequences)
        sentence_embed = concatenate(inputs=conv_layers)
        return sentence_embed


class RCNN(MultiTaskClsModel):
    def __init__(self, config):
        super(RCNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        # We shift the document to the right to obtain the left-side contexts
        l_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, :-1]], axis=1))(text_embed)
        # We shift the document to the left to obtain the right-side contexts
        r_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, 1:]], axis=1))(text_embed)
        # use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(300, return_sequences=True)(l_embedding)  # See equation (1)
        backward = LSTM(300, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2)
        # Keras returns the output sequences in reverse order.
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
        together = concatenate([forward, text_embed, backward], axis=2)  # See equation (3).

        # use conv1D instead of TimeDistributed and Dense
        semantic = Conv1D(300, kernel_size=1, activation="tanh")(together)  # See equation (4).
        if return_sequence:
            return semantic
        sentence_embed = Lambda(lambda x: K.max(x, axis=1))(semantic)  # See equation (5).

        return sentence_embed


class RNNCNN(MultiTaskClsModel):
    def __init__(self, config):
        super(RNNCNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        gru_layer = Bidirectional(GRU(300, return_sequences=True))(text_embed)

        conv_layer = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(gru_layer)

        if return_sequence:
            return conv_layer

        avg_pool = GlobalAveragePooling1D()(conv_layer)
        max_pool = GlobalMaxPooling1D()(conv_layer)
        sentence_embed = concatenate([avg_pool, max_pool])
        return sentence_embed


class TextCNN(MultiTaskClsModel):
    def __init__(self, config):
        super(TextCNN, self).__init__(config, mask_zero=False)

    def encoder(self, text_embed, return_sequence):
        filter_lengths = [2, 3, 4, 5]
        conv_sequences = []
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                strides=1, activation='relu')(text_embed)
            conv_sequences.append(conv_layer)
            conv_layers.append(GlobalMaxPooling1D()(conv_layer))
        if return_sequence:
            return concatenate(conv_sequences)
        sentence_embed = concatenate(inputs=conv_layers)
        return sentence_embed


class TransformerClsModel(MultiTaskClsModel):
    def __init__(self, config):
        self.transformer = get_transformer(bert_model_type=config.model_type,
                                           output_hidden_states=config.output_hidden_state)
        if not config.bert_trainable:
            self.transformer.trainable = False
        self.kernel_initializer = get_initializer(self.transformer.config.initializer_range)
        super(TransformerClsModel, self).__init__(config)

    def build(self):
        inputs = self.build_input()
        if self.config.use_harl or self.config.use_hal:
            sequence_summary = self.get_hidden_sequence(inputs)
        else:
            sequence_summary = self.get_sequence_summary(inputs)
        return self.build_model(inputs, sequence_summary, self.kernel_initializer)

    def build_input(self):
        input_text = Input(shape=(self.config.max_len,), name='input_text', dtype='int32')
        input_mask = Input(shape=(self.config.max_len,), name='input_mask', dtype='int32')
        input_type = Input(shape=(self.config.max_len,), name='input_type', dtype='int32')
        inputs = [input_text, input_mask, input_type]
        return inputs

    def get_hidden_sequence(self, inputs):
        transformer_output = self.transformer(inputs)
        if self.config.n_last_hidden_layer == 1:
            bert_hidden_states = transformer_output[0]
        else:
            bert_hidden_states = concatenate(list(transformer_output[2][-self.config.n_last_hidden_layer:]))
        if self.config.use_bert_type == 'hidden':
            seq_output = bert_hidden_states
        elif self.config.use_bert_type == 'lstm':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                       return_sequences=True))(bert_hidden_states)
        elif self.config.use_bert_type == 'gru':
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                       return_sequences=True))(bert_hidden_states)
        elif self.config.use_bert_type == 'lstm_gru':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(bert_hidden_states)
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                       return_sequences=True))(seq_output)
        elif self.config.use_bert_type == 'gru_lstm':
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                           return_sequences=True))(bert_hidden_states)
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                       return_sequences=True))(seq_output)
        elif self.config.use_bert_type == 'cnn':
            seq_output = Conv1D(filters=300, kernel_size=3, padding='same',
                                strides=1, activation='relu')(bert_hidden_states)
        elif self.config.use_bert_type == 'lstm_cnn':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(bert_hidden_states)
            seq_output = Conv1D(filters=300, kernel_size=3, padding='same',
                                strides=1, activation='relu')(seq_output)
        else:
            raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')

        if self.config.dense_after_bert:
            seq_output = Dense(
                self.transformer.config.hidden_size,
                kernel_initializer=self.kernel_initializer,
                activation="tanh",
                name=f"{self.config.use_bert_type}_dense",
            )(seq_output)
        return seq_output

    def get_sequence_summary(self, inputs):
        if self.config.use_bert_type == 'pooler':
            sequence_summary = self.transformer(inputs)[1]
        elif self.config.use_bert_type in ['hidden', 'hidden_pooler']:
            transformer_output = self.transformer(inputs)
            needed_hidden_states = []
            for i in range(-1, -1-self.config.n_last_hidden_layer, -1):
                # taking the hidden state corresponding to the first token
                needed_hidden_states.append(transformer_output[2][i][:, 0])  # [batch_size, 768]
            if self.config.use_bert_type == 'hidden_pooler':
                # add pooler output
                needed_hidden_states.append(transformer_output[1])
            if len(needed_hidden_states) == 1:
                sequence_summary = needed_hidden_states[0]
            else:
                sequence_summary = concatenate(needed_hidden_states)
        elif self.config.use_bert_type in ['lstm', 'gru', 'lstm_gru', 'gru_lstm', 'cnn', 'lstm_cnn']:
            transformer_output = self.transformer(inputs)
            needed_hidden_states = [transformer_output[1]]  # pooler output
            if self.config.use_bert_type == 'lstm':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
            elif self.config.use_bert_type == 'gru':
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(transformer_output[0])
            elif self.config.use_bert_type == 'lstm_gru':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(seq_output)
            elif self.config.use_bert_type == 'gru_lstm':
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(transformer_output[0])
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(seq_output)
            elif self.config.use_bert_type == 'cnn':
                seq_output = Conv1D(filters=300, kernel_size=3, padding='valid',
                                    strides=1, activation='relu')(transformer_output[0])
            elif self.config.use_bert_type == 'lstm_cnn':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
                seq_output = Conv1D(filters=300, kernel_size=3, padding='valid',
                                    strides=1, activation='relu')(seq_output)
            else:
                raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')
            needed_hidden_states.append(GlobalMaxPooling1D()(seq_output))
            needed_hidden_states.append(GlobalAveragePooling1D()(seq_output))
            if 'cnn' not in self.config.use_bert_type:
                needed_hidden_states.append(seq_output[:, -1])  # last hidden state
            sequence_summary = concatenate(needed_hidden_states)
        else:
            raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')

        if self.config.dense_after_bert:
            sequence_summary = Dense(
                self.transformer.config.hidden_size,
                kernel_initializer=self.kernel_initializer,
                activation="tanh",
                name=f"{self.config.use_bert_type}_dense",
            )(sequence_summary)
        return Dropout(self.transformer.config.hidden_dropout_prob)(sequence_summary)

    # unused function
    def encoder(self, text_embed, return_sequence):
        pass


# from transformers import TFBertForSequenceClassification, TFBertModel
# from transformers import TFAlbertForSequenceClassification, TFAlbertModel
class BertClsModel(TransformerClsModel):
    def __init__(self, config):
        super(BertClsModel, self).__init__(config)


# from transformers import TFRobertaForSequenceClassification, TFRobertaModel, TFRobertaMainLayer
class RobertaClsModel(TransformerClsModel):
    def __init__(self, config):
        super(RobertaClsModel, self).__init__(config)

    def build_input(self):
        input_text = Input(shape=(self.config.max_len,), name='input_text', dtype='int32')
        input_mask = Input(shape=(self.config.max_len,), name='input_mask', dtype='int32')
        inputs = [input_text, input_mask]
        return inputs


class TransformerWoHeadClsModel(TransformerClsModel):
    def __init__(self, config, hidden_states_index=1, ):
        self.hidden_states_index = hidden_states_index
        super(TransformerWoHeadClsModel, self).__init__(config)

    def get_hidden_sequence(self, inputs):
        inputs_dict = {
            'input_ids': inputs[0],
            'attention_mask': inputs[1],
        }
        if len(inputs) == 3:
            inputs_dict['token_type_ids'] = inputs[2]
        transformer_output = self.transformer(inputs_dict, **kwargs)

        if self.config.n_last_hidden_layer == 1:
            bert_hidden_states = transformer_output[0]
        else:
            bert_hidden_states = concatenate(
                list(transformer_output[self.hidden_states_index][-self.config.n_last_hidden_layer:]))
        if self.config.use_bert_type == 'hidden':
            seq_output = bert_hidden_states
        elif self.config.use_bert_type == 'lstm':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(bert_hidden_states)
        elif self.config.use_bert_type == 'gru':
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                           return_sequences=True))(bert_hidden_states)
        elif self.config.use_bert_type == 'lstm_gru':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(bert_hidden_states)
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                           return_sequences=True))(seq_output)
        elif self.config.use_bert_type == 'gru_lstm':
            seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                           return_sequences=True))(bert_hidden_states)
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(seq_output)
        elif self.config.use_bert_type == 'cnn':
            seq_output = Conv1D(filters=300, kernel_size=3, padding='same',
                                strides=1, activation='relu')(bert_hidden_states)
        elif self.config.use_bert_type == 'lstm_cnn':
            seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                            return_sequences=True))(bert_hidden_states)
            seq_output = Conv1D(filters=300, kernel_size=3, padding='same',
                                strides=1, activation='relu')(seq_output)
        else:
            raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')

        if self.config.dense_after_bert:
            seq_output = Dense(
                self.transformer.config.hidden_size,
                kernel_initializer=self.kernel_initializer,
                activation="tanh",
                name=f"{self.config.use_bert_type}_dense",
            )(seq_output)
        return seq_output

    def get_pooler_output(self, last_hidden_states):
        raise NotImplementedError

    def get_sequence_summary(self, inputs):
        inputs_dict = {
            'input_ids': inputs[0],
            'attention_mask': inputs[1],
        }
        if len(inputs) == 3:
            inputs_dict['token_type_ids'] = inputs[2]

        if self.config.use_bert_type == 'pooler':
            transformer_output = self.transformer(inputs_dict)
            sequence_summary = self.get_pooler_output(transformer_output[0])
        elif self.config.use_bert_type in ['hidden', 'hidden_pooler']:
            transformer_output = self.transformer(inputs_dict)
            needed_hidden_states = []
            for i in range(-1, -1 - self.config.n_last_hidden_layer, -1):
                # taking the hidden state corresponding to the first token
                needed_hidden_states.append(transformer_output[self.hidden_states_index][i][:, 0])  # [batch_size, 768]

            if self.config.use_bert_type == 'hidden_pooler':
                # add pooler output
                pooler_output = self.get_pooler_output(transformer_output[0])
                needed_hidden_states.append(pooler_output)
            if len(needed_hidden_states) == 1:
                sequence_summary = needed_hidden_states[0]
            else:
                sequence_summary = concatenate(needed_hidden_states)
        elif self.config.use_bert_type in ['lstm', 'gru', 'cnn', 'lstm_gru', 'gru_lstm']:
            transformer_output = self.transformer(inputs_dict, **kwargs)
            needed_hidden_states = []  # pooler output
            # add pooler output
            pooler_output = self.get_pooler_output(transformer_output[0])
            needed_hidden_states.append(pooler_output)
            if self.config.use_bert_type == 'lstm':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
            elif self.config.use_bert_type == 'gru':
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(transformer_output[0])
            elif self.config.use_bert_type == 'lstm_gru':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(seq_output)
            elif self.config.use_bert_type == 'gru_lstm':
                seq_output = Bidirectional(GRU(self.transformer.config.hidden_size,
                                               return_sequences=True))(transformer_output[0])
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(seq_output)
            elif self.config.use_bert_type == 'cnn':
                seq_output = Conv1D(filters=300, kernel_size=3, padding='valid',
                                    strides=1, activation='relu')(transformer_output[0])
            elif self.config.use_bert_type == 'lstm_cnn':
                seq_output = Bidirectional(LSTM(self.transformer.config.hidden_size,
                                                return_sequences=True))(transformer_output[0])
                seq_output = Conv1D(filters=300, kernel_size=3, padding='valid',
                                    strides=1, activation='relu')(seq_output)
            else:
                raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')
            needed_hidden_states.append(GlobalMaxPooling1D()(seq_output))
            needed_hidden_states.append(GlobalAveragePooling1D()(seq_output))
            if 'cnn' not in self.config.use_bert_type:
                needed_hidden_states.append(seq_output[:, -1])  # last hidden state
            sequence_summary = concatenate(needed_hidden_states)
        else:
            raise ValueError(f'`use_bert_type` not understood: {self.config.use_bert_type}')

        if self.config.dense_after_bert:
            sequence_summary = Dense(
                self.transformer.config.hidden_size,
                kernel_initializer=self.kernel_initializer,
                activation="tanh",
                name=f"{self.config.use_bert_type}_dense",
            )(sequence_summary)

        try:
            return Dropout(self.transformer.config.dropout)(sequence_summary)
        except AttributeError:
            return sequence_summary


# from transformers import TFXLNetForSequenceClassification, TFXLNetModel
class XLNetClsModel(TransformerWoHeadClsModel):
    def __init__(self, config):
        super(XLNetClsModel, self).__init__(config, hidden_states_index=1)

    def get_pooler_output(self, last_hidden_states):
        return TFSequenceSummary(
            self.transformer.config,
            initializer_range=self.transformer.config.initializer_range,
            name="pooler_output"
        )(last_hidden_states)


# from transformers import TFGPT2Model
class GPT2ClsModel(TransformerWoHeadClsModel):
    def __init__(self, config):
        super(GPT2ClsModel, self).__init__(config, hidden_states_index=2)

    def get_pooler_output(self, last_hidden_states):
        return TFSequenceSummary(
            self.transformer.config,
            initializer_range=self.transformer.config.initializer_range,
            name="pooler_output"
        )(last_hidden_states)


class TransfoXLClsModel(TransformerWoHeadClsModel):
    def __init__(self, config):
        super(TransfoXLClsModel, self).__init__(config, hidden_states_index=2)

    def get_pooler_output(self, last_hidden_states):
        return TFSequenceSummary(
            self.transformer.config,
            initializer_range=self.transformer.config.initializer_range,
            name="pooler_output"
        )(last_hidden_states)


# from transformers import TFDistilBertModel, TFDistilBertForSequenceClassification
class DistllBertClsModel(TransformerWoHeadClsModel):
    def __init__(self, config):
        super(DistllBertClsModel, self).__init__(config, hidden_states_index=1)

    def build_input(self):
        input_text = Input(shape=(self.config.max_len,), name='input_text', dtype='int32')
        input_mask = Input(shape=(self.config.max_len,), name='input_mask', dtype='int32')
        inputs = [input_text, input_mask]
        return inputs

    def get_pooler_output(self, last_hidden_states):
        pre_classifier = Dense(
            self.transformer.config.dim,
            kernel_initializer=get_initializer(self.transformer.config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        pooled_output = last_hidden_states[:, 0]
        pooled_output = pre_classifier(pooled_output)
        return pooled_output
