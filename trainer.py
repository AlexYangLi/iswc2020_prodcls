# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2020/5/21 22:06

@desc:

"""
import os
import time
import gc

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from config import ModelConfig, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, VOCABULARY_TEMPLATE, \
    PERFORMANCE_MD, \
    MAX_LEN, IDX2TOKEN_TEMPLATE, CATE1_TO_CATE2_DICT, CATE2_TO_CATE3_DICT, CATE1_TO_CATE3_DICT, \
    CATE1_COUNT_DICT, \
    CATE2_COUNT_DICT, CATE3_COUNT_DICT
from utils import format_filename, pickle_load, writer_md
from models import MultiTaskClsModel
from data_loader import MultiTaskClsDataGenerator


def prepare_config(model_type='bert-base-uncased',
                   input_type='name_desc',
                   use_multi_task=True,
                   use_harl=False,
                   use_hal=False,
                   cate_embed_dim=100,
                   use_word_input=False,
                   word_embed_type='w2v',
                   word_embed_trainable=True,
                   word_embed_dim=300,
                   use_bert_input=True,
                   bert_trainable=True,
                   use_bert_type='pooler',
                   n_last_hidden_layer=0,
                   dense_after_bert=True,
                   use_pair_input=True,
                   max_len=None,
                   share_father_pred='no',
                   use_mask_for_cate2=False,
                   use_mask_for_cate3=True,
                   cate3_mask_type='cate1',
                   cate1_loss_weight=1.,
                   cate2_loss_weight=1.,
                   cate3_loss_weight=1.,
                   batch_size=32,
                   predict_batch_size=32,
                   n_epoch=50,
                   learning_rate=2e-5,
                   optimizer='adam',
                   use_focal_loss=False,
                   callbacks_to_add=None,
                   swa_start=15,
                   early_stopping_patience=5,
                   max_lr=6e-5,
                   min_lr=1e-5,
                   train_on_cv=False,
                   cv_random_state=42,
                   cv_fold=5,
                   cv_index=0,
                   exchange_pair=False,
                   exchange_threshold=0.1,
                   use_pseudo_label=False,
                   pseudo_path=None,
                   pseudo_random_state=42,
                   pseudo_rate=0.1,
                   pseudo_index=0,
                   pseudo_name=None,
                   exp_name=None):
    config = ModelConfig()
    config.model_type = model_type
    config.input_type = input_type
    config.use_multi_task = use_multi_task
    config.use_harl = use_harl
    config.use_hal = use_hal
    assert not (config.use_harl and config.use_hal)
    config.cate_embed_dim = cate_embed_dim

    config.use_word_input = use_word_input
    config.word_embed_type = word_embed_type
    if config.use_word_input:
        if word_embed_type:
            config.word_embeddings = np.load(
                format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                type=word_embed_type))
            config.word_embed_trainable = word_embed_trainable
            config.word_embed_dim = config.word_embeddings.shape[1]
        else:
            config.word_embeddings = None
            config.word_embed_trainable = True
            config.word_embed_dim = word_embed_dim
        config.word_vocab = pickle_load(
            format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'))
        config.word_vocab_size = len(config.word_vocab) + 2  # 0: mask, 1: padding
    else:
        config.word_vocab = None

    config.use_bert_input = use_bert_input
    config.bert_trainable = bert_trainable
    if config.use_bert_input:
        config.use_bert_type = use_bert_type
        config.dense_after_bert = dense_after_bert
        if config.use_bert_type in ['hidden', 'hidden_pooler'] or \
                (config.use_multi_task and (config.use_harl or config.use_hal)):
            config.output_hidden_state = True
            config.n_last_hidden_layer = n_last_hidden_layer
        else:
            config.output_hidden_state = False
            config.n_last_hidden_layer = 0
    if config.input_type == 'name_desc':
        config.use_pair_input = use_pair_input
    else:
        config.use_pair_input = False

    if config.use_bert_input and max_len is None:
        config.max_len = MAX_LEN[input_type]
    else:
        config.max_len = max_len

    config.cate1_vocab = pickle_load(
        format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate1'))
    config.cate2_vocab = pickle_load(
        format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate2'))
    config.cate3_vocab = pickle_load(
        format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate3'))
    config.all_cate_vocab = pickle_load(
        format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='all_cate'))
    config.idx2cate1 = pickle_load(
        format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate1'))
    config.idx2cate2 = pickle_load(
        format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate2'))
    config.idx2cate3 = pickle_load(
        format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate3'))
    config.idx2all_cate = pickle_load(
        format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='all_cate'))
    config.cate1_to_cate2 = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE1_TO_CATE2_DICT))
    config.cate2_to_cate3 = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE2_TO_CATE3_DICT))
    config.cate1_to_cate3 = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE1_TO_CATE3_DICT))
    config.cate1_count_dict = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE1_COUNT_DICT))
    config.cate2_count_dict = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE2_COUNT_DICT))
    config.cate3_count_dict = pickle_load(format_filename(PROCESSED_DATA_DIR, CATE3_COUNT_DICT))
    config.n_cate1 = len(config.cate1_vocab)
    config.n_cate2 = len(config.cate2_vocab)
    config.n_cate3 = len(config.cate3_vocab)
    config.n_all_cate = len(config.all_cate_vocab)

    if config.use_multi_task and (config.use_harl or config.use_hal):
        config.share_father_pred = 'no'
        config.use_mask_for_cate2 = False
        config.use_mask_for_cate3 = False
        config.cate3_mask_type = None
    else:
        config.share_father_pred = share_father_pred
        config.use_mask_for_cate2 = use_mask_for_cate2
        config.use_mask_for_cate3 = use_mask_for_cate3
        config.cate3_mask_type = cate3_mask_type
        # if config.use_mask_for_cate2:
        if config.use_mask_for_cate3:
            if config.cate3_mask_type == 'cate1':
                config.cate_to_cate3 = pickle_load(
                    format_filename(PROCESSED_DATA_DIR, CATE1_TO_CATE3_DICT))
            elif config.cate3_mask_type == 'cate2':
                config.cate_to_cate3 = pickle_load(
                    format_filename(PROCESSED_DATA_DIR, CATE2_TO_CATE3_DICT))
    config.cate1_loss_weight = cate1_loss_weight
    config.cate2_loss_weight = cate2_loss_weight
    config.cate3_loss_weight = cate3_loss_weight

    config.batch_size = batch_size
    config.predict_batch_size = predict_batch_size
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.optimizer = optimizer
    config.learning_rate = learning_rate
    config.use_focal_loss = use_focal_loss
    config.callbacks_to_add = callbacks_to_add or ['modelcheckpoint', 'earlystopping']
    if 'swa' in config.callbacks_to_add:
        config.swa_start = swa_start
        config.early_stopping_patience = early_stopping_patience
    for lr_scheduler in ['clr', 'sgdr', 'clr_1', 'clr_2', 'warm_up', 'swa_clr']:
        if lr_scheduler in config.callbacks_to_add:
            config.max_lr = max_lr
            config.min_lr = min_lr

    config.train_on_cv = train_on_cv
    if config.train_on_cv:
        config.cv_random_state = cv_random_state
        config.cv_fold = cv_fold
        config.cv_index = cv_index

    config.exchange_pair = exchange_pair
    if config.exchange_pair:
        config.exchange_threshold = exchange_threshold

    config.use_pseudo_label = use_pseudo_label
    if config.use_pseudo_label:
        config.pseudo_path = pseudo_path
        config.pseudo_random_state = pseudo_random_state
        config.pseudo_rate = pseudo_rate
        config.pseudo_index = pseudo_index

    # build experiment name from parameter configuration
    config.exp_name = f'{config.model_type}_{config.input_type}'
    if config.use_pair_input:
        config.exp_name += '_pair'
    config.exp_name += f'_len_{config.max_len}'
    if config.use_word_input:
        config.exp_name += f"_word_{config.word_embed_type}_{'tune' if config.word_embed_trainable else 'fix'}"
    if config.use_bert_input:
        config.exp_name += f"_bert_{config.use_bert_type}_{'tune' if config.bert_trainable else 'fix'}"
        if config.output_hidden_state:
            config.exp_name += f'_hid_{config.n_last_hidden_layer}'
        if config.dense_after_bert:
            config.exp_name += '_dense'
    if config.use_multi_task:
        if config.use_harl:
            config.exp_name += f'_harl_{config.cate_embed_dim}'
        elif config.use_hal:
            config.exp_name += f'_hal_{config.cate_embed_dim}'
        config.exp_name += f'_{config.cate1_loss_weight}_{config.cate2_loss_weight}_{config.cate3_loss_weight}'
    else:
        config.exp_name += f'_not_multi_task'
    if config.share_father_pred in ['after', 'before']:
        config.exp_name += f'_{config.share_father_pred}'
    if config.use_mask_for_cate2:
        config.exp_name += f'_mask_cate2'
    if config.use_mask_for_cate3:
        config.exp_name += f'_mask_cate3_with_{config.cate3_mask_type}'
    if config.use_focal_loss:
        config.exp_name += f'_focal'
    config.exp_name += f'_{config.optimizer}_{config.learning_rate}_{config.batch_size}_{config.n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str
    if config.train_on_cv:
        config.exp_name += f'_{config.cv_random_state}_{config.cv_fold}_{config.cv_index}'
    if config.exchange_pair:
        config.exp_name += f"_ex_pair_{config.exchange_threshold}"
    if config.use_pseudo_label:
        if pseudo_name:
            config.exp_name += f"_{pseudo_name}_pseudo_{pseudo_random_state}_{pseudo_rate}_{pseudo_index}"
        elif 'dev' in config.pseudo_path:
            config.exp_name += f"_dev_pseudo_{pseudo_random_state}_{pseudo_rate}_{pseudo_index}"
        else:
            config.exp_name += f"_test_pseudo_{pseudo_random_state}_{pseudo_rate}_{pseudo_index}"

    if exp_name:
        config.exp_name = exp_name

    return config


def train(config: ModelConfig,
          use_gpu_id=5):
    # see: https://www.bookstack.cn/read/TensorFlow2.0/spilt.6.3b87bc87b85cbe5d.md
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[use_gpu_id], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[use_gpu_id], enable=True)

    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = MultiTaskClsModel[config.model_type](config)
    model.summary()

    train_generator = MultiTaskClsDataGenerator(data_type='train',
                                                batch_size=config.batch_size,
                                                use_multi_task=config.use_multi_task,
                                                input_type=config.input_type,
                                                use_word_input=config.use_word_input,
                                                word_vocab=config.word_vocab,
                                                use_bert_input=config.use_bert_input,
                                                use_pair_input=config.use_pair_input,
                                                bert_model_type=config.model_type,
                                                max_len=config.max_len,
                                                cate1_vocab=config.cate1_vocab,
                                                cate2_vocab=config.cate2_vocab,
                                                cate3_vocab=config.cate3_vocab,
                                                all_cate_vocab=config.all_cate_vocab,
                                                use_mask_for_cate2=config.use_mask_for_cate2,
                                                use_mask_for_cate3=config.use_mask_for_cate3,
                                                cate3_mask_type=config.cate3_mask_type,
                                                cate1_to_cate2=config.cate1_to_cate2,
                                                cate_to_cate3=config.cate_to_cate3,
                                                train_on_cv=config.train_on_cv,
                                                cv_random_state=config.cv_random_state,
                                                cv_fold=config.cv_fold,
                                                cv_index=config.cv_index,
                                                exchange_pair=config.exchange_pair,
                                                exchange_threshold=config.exchange_threshold,
                                                cate3_count_dict=config.cate3_count_dict,
                                                use_pseudo_label=config.use_pseudo_label,
                                                pseudo_path=config.pseudo_path,
                                                pseudo_random_state=config.pseudo_random_state,
                                                pseudo_rate=config.pseudo_rate,
                                                pseudo_index=config.pseudo_index
                                                )
    valid_generator = MultiTaskClsDataGenerator(data_type='dev',
                                                batch_size=config.predict_batch_size,
                                                use_multi_task=True,
                                                input_type=config.input_type,
                                                use_word_input=config.use_word_input,
                                                word_vocab=config.word_vocab,
                                                use_bert_input=config.use_bert_input,
                                                use_pair_input=config.use_pair_input,
                                                bert_model_type=config.model_type,
                                                max_len=config.max_len,
                                                cate1_vocab=config.cate1_vocab,
                                                cate2_vocab=config.cate2_vocab,
                                                cate3_vocab=config.cate3_vocab,
                                                all_cate_vocab=config.all_cate_vocab,
                                                use_mask_for_cate2=config.use_mask_for_cate2,
                                                use_mask_for_cate3=config.use_mask_for_cate3,
                                                cate3_mask_type=config.cate3_mask_type,
                                                cate1_to_cate2=config.cate1_to_cate2,
                                                cate_to_cate3=config.cate_to_cate3,
                                                train_on_cv=config.train_on_cv,
                                                cv_random_state=config.cv_random_state,
                                                cv_fold=config.cv_fold,
                                                cv_index=config.cv_index
                                                )
    test_generator = MultiTaskClsDataGenerator(data_type='test',
                                               batch_size=config.predict_batch_size,
                                               use_multi_task=True,
                                               input_type=config.input_type,
                                               use_word_input=config.use_word_input,
                                               word_vocab=config.word_vocab,
                                               use_bert_input=config.use_bert_input,
                                               use_pair_input=config.use_pair_input,
                                               bert_model_type=config.model_type,
                                               max_len=config.max_len,
                                               cate1_vocab=config.cate1_vocab,
                                               cate2_vocab=config.cate2_vocab,
                                               cate3_vocab=config.cate3_vocab,
                                               all_cate_vocab=config.all_cate_vocab,
                                               use_mask_for_cate2=config.use_mask_for_cate2,
                                               use_mask_for_cate3=config.use_mask_for_cate3,
                                               cate3_mask_type=config.cate3_mask_type,
                                               cate1_to_cate2=config.cate1_to_cate2,
                                               cate_to_cate3=config.cate_to_cate3)

    train_logger = {}
    if not os.path.exists(model_save_path):
        start_time = time.time()
        model.train(train_generator, valid_generator)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_logger['epoch'] = model.return_trained_epoch()
        train_logger['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        train_logger['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    print('Logging Info - Loading best model...')
    model.load_best_model()
    print('Logging Info - Evaluating valid set...')
    eval_results = model.evaluate(valid_generator,
                                  save_diff=True,
                                  save_prob=True,
                                  prob_file=f'{config.exp_name}_dev_prob.pkl',
                                  diff_file=f'{config.exp_name}_diff.txt')

    print('Logging Info - Predicting test set...')
    model.predict(test_generator,
                  save_prob=True,
                  prob_file=f'{config.exp_name}_test_prob.pkl',
                  submit=True,
                  submit_file=f'{config.exp_name}_submit.csv',
                  submit_with_text=True)
    if train_logger:
        train_logger['eval_result'] = eval_results

    swa_type = None
    if 'swa' in config.callbacks_to_add:
        swa_type = 'swa'
    elif 'swa_clr' in config.callbacks_to_add:
        swa_type = 'swa_clr'
    if swa_type:
        print('Logging Info - Loading swa model...')
        model.load_swa_model(swa_type)
        print('Logging Info - Evaluating valid set...')
        swa_results = model.evaluate(valid_generator,
                                     save_prob=True,
                                     prob_file=f'{config.exp_name}_{swa_type}_dev_prob.pkl',
                                     save_diff=True,
                                     diff_file=f'{config.exp_name}_{swa_type}_diff.txt')
        print('Logging Info - Predicting test set...')
        model.predict(test_generator,
                      save_prob=True,
                      prob_file=f'{config.exp_name}_{swa_type}_test_prob.pkl',
                      submit=True,
                      submit_file=f'{config.exp_name}_{swa_type}_submit.csv',
                      submit_with_text=True)
        if train_logger:
            train_logger['swa_result'] = swa_results

    if train_logger:
        writer_md(filename=PERFORMANCE_MD, config=config, trainer_logger=train_logger)

    del model
    gc.collect()
    K.clear_session()


def main(model_type='bert-base-uncased',
         input_type='name_desc',
         use_multi_task=True,
         use_harl=False,
         use_hal=False,
         cate_embed_dim=100,
         use_word_input=False,
         word_embed_type='w2v',
         word_embed_trainable=True,
         word_embed_dim=300,
         use_bert_input=True,
         bert_trainable=True,
         use_bert_type='pooler',
         n_last_hidden_layer=0,
         dense_after_bert=True,
         use_pair_input=True,
         max_len=None,
         share_father_pred='no',
         use_mask_for_cate2=False,
         use_mask_for_cate3=True,
         cate3_mask_type='cate1',
         cate1_loss_weight=1.,
         cate2_loss_weight=1.,
         cate3_loss_weight=1.,
         batch_size=32,
         predict_batch_size=32,
         n_epoch=50,
         learning_rate=2e-5,
         optimizer='adam',
         use_focal_loss=False,
         callbacks_to_add=None,
         swa_start=15,
         early_stopping_patience=5,
         max_lr=6e-5,
         min_lr=1e-5,
         train_on_cv=False,
         cv_random_state=42,
         cv_fold=5,
         cv_index=0,
         exchange_pair=False,
         exchange_threshold=0.1,
         use_pseudo_label=False,
         pseudo_path=None,
         pseudo_random_state=42,
         pseudo_rate=0.1,
         pseudo_index=0,
         pseudo_name=None,
         exp_name=None,
         use_gpu_id=5):
    model_config = prepare_config(model_type=model_type,
                                  input_type=input_type,
                                  use_multi_task=use_multi_task,
                                  use_harl=use_harl,
                                  use_hal=use_hal,
                                  cate_embed_dim=cate_embed_dim,
                                  use_word_input=use_word_input,
                                  word_embed_type=word_embed_type,
                                  word_embed_trainable=word_embed_trainable,
                                  word_embed_dim=word_embed_dim,
                                  use_bert_input=use_bert_input,
                                  bert_trainable=bert_trainable,
                                  use_bert_type=use_bert_type,
                                  n_last_hidden_layer=n_last_hidden_layer,
                                  dense_after_bert=dense_after_bert,
                                  use_pair_input=use_pair_input,
                                  max_len=max_len,
                                  share_father_pred=share_father_pred,
                                  use_mask_for_cate2=use_mask_for_cate2,
                                  use_mask_for_cate3=use_mask_for_cate3,
                                  cate3_mask_type=cate3_mask_type,
                                  cate1_loss_weight=cate1_loss_weight,
                                  cate2_loss_weight=cate2_loss_weight,
                                  cate3_loss_weight=cate3_loss_weight,
                                  batch_size=batch_size,
                                  predict_batch_size=predict_batch_size,
                                  n_epoch=n_epoch,
                                  learning_rate=learning_rate,
                                  optimizer=optimizer,
                                  use_focal_loss=use_focal_loss,
                                  callbacks_to_add=callbacks_to_add,
                                  swa_start=swa_start,
                                  early_stopping_patience=early_stopping_patience,
                                  max_lr=max_lr,
                                  min_lr=min_lr,
                                  train_on_cv=train_on_cv,
                                  cv_random_state=cv_random_state,
                                  cv_fold=cv_fold,
                                  cv_index=cv_index,
                                  exchange_pair=exchange_pair,
                                  exchange_threshold=exchange_threshold,
                                  use_pseudo_label=use_pseudo_label,
                                  pseudo_path=pseudo_path,
                                  pseudo_random_state=pseudo_random_state,
                                  pseudo_rate=pseudo_rate,
                                  pseudo_index=pseudo_index,
                                  pseudo_name=pseudo_name,
                                  exp_name=exp_name)
    train(model_config, use_gpu_id=use_gpu_id)
