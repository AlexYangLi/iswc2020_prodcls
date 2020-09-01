# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2020/5/21 22:06

@desc:

"""

from trainer import main

if __name__ == '__main__':

    # train without pseudo labeling
    for n_hidden in range(1, 6):
        for cv_index in range(5):
            main(model_type='bert-base-uncased',
                 input_type='name_desc',
                 use_word_input=False,
                 use_bert_input=True,
                 bert_trainable=True,
                 batch_size=32,
                 predict_batch_size=32,
                 use_pair_input=True,
                 use_bert_type='hidden',
                 n_last_hidden_layer=n_hidden,
                 dense_after_bert=True,
                 learning_rate=2e-5,
                 use_multi_task=True,
                 use_harl=False,
                 use_mask_for_cate2=False,
                 use_mask_for_cate3=True,
                 cate3_mask_type='cate1',
                 train_on_cv=True,
                 cv_random_state=42,
                 cv_fold=5,
                 cv_index=cv_index,
                 exchange_pair=True,
                 use_pseudo_label=False,
                 use_gpu_id=7)

    for n_hidden in range(1, 6):
        for cv_index in range(5):
            main(model_type='bert-base-uncased',
                 input_type='name_desc',
                 use_word_input=False,
                 use_bert_input=True,
                 bert_trainable=True,
                 batch_size=32,
                 predict_batch_size=32,
                 use_pair_input=True,
                 use_bert_type='hidden_pooler',
                 n_last_hidden_layer=n_hidden,
                 dense_after_bert=True,
                 learning_rate=2e-5,
                 use_multi_task=True,
                 use_harl=False,
                 use_mask_for_cate2=False,
                 use_mask_for_cate3=True,
                 cate3_mask_type='cate1',
                 train_on_cv=True,
                 cv_random_state=42,
                 cv_fold=5,
                 cv_index=cv_index,
                 exchange_pair=True,
                 use_pseudo_label=False,
                 use_gpu_id=6)

    for use_bert_type in ['lstm', 'gru', 'lstm_gru', 'gru_lstm', 'cnn', 'lstm_cnn']:
        for cv_index in range(5):
            main(model_type='bert-base-uncased',
                 input_type='name_desc',
                 use_word_input=False,
                 use_bert_input=True,
                 bert_trainable=True,
                 batch_size=32,
                 predict_batch_size=32,
                 use_pair_input=True,
                 use_bert_type=use_bert_type,
                 n_last_hidden_layer=0,
                 dense_after_bert=True,
                 learning_rate=2e-5,
                 use_multi_task=True,
                 use_harl=False,
                 use_mask_for_cate2=False,
                 use_mask_for_cate3=True,
                 cate3_mask_type='cate1',
                 train_on_cv=True,
                 cv_random_state=42,
                 cv_fold=5,
                 cv_index=cv_index,
                 exchange_pair=True,
                 use_pseudo_label=False,
                 use_gpu_id=5)

    for cv_index in range(5):
        main(model_type='bert-base-uncased',
             input_type='name_desc',
             use_word_input=False,
             use_bert_input=True,
             bert_trainable=True,
             batch_size=32,
             predict_batch_size=32,
             use_pair_input=True,
             use_bert_type='hidden',
             n_last_hidden_layer=1,
             dense_after_bert=False,
             learning_rate=2e-5,
             use_multi_task=True,
             use_harl=True,
             use_mask_for_cate2=False,
             use_mask_for_cate3=True,
             cate3_mask_type='cate1',
             train_on_cv=True,
             cv_random_state=42,
             cv_fold=5,
             cv_index=cv_index,
             exchange_pair=True,
             use_pseudo_label=False,
             use_gpu_id=4)

    for cv_index in range(5):
        main(model_type='bert-base-uncased',
             input_type='name_desc',
             use_word_input=False,
             use_bert_input=True,
             bert_trainable=True,
             batch_size=32,
             predict_batch_size=32,
             use_pair_input=True,
             use_bert_type='pooler',
             n_last_hidden_layer=0,
             dense_after_bert=False,
             learning_rate=2e-5,
             use_multi_task=True,
             use_mask_for_cate2=False,
             use_mask_for_cate3=True,
             cate3_mask_type='cate1',
             train_on_cv=True,
             cv_random_state=42,
             cv_fold=5,
             cv_index=cv_index,
             exchange_pair=True,
             use_pseudo_label=False,
             use_gpu_id=3)
