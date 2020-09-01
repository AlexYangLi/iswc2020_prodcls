# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: ensemble.py

@time: 2020/6/8 22:01

@desc:

"""


from two_level_ensembler import voting_of_averaging


if __name__ == '__main__':
    '''4. 伪标签交叉验证模型集成'''
    cv_model_list = [
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_pooler_tune_hid_5_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_pooler_tune_hid_4_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_pooler_tune_hid_3_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_pooler_tune_hid_2_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_pooler_tune_hid_1_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_5_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_4_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_3_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_2_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_1_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_hidden_tune_hid_1_harl_100_1.0_1.0_1.0_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_lstm_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_gru_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_gru_lstm_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_lstm_gru_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_cnn_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_lstm_cnn_tune_dense_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50',
        'bert-base-uncased_name_desc_pair_len_200_bert_pooler_tune_1.0_1.0_1.0_mask_cate3_with_cate1_adam_2e-05_32_50'
    ]
    voting_of_averaging(prefix_model_name_list=cv_model_list,
                        submit_file_prefix='cross_validation',
                        cv_random_state=42,
                        cv_fold=5,
                        use_ex_pair=False,
                        use_pseudo=True,
                        pseudo_random_state=42,
                        pseudo_rate=5)
