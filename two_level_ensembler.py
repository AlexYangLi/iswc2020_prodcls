# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: ensemble.py

@time: 2020/6/8 22:01

@desc:

"""

import os
import numpy as np
from collections import Counter

from config import PREDICT_DIR, PROCESSED_DATA_DIR, TEST_DATA_FILENAME, IDX2TOKEN_TEMPLATE
from utils import pickle_load, format_filename, submit_result


def load_pred_prob_list(model_name_list, data_type):
    pred_prob_cate1_list = []
    pred_prob_cate2_list = []
    pred_prob_cate3_list = []
    for model_name in model_name_list:
        pred_probs = pickle_load(os.path.join(PREDICT_DIR, f'{model_name}_{data_type}_prob.pkl'))
        pred_prob_cate1_list.append(pred_probs[0])
        pred_prob_cate2_list.append(pred_probs[1])
        pred_prob_cate3_list.append(pred_probs[2])
    return pred_prob_cate1_list, pred_prob_cate2_list, pred_prob_cate3_list


def load_idx2cate():
    idx2cate1 = pickle_load(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate1'))
    idx2cate2 = pickle_load(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate2'))
    idx2cate3 = pickle_load(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate3'))
    return idx2cate1, idx2cate2, idx2cate3


def avg_ensemble(pred_prob_list, return_prob=False, weights=None):
    model_weights = weights or [1.] * len(pred_prob_list)
    assert np.sum(model_weights) == len(pred_prob_list)
    weighted_pred_prob_list = [pred_prob * weight for weight, pred_prob in zip(model_weights, pred_prob_list)]

    avg_pred_prob = np.mean(np.stack(weighted_pred_prob_list, axis=2), axis=2)
    if return_prob:
        return np.argmax(avg_pred_prob, axis=1).tolist(), avg_pred_prob
    else:
        return np.argmax(avg_pred_prob, axis=1).tolist()


def vote_ensemble(pred_prob_list, weights=None):
    model_weights = weights or [1] * len(pred_prob_list)
    pred_label_list = [np.argmax(pred_prob, axis=1) for pred_prob in pred_prob_list]
    ensemble_pred_label = []
    for sample_pred in zip(*pred_label_list):
        label_counter = Counter()
        for idx, pred in enumerate(sample_pred):
            label_counter[pred] += 1 * model_weights[idx]
        majority_label, majority_count = label_counter.most_common(1)[0]
        ensemble_pred_label.append(majority_label)
    return ensemble_pred_label


def voting_of_averaging(prefix_model_name_list,
                        submit_file_prefix,
                        cv_random_state=42,
                        cv_fold=5,
                        use_ex_pair=False,
                        ex_threshold=0.1,
                        use_pseudo=False,
                        pseudo_random_state=42,
                        pseudo_rate=5):
    use_pseudo_str = 'pseudo_' if use_pseudo else ''
    use_ex_pair_str = 'ex_pair_' if use_ex_pair else ''
    cv_pred_prob_cate1_list, cv_pred_prob_cate2_list, cv_pred_prob_cate3_list = [], [], []
    test_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME))
    idx2cate1, idx2cate2, idx2cate3 = load_idx2cate()
    for model_name in prefix_model_name_list:
        ex_pair_str = f'_ex_pair_{ex_threshold}' if use_ex_pair else ''
        cv_model_name_list = [f'{model_name}_{cv_random_state}_{cv_fold}_{fold}{ex_pair_str}'
                              for fold in range(cv_fold)]
        if use_pseudo:
            cv_model_name_list = [f'{cv_model_name}_cv_test_vote_pseudo' \
                                  f'_{pseudo_random_state}_{pseudo_rate}_{fold}'
                                  for fold, cv_model_name in enumerate(cv_model_name_list)]
        pred_prob_cate1_list, pred_prob_cate2_list, pred_prob_cate3_list = load_pred_prob_list(
            cv_model_name_list, data_type='test')

        print(f'Logging Info - average ensemble for {model_name}...')
        cv_pred_cate1, cv_pred_prob_cate1 = avg_ensemble(pred_prob_cate1_list, return_prob=True)
        cv_pred_cate2, cv_pred_prob_cate2 = avg_ensemble(pred_prob_cate2_list, return_prob=True)
        cv_pred_cate3, cv_pred_prob_cate3 = avg_ensemble(pred_prob_cate3_list, return_prob=True)
        submit_result(test_data=test_data,
                      pred_cate1_list=cv_pred_cate1,
                      pred_cate2_list=cv_pred_cate2,
                      pred_cate3_list=cv_pred_cate3,
                      idx2cate1=idx2cate1,
                      idx2cate2=idx2cate2,
                      idx2cate3=idx2cate3,
                      submit_file=f'{use_ex_pair_str}{use_pseudo_str}{submit_file_prefix}_{model_name}_test_avg_ensemble.csv',
                      submit_with_text=True)
        cv_pred_prob_cate1_list.append(cv_pred_prob_cate1)
        cv_pred_prob_cate2_list.append(cv_pred_prob_cate2)
        cv_pred_prob_cate3_list.append(cv_pred_prob_cate3)

        print(f'Logging Info - voting ensemble for {model_name}...')
        cv_pred_cate1 = vote_ensemble(pred_prob_cate1_list)
        cv_pred_cate2 = vote_ensemble(pred_prob_cate2_list)
        cv_pred_cate3 = vote_ensemble(pred_prob_cate3_list)
        submit_result(test_data=test_data,
                      pred_cate1_list=cv_pred_cate1,
                      pred_cate2_list=cv_pred_cate2,
                      pred_cate3_list=cv_pred_cate3,
                      idx2cate1=idx2cate1,
                      idx2cate2=idx2cate2,
                      idx2cate3=idx2cate3,
                      submit_file=f'{use_ex_pair_str}{use_pseudo_str}{submit_file_prefix}_{model_name}_test_vote_ensemble.csv',
                      submit_with_text=True)

    print(f'Logging Info - average ensemble for all cross validation model...')
    cv_pred_cate1 = avg_ensemble(cv_pred_prob_cate1_list)
    cv_pred_cate2 = avg_ensemble(cv_pred_prob_cate2_list)
    cv_pred_cate3 = avg_ensemble(cv_pred_prob_cate3_list)
    submit_result(test_data=test_data,
                  pred_cate1_list=cv_pred_cate1,
                  pred_cate2_list=cv_pred_cate2,
                  pred_cate3_list=cv_pred_cate3,
                  idx2cate1=idx2cate1,
                  idx2cate2=idx2cate2,
                  idx2cate3=idx2cate3,
                  submit_file=f'{use_ex_pair_str}{use_pseudo_str}{submit_file_prefix}_test_avg_ensemble.csv',
                  submit_with_text=True)

    print(f'Logging Info - voting ensemble for all cross validation model...')
    cv_pred_cate1 = vote_ensemble(cv_pred_prob_cate1_list)
    cv_pred_cate2 = vote_ensemble(cv_pred_prob_cate2_list)
    cv_pred_cate3 = vote_ensemble(cv_pred_prob_cate3_list)
    submit_result(test_data=test_data,
                  pred_cate1_list=cv_pred_cate1,
                  pred_cate2_list=cv_pred_cate2,
                  pred_cate3_list=cv_pred_cate3,
                  idx2cate1=idx2cate1,
                  idx2cate2=idx2cate2,
                  idx2cate3=idx2cate3,
                  submit_file=f'{use_ex_pair_str}{use_pseudo_str}{submit_file_prefix}_test_vote_ensemble.csv',
                  submit_with_text=True)
