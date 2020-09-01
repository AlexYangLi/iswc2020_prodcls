# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: io.py

@time: 2020/5/8 9:52

@desc:

"""

import os
import json
import pickle

import numpy as np

from config import ModelConfig, PREDICT_DIR, SUBMIT_DIR, LOG_DIR


def pickle_load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)
    except EOFError:
        print('Logging Error - Cannot load:', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def write_log(filename, log, mode='w'):
    with open(filename, mode) as writer:
        writer.write('\n')
        json.dump(log, writer, indent=4, ensure_ascii=False)


def writer_md(filename, config: ModelConfig, trainer_logger, mode='a'):
    with open(os.path.join(LOG_DIR, filename), mode) as writer:
        parameters = f"|{config.exp_name:^100}|{trainer_logger['epoch']:^2}"
        writer.write(parameters)
        performance = '|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|'.format(
            trainer_logger['eval_result']['cate1_p'],
            trainer_logger['eval_result']['cate1_r'],
            trainer_logger['eval_result']['cate1_f1'],
            trainer_logger['eval_result']['cate2_p'],
            trainer_logger['eval_result']['cate2_r'],
            trainer_logger['eval_result']['cate2_f1'],
            trainer_logger['eval_result']['cate3_p'],
            trainer_logger['eval_result']['cate3_r'],
            trainer_logger['eval_result']['cate3_f1'],
            trainer_logger['eval_result']['val_p'],
            trainer_logger['eval_result']['val_r'],
            trainer_logger['eval_result']['val_f1']
        )
        performance += '{:^8}|{:^20}|'.format(
            trainer_logger['train_time'],
            trainer_logger['timestamp']
        )
        writer.write(performance)
        writer.write('\n')

        if 'swa_result' in trainer_logger:
            exp_name = config.exp_name + '_swa'
            parameters = f"|{exp_name:^100}|{trainer_logger['epoch']:^2}"
            writer.write(parameters)
            performance = '|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|'.format(
                trainer_logger['swa_result']['cate1_p'],
                trainer_logger['swa_result']['cate1_r'],
                trainer_logger['swa_result']['cate1_f1'],
                trainer_logger['swa_result']['cate2_p'],
                trainer_logger['swa_result']['cate2_r'],
                trainer_logger['swa_result']['cate2_f1'],
                trainer_logger['swa_result']['cate3_p'],
                trainer_logger['swa_result']['cate3_r'],
                trainer_logger['swa_result']['cate3_f1'],
                trainer_logger['swa_result']['val_p'],
                trainer_logger['swa_result']['val_r'],
                trainer_logger['swa_result']['val_f1']
            )
            performance += '{:^8}|{:^20}|'.format(
                trainer_logger['train_time'],
                trainer_logger['timestamp']
            )

            writer.write(performance)
            writer.write('\n')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_filename(_dir, filename_template, **kwargs):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename


def submit_result(test_data, pred_cate1_list, pred_cate2_list, pred_cate3_list, idx2cate1, idx2cate2, idx2cate3,
                  submit_file, submit_with_text=False):
    test_id_list = test_data['id']
    assert len(test_id_list) == len(pred_cate1_list) == len(pred_cate2_list) == len(pred_cate3_list)
    with open(os.path.join(SUBMIT_DIR, submit_file), 'w', encoding='utf8') as writer:
        for idx, (test_id, pred_cate1_id, pred_cate2_id, pred_cate3_id) in enumerate(zip(test_id_list,
                                                                                         pred_cate1_list,
                                                                                         pred_cate2_list,
                                                                                         pred_cate3_list)):

            pred_cate1 = idx2cate1[pred_cate1_id]
            pred_cate2 = idx2cate2[pred_cate2_id]
            pred_cate3 = idx2cate3[pred_cate3_id]
            if submit_with_text:
                test_name = test_data['name'][idx]
                test_desc = test_data['desc'][idx]
                writer.write(f'{test_id}##{test_name}##{test_desc}##{pred_cate1}##{pred_cate2}##{pred_cate3}\n')
            else:
                writer.write(f'{test_id},{pred_cate1},{pred_cate2},{pred_cate3}\n')


def save_prob_to_file(pred_prob_cate1_list, pred_prob_cate2_list, pred_prob_cate3_list, pred_prob_all_cate_list,
                      prob_file, use_multi_task):
    if not use_multi_task:
        pickle_dump(os.path.join(prob_file), np.vstack(pred_prob_all_cate_list))
    else:
        pickle_dump(os.path.join(PREDICT_DIR, prob_file),
                    [np.vstack(pred_prob_cate1_list),
                     np.vstack(pred_prob_cate2_list),
                     np.vstack(pred_prob_cate3_list)])


def save_diff_to_file(data, pred_cate1_list, pred_cate2_list, pred_cate3_list, true_cate1_list, true_cate2_list,
                      true_cate3_list, cate1_to_cate2, cate2_to_cate3, cate1_to_cate3, idx2cate1, idx2cate2, idx2cate3,
                      cate1_count_dict, cate2_count_dict, cate3_count_dict,
                      diff_file):
    with open(os.path.join(LOG_DIR, diff_file), 'w') as writer:
        for i, data_id in enumerate(data['id']):
            t_cate1, p_cate1 = true_cate1_list[i], pred_cate1_list[i]
            t_cate2, p_cate2 = true_cate2_list[i], pred_cate2_list[i]
            t_cate3, p_cate3 = true_cate3_list[i], pred_cate3_list[i]

            cate1_diff = t_cate1 != p_cate1
            cate2_diff = t_cate2 != p_cate2
            cate3_diff = t_cate3 != p_cate3

            cate2_in_cate1 = p_cate2 in cate1_to_cate2[p_cate1]
            cate3_in_cate2 = p_cate3 in cate2_to_cate3[p_cate2]
            cate3_in_cate1 = p_cate3 in cate1_to_cate3[p_cate1]

            cate1_count = cate1_count_dict[idx2cate1[t_cate1]]
            cate2_count = cate2_count_dict[idx2cate2[t_cate2]]
            cate3_count = cate3_count_dict[idx2cate3[t_cate3]]

            if cate1_diff or cate2_diff or cate3_diff:
                writer.write(f'{data_id}\t')
                writer.write(f'{not cate1_diff}\t{not cate2_diff}\t{not cate3_diff}\t')
                writer.write(f'{cate2_in_cate1}\t{cate3_in_cate2}\t{cate3_in_cate1}\t')
                writer.write(f'{cate1_count:.6f}\t{cate2_count:.6f}\t{cate3_count:.6f}\t')
                writer.write(f'{idx2cate1[t_cate1]}|{idx2cate1[p_cate1]}\t')
                writer.write(f'{idx2cate2[t_cate2]}|{idx2cate2[p_cate2]}\t')
                writer.write(f'{idx2cate3[t_cate3]}|'
                             f'{idx2cate3[p_cate3]}')
                writer.write('\n')
