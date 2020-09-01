# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: preprocess.py

@time: 2020/5/19 21:55

@desc:

"""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from config import PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, SUBMIT_DIR, RAW_TRAIN_FILENAME, RAW_DEV_FILENAME, \
    TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, VOCABULARY_TEMPLATE, IDX2TOKEN_TEMPLATE, CATE1_TO_CATE2_DICT, \
    CATE1_TO_CATE3_DICT, CATE2_TO_CATE3_DICT, RAW_TEST_FILENAME, TEST_DATA_FILENAME, TRAIN_CV_DATA_TEMPLATE, \
    DEV_CV_DATA_TEMPLATE, PREDICT_DIR, CATE1_COUNT_DICT, CATE2_COUNT_DICT, CATE3_COUNT_DICT
from utils import ensure_dir, format_filename, pickle_dump


def load_raw_data(file_path, data_type='train'):
    data = {
        'id': [],
        'name': [],
        'desc': [],
        'cate1': [],
        'cate2': [],
        'cate3': []
    }

    cate1_to_cate2 = dict()
    cate2_to_cate3 = dict()
    cate1_to_cate3 = dict()

    with open(file_path, 'r') as reader:
        for line in reader:
            product_info = json.loads(line.strip())
            data['id'].append(product_info['ID'])
            data['name'].append(product_info['Name'])
            data['desc'].append(product_info['Description'])

            if data_type != 'test':
                cate1 = product_info['lvl1']
                cate2 = product_info['lvl2']
                cate3 = product_info['lvl3']
                data['cate1'].append(cate1)
                data['cate2'].append(cate2)
                data['cate3'].append(cate3)

                if cate1 not in cate1_to_cate2:
                    cate1_to_cate2[cate1] = set()
                cate1_to_cate2[cate1].add(cate2)

                if cate2 not in cate2_to_cate3:
                    cate2_to_cate3[cate2] = set()
                cate2_to_cate3[cate2].add(cate3)

                if cate1 not in cate1_to_cate3:
                    cate1_to_cate3[cate1] = set()
                cate1_to_cate3[cate1].add(cate3)

    print('Logging Info - Data Size:', len(data['id']))
    if data_type == 'train':
        assert len(data['id']) == len(data['name']) == len(data['desc']) == len(data['cate1']) == len(
            data['cate2']) == len(data['cate3'])
        return data, cate1_to_cate2, cate2_to_cate3, cate1_to_cate3
    else:
        return data


def load_cate_count(data, cate_key):
    cate_count = {}
    for cate in data[cate_key]:
        if cate not in cate_count:
            cate_count[cate] = 0
        cate_count[cate] += 1
    for cate in cate_count:
        cate_count[cate] /= len(data[cate_key])
    return cate_count


def load_vocab_and_corpus(data, cut_func, min_count=1):
    print('Logging Info: Constructing vocabulary and corpus...')
    tokens = dict()
    corpus = []
    for name, desc in zip(data['name'], data['desc']):
        if desc is None or desc == '':
            text = name
        else:
            text = f'{name} {desc}'
        text_cut = cut_func(text)
        for token in text_cut:
            tokens[token] = tokens.get(token, 0) + 1
        corpus.append(text_cut)
    tokens = [token for token, token_count in tokens.items() if token_count >= min_count]
    idx2token = {idx + 2: token for idx, token in enumerate(tokens)}  # 0: mask, 1: padding
    token2idx = {token: idx for idx, token in idx2token.items()}
    print(f'Logging Info - Token Vocabulary: {len(token2idx)}')
    return token2idx, idx2token, corpus


def load_label_vocab(data):
    cate1_vocab = {}
    cate2_vocab = {}
    cate3_vocab = {}
    all_cate_vocab = {}
    for cate1, cate2, cate3 in zip(data['cate1'], data['cate2'], data['cate3']):
        if cate1 not in cate1_vocab:
            cate1_vocab[cate1] = len(cate1_vocab)
        if cate2 not in cate2_vocab:
            cate2_vocab[cate2] = len(cate2_vocab)
        if cate3 not in cate3_vocab:
            cate3_vocab[cate3] = len(cate3_vocab)
        all_cate = f"{cate1}|{cate2}|{cate3}"
        if all_cate not in all_cate_vocab:
            all_cate_vocab[all_cate] = len(all_cate_vocab)

    print(f'Logging Info - Level 1 Category: {len(cate1_vocab)}')
    print(f'Logging Info - Level 2 Category: {len(cate2_vocab)}')
    print(f'Logging Info - Level 3 Category: {len(cate3_vocab)}')
    print(f'Logging Info - All Level Category: {len(all_cate_vocab)}')

    return cate1_vocab, cate2_vocab, cate3_vocab, all_cate_vocab


def convert_to_id(cate1_to_cate2, cate1_vocab, cate2_vocab):
    cate1_to_cate2_id = dict()
    for cate1 in cate1_to_cate2:
        cate1_id = cate1_vocab[cate1]
        cate1_to_cate2_id[cate1_id] = set(map(cate2_vocab.get, cate1_to_cate2[cate1]))
    return cate1_to_cate2_id


def cv_split(train_data, dev_data, cate3_vocab, fold=5, balanced=True, random_state=42):
    def indexing_data(data, indices):
        part_data = {}
        for k in data.keys():
            part_data[k] = [data[k][i] for i in indices]
        return part_data

    all_data = {}
    for key in train_data.keys():
        all_data[key] = train_data[key] + dev_data[key]

    # some category in validation set is not in cate3_vocab
    cate3_id_list = [cate3_vocab.get(cate3, 0) for cate3 in all_data['cate3']]
    index_range = np.arange(len(all_data['id']))

    if balanced:
        kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)

    for idx, (train_index, dev_index) in enumerate(kf.split(index_range, cate3_id_list)):
        train_data_fold = indexing_data(all_data, train_index)
        dev_data_fold = indexing_data(all_data, dev_index)

        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_CV_DATA_TEMPLATE, random=random_state,
                                    fold=fold, index=idx), train_data_fold)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_CV_DATA_TEMPLATE, random=random_state,
                                    fold=fold, index=idx), dev_data_fold)


if __name__ == '__main__':
    # create directory
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(MODEL_SAVED_DIR)
    ensure_dir(SUBMIT_DIR)
    ensure_dir(PREDICT_DIR)

    # read dataset
    train_data, cate1_to_cate2, cate2_to_cate3, cate1_to_cate3 = load_raw_data(RAW_TRAIN_FILENAME,
                                                                               data_type='train')
    dev_data = load_raw_data(RAW_DEV_FILENAME, data_type='dev')
    test_data = load_raw_data(RAW_TEST_FILENAME, data_type='test')
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME), train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME), dev_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME), test_data)

    cate1_count = load_cate_count(train_data, 'cate1')
    cate2_count = load_cate_count(train_data, 'cate2')
    cate3_count = load_cate_count(train_data, 'cate3')
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE1_COUNT_DICT), cate1_count)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE2_COUNT_DICT), cate2_count)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE3_COUNT_DICT), cate3_count)

    cate1_vocab, cate2_vocab, cate3_vocab, all_cate_vocab = load_label_vocab(train_data)
    idx2cate1 = dict((idx, cate) for cate, idx in cate1_vocab.items())
    idx2cate2 = dict((idx, cate) for cate, idx in cate2_vocab.items())
    idx2cate3 = dict((idx, cate) for cate, idx in cate3_vocab.items())
    idx2all_cate = dict((idx, cate) for cate, idx in all_cate_vocab.items())
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate1'), cate1_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate1'), idx2cate1)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate2'), cate2_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate2'), idx2cate2)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='cate3'), cate3_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='cate3'), idx2cate3)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='all_cate'), all_cate_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='all_cate'), idx2all_cate)

    cate1_to_cate2_id = convert_to_id(cate1_to_cate2, cate1_vocab, cate2_vocab)
    cate2_to_cate3_id = convert_to_id(cate2_to_cate3, cate2_vocab, cate3_vocab)
    cate1_to_cate3_id = convert_to_id(cate1_to_cate3, cate1_vocab, cate3_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE1_TO_CATE2_DICT), cate1_to_cate2_id)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE2_TO_CATE3_DICT), cate2_to_cate3_id)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, CATE1_TO_CATE3_DICT), cate1_to_cate3_id)

    cv_split(train_data, dev_data, cate3_vocab, fold=5, balanced=True, random_state=42)
