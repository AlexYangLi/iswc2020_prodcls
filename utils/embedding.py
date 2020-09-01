# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: embedding.py

@time: 2020/4/26 13:34

@desc:

"""

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText


def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()

            try:
                word = line[0]
                word_vector = np.array([float(v) for v in line[1:]])
            except ValueError:
                continue

            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)

            if len(word_vector) != embeddings_dim:
                continue

            word_vectors[word] = word_vector

    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())

    return word_vectors, embeddings_dim


def load_pre_trained(load_filename, vocabulary=None):
    word_vectors = {}
    try:
        model = KeyedVectors.load_word2vec_format(load_filename)
        weights = model.wv.syn0
        embedding_dim = weights.shape[1]
        for k, v in model.wv.vocab.items():
            word_vectors[k] = weights[v.index, :]
    except ValueError:
        word_vectors, embedding_dim = load_glove_format(load_filename)

    if vocabulary is not None:
        emb = np.zeros(shape=(len(vocabulary) + 2, embedding_dim), dtype='float32')
        emb[1] = np.random.normal(0, 0.05, embedding_dim)

        nb_unk = 0
        for w, i in vocabulary.items():
            if w not in word_vectors:
                nb_unk += 1
                emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
            else:
                emb[i, :] = word_vectors[w]
        print('Logging Info - From {} Embedding matrix created : {}, unknown tokens: {}'.format(load_filename, emb.shape,
                                                                                                nb_unk))
        return emb
    else:
        print('Logging Info - Loading {} Embedding : {}'.format(load_filename, (len(word_vectors), embedding_dim)))
        return word_vectors


def train_w2v(corpus, vocabulary, embedding_dim=300):
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocabulary) + 2, embedding_dim), dtype='float32')     # 0 for mask, 1 for unknown token
    emb[1] = np.random.normal(0, 0.05, embedding_dim)

    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in d:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    print('Logging Info - Word2Vec Embedding matrix created: {}, unknown tokens: {}'.format(emb.shape, nb_unk))
    return emb


def train_fasttext(corpus, vocabulary, embedding_dim=300):
    model = FastText(size=embedding_dim, min_count=1, window=5, sg=1, word_ngrams=1)
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

    emb = np.zeros(shape=(len(vocabulary) + 2, embedding_dim), dtype='float32')     # 0 for mask, 1 for unknown token
    emb[1] = np.random.normal(0, 0.05, embedding_dim)

    for w, i in vocabulary.items():
        emb[i, :] = model.wv[w]  # note that oov words can still have word vectors

    print('Logging Info - Fasttext Embedding matrix created: {}'.format(emb.shape))
    return emb
