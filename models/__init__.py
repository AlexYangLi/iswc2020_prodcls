# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: __init__.py.py

@time: 2020/5/21 7:24

@desc:

"""

from .multitask_classify_model import BiLSTM, CNNRNN, DPCNN, MultiTextCNN, RCNN, RNNCNN, TextCNN, BertClsModel, \
    RobertaClsModel, XLNetClsModel, GPT2ClsModel, TransfoXLClsModel, DistllBertClsModel
from .sklearn_base_model import BernoulliNBModel, DecisionTreeModel, ExtraTreeModel, EnsembleExtraTreeModel, \
    GaussianNBModel, KNeighborsModel, LDAModel, LinearSVCModel, LRCVModel, LRModel, MLPModel, RandomForestModel, \
    GBDTModel, XGBoostModel

MultiTaskClsModel = {
    'bilstm': BiLSTM,
    'cnnrnn': CNNRNN,
    'dpcnn': DPCNN,
    'multicnn': MultiTextCNN,
    'rnncnn': RNNCNN,
    'cnn': TextCNN,
    'bert-base-uncased': BertClsModel,
    'bert-base-cased': BertClsModel,
    'bert-large-uncased': BertClsModel,
    'bert-large-uncased-whole-word-masking': BertClsModel,
    'bert-large-uncased-whole-word-masking-finetuned-squad': BertClsModel,
    'roberta-base': RobertaClsModel,
    'prod-bert-base-uncased': BertClsModel,
    'prod-roberta-base-cased': RobertaClsModel,
    'roberta-large': RobertaClsModel,
    'roberta-large-mnli': RobertaClsModel,
    'distilroberta-base': RobertaClsModel,
    'tune_bert-base-uncased_nsp': BertClsModel,
    'xlnet-base-cased': XLNetClsModel,
    'albert-base-v1': BertClsModel,
    'albert-large-v1': BertClsModel,
    'albert-xlarge-v1': BertClsModel,
    'albert-xxlarge-v1': BertClsModel,
    'gpt2': GPT2ClsModel,
    'gpt2-medium': GPT2ClsModel,
    'transfo-xl': TransfoXLClsModel,
    'distilbert-base-uncased': DistllBertClsModel,
    'distilbert-base-uncased-distilled-squad': DistllBertClsModel,
}


SklearnEnsembleModel = {
    'bnb': BernoulliNBModel,  # 输出概率
    'dt': DecisionTreeModel,  # 输出准确值
    'et': ExtraTreeModel,  # 输出准确值
    'eet': EnsembleExtraTreeModel,  # 输出准确值
    'gnb': GaussianNBModel,  # 输出准确值
    'kn': KNeighborsModel,  # 输出准确值
    'lda': LDAModel,
    'svc': LinearSVCModel,
    'lr': LRModel,
    'mlp': MLPModel,
    'rf': RandomForestModel,
    'gbdt': GBDTModel,
    'xgboost': XGBoostModel
}
