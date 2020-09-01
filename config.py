# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2020/5/17 15:33

@desc:

"""

from os import path

RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'
SUBMIT_DIR = './submit'
IMG_DIR = './img'
PREDICT_DIR = './predict'

NLTK_DATA = path.join(RAW_DATA_DIR, 'nltk_data')
RAW_TRAIN_FILENAME = path.join(RAW_DATA_DIR, 'train.json')
RAW_DEV_FILENAME = path.join(RAW_DATA_DIR, 'validation.json')
RAW_TEST_FILENAME = path.join(RAW_DATA_DIR, 'test_public.json')

TRAIN_DATA_FILENAME = 'train.pkl'
TRAIN_CV_DATA_TEMPLATE = 'train_{random}_{fold}_{index}.pkl'
DEV_CV_DATA_TEMPLATE = 'dev_{random}_{fold}_{index}.pkl'
DEV_DATA_FILENAME = 'dev.pkl'

TEST_DATA_FILENAME = 'test.pkl'

VOCABULARY_TEMPLATE = '{level}_vocab.pkl'
IDX2TOKEN_TEMPLATE = 'idx2{level}.pkl'
EMBEDDING_MATRIX_TEMPLATE = '{type}_embeddings.npy'
PERFORMANCE_MD = 'performance.md'

CATE1_TO_CATE2_DICT = 'cate1_to_cate2.pkk'
CATE1_TO_CATE3_DICT = 'cate1_to_cate3.pkl'
CATE2_TO_CATE3_DICT = 'cate2_to_cate3.pkl'

CATE1_COUNT_DICT = 'cate1_count_dict.pkl'
CATE2_COUNT_DICT = 'cate2_count_dict.pkl'
CATE3_COUNT_DICT = 'cate3_count_dict.pkl'

RANDOM_SEED = 2020

EXTERNAL_EMBEDDINGS_DIR = path.join(RAW_DATA_DIR, 'embeddings')

BERT_VOCAB_FILE = {
    'bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-uncased', 'bert-base-uncased-vocab.txt'),
    'bert-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-cased', 'bert-base-cased-vocab.txt'),
    'bert-large-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased', 'bert-large-uncased-vocab.txt'),
    'bert-large-uncased-whole-word-masking': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking', 'bert-large-uncased-whole-word-masking-vocab.txt'),
    'bert-large-uncased-whole-word-masking-finetuned-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt'),
    'roberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-base', 'roberta-base-vocab.json'),
    'roberta-large': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large', 'roberta-large-vocab.json'),
    'roberta-large-mnli': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large-mnli', 'roberta-large-mnli-vocab.json'),
    'distilroberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilroberta-base', 'distilroberta-base-vocab.json'),
    'prod-bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-bert-base-uncased', 'vocab.txt'),
    'prod-roberta-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-roberta-base-cased', 'vocab.json'),
    'tune_bert-base-uncased_nsp': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-uncased', 'bert-base-uncased-vocab.txt'),
    'xlnet-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'xlnet-base-cased', 'xlnet-base-cased-spiece.model'),
    'albert-base-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-base-v1', 'albert-base-v1-spiece.model'),
    'albert-large-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-large-v1', 'albert-large-v1-spiece.model'),
    'albert-xlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xlarge-v1', 'albert-xlarge-v1-spiece.model'),
    'albert-xxlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xxlarge-v1', 'albert-xxlarge-v1-spiece.model'),
    'gpt2': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2', 'gpt2-vocab.json'),
    'gpt2-medium': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2-medium', 'gpt2-medium-vocab.json'),
    'transfo-xl': path.join(EXTERNAL_EMBEDDINGS_DIR, 'transfo-xl', 'transfo-xl-wt103-vocab.json'),
    'distilbert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased', 'bert-base-uncased-vocab.txt'),
    'distilbert-base-uncased-distilled-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased-distilled-squad', 'bert-large-uncased-vocab.txt'),
}

BERT_CONFIG_FILE = {
    'bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-uncased', 'bert-base-uncased-config.json'),
    'bert-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-cased', 'bert-base-cased-config.json'),
    'bert-large-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased', 'bert-large-uncased-config.json'),
    'bert-large-uncased-whole-word-masking': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking', 'bert-large-uncased-whole-word-masking-config.json'),
    'bert-large-uncased-whole-word-masking-finetuned-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking-finetuned-squad-config.json'),
    'roberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-base', 'roberta-base-config.json'),
    'roberta-large': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large', 'roberta-large-config.json'),
    'roberta-large-mnli': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large-mnli', 'roberta-large-mnli-config.json'),
    'distilroberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilroberta-base', 'distilroberta-base-config.json'),
    'prod-bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-bert-base-uncased', 'config.json'),
    'prod-roberta-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-roberta-base-cased', 'config.json'),
    'tune_bert-base-uncased_nsp': path.join(EXTERNAL_EMBEDDINGS_DIR, 'tune_bert-base-uncased_nsp_neg_1', 'config.json'),
    'xlnet-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'xlnet-base-cased', 'xlnet-base-cased-config.json'),
    'albert-base-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-base-v1', 'albert-base-v1-config.json'),
    'albert-large-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-large-v1', 'albert-large-v1-config.json'),
    'albert-xlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xlarge-v1', 'albert-xlarge-v1-config.json'),
    'albert-xxlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xxlarge-v1', 'albert-xxlarge-v1-config.json'),
    'gpt2': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2', 'gpt2-config.json'),
    'gpt2-medium': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2-medium', 'gpt2-medium-config.json'),
    'transfo-xl': path.join(EXTERNAL_EMBEDDINGS_DIR, 'transfo-xl', 'transfo-xl-wt103-config.json'),
    'distilbert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased', 'distilbert-base-uncased-config.json'),
    'distilbert-base-uncased-distilled-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-distilled-squad-config.json'),
}

BERT_MODEL_FILE = {
    'bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-uncased', 'bert-base-uncased-tf_model.h5'),
    'bert-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-cased', 'bert-base-cased-tf_model.h5'),
    'bert-large-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased', 'bert-large-uncased-tf_model.h5'),
    'bert-large-uncased-whole-word-masking': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking', 'bert-large-uncased-whole-word-masking-tf_model.h5'),
    'bert-large-uncased-whole-word-masking-finetuned-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking-finetuned-squad-tf_model.h5'),
    'roberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-base', 'roberta-base-tf_model.h5'),
    'roberta-large': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large', 'roberta-large-tf_model.h5'),
    'roberta-large-mnli': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large-mnli', 'roberta-large-mnli-tf_model.h5'),
    'distilroberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilroberta-base', 'distilroberta-base-tf_model.h5'),
    'prod-bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-bert-base-uncased', 'pytorch_model.bin'),
    'prod-roberta-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-roberta-base-cased', 'pytorch_model.bin'),
    'tune_bert-base-uncased_nsp': path.join(EXTERNAL_EMBEDDINGS_DIR, 'tune_bert-base-uncased_nsp_neg_1', 'pytorch_model.bin'),
    'xlnet-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'xlnet-base-cased', 'xlnet-base-cased-tf_model.h5'),
    'albert-base-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-base-v1', 'albert-base-v1-with-prefix-tf_model.h5'),
    'albert-large-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-large-v1', 'albert-large-v1-with-prefix-tf_model.h5'),
    'albert-xlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xlarge-v1', 'albert-xlarge-v1-with-prefix-tf_model.h5'),
    'albert-xxlarge-v1': path.join(EXTERNAL_EMBEDDINGS_DIR, 'albert-xxlarge-v1', 'albert-xxlarge-v1-with-prefix-tf_model.h5'),
    'gpt2': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2', 'gpt2-tf_model.h5'),
    'gpt2-medium': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2-medium', 'gpt2-medium-tf_model.h5'),
    'transfo-xl': path.join(EXTERNAL_EMBEDDINGS_DIR, 'transfo-xl', 'transfo-xl-wt103-tf_mode.h5'),
    'distilbert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased', 'distilbert-base-uncased-tf_model.h5'),
    'distilbert-base-uncased-distilled-squad': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-distilled-squad-tf_model.h5'),
}

BERT_TORCH_MODEL_FILE = {
    'bert-base-uncased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'bert-base-uncased', 'bert-base-uncased-pytorch_model.bin'),
}

BERT_MERGE_FILE = {
    'roberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-base', 'roberta-base-merges.txt'),
    'roberta-large': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large', 'roberta-large-merges.txt'),
    'roberta-large-mnli': path.join(EXTERNAL_EMBEDDINGS_DIR, 'roberta-large-mnli', 'roberta-large-mnli-merges.txt'),
    'distilroberta-base': path.join(EXTERNAL_EMBEDDINGS_DIR, 'distilroberta-base', 'distilroberta-base-merges.txt'),
    'prod-roberta-base-cased': path.join(EXTERNAL_EMBEDDINGS_DIR, 'prod-roberta-base-cased', 'merges.txt'),
    'gpt2': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2', 'gpt2-merges.txt'),
    'gpt2-medium': path.join(EXTERNAL_EMBEDDINGS_DIR, 'gpt2-medium', 'gpt2-medium-merges.txt'),
}

TEXT_CORPUS_DIR = path.join(RAW_DATA_DIR, 'text_corpus')
TEST_TEXT_COPRUS_DIR = path.join(RAW_DATA_DIR, 'test_text_corpus')

MAX_LEN = {
    'name': 20,
    'desc': 200,
    'name_desc': 200
}


class ModelConfig(object):
    def __init__(self):
        # model general configuration
        self.exp_name = None
        self.model_type = 'bert-base-uncased'

        # input general configuration
        self.input_type = 'name_desc'
        self.use_multi_task = True
        self.use_harl = False
        self.use_hal = False
        self.cate_embed_dim = 100

        self.use_word_input = False
        self.word_embed_trainable = True
        self.word_embed_type = None
        self.word_embeddings = None
        self.word_embed_dim = 300
        self.word_vocab = None
        self.word_vocab_size = -1

        self.use_bert_input = True
        self.bert_trainable = True
        self.use_bert_type = 'pooler'
        self.output_hidden_state = False
        self.n_last_hidden_layer = 0
        self.dense_after_bert = False
        self.use_pair_input = True

        self.max_len = None
        self.cate1_vocab = None
        self.cate2_vocab = None
        self.cate3_vocab = None
        self.all_cate_vocab = None
        self.idx2cate1 = None
        self.idx2cate2 = None
        self.idx2cate3 = None
        self.idx2all_cate = None
        self.n_cate1 = -1
        self.n_cate2 = -1
        self.n_cate3 = -1
        self.n_all_cate = -1

        # model training configuration
        self.share_father_pred = 'no'
        self.use_mask_for_cate2 = False
        self.use_mask_for_cate3 = False
        self.cate3_mask_type = 'cate1'
        self.cate1_to_cate2 = None
        self.cate1_to_cate2 = None
        self.cate2_to_cate3 = None
        self.cate1_to_cate3 = None
        self.cate_to_cate3 = None
        self.cate1_loss_weight = 1.
        self.cate2_loss_weight = 1.
        self.cate3_loss_weight = 1.
        self.cate1_count_dict = None
        self.cate2_count_dict = None
        self.cate3_count_dict = None

        self.batch_size = 32
        self.predict_batch_size = 32
        self.n_epoch = 50
        self.learning_rate = 2e-5
        self.optimizer = 'adam'
        self.use_focal_loss = False
        self.callbacks_to_add = None

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_f1'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stopping configuration
        self.early_stopping_monitor = 'val_f1'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

        # ensembler configuration
        self.swa_start = 10

        # lr scheduler configuration
        self.max_lr = 5e-5
        self.min_lr = 1e-5

        # cross validation
        self.train_on_cv = False
        self.cv_random_state = 42
        self.cv_fold = 5
        self.cv_index = 0

        # data augmentation
        self.exchange_pair = False
        self.exchange_threshold = 0.1

        self.use_pseudo_label = False
        self.pseudo_path = None
        self.pseudo_random_state = 42
        self.pseudo_rate = 0.1
        self.pseudo_index = 0


class LanguageModelConfig:
    def __init__(self):
        self.model_name = None

        self.fine_tune = True
        self.model_type = 'bert-base-uncased'

        self.do_lm = True  # train with language modeling
        self.do_mlm = True  # train with masked language modeling
        self.lm_with_cls_corpus = True  # other than product corpus, also use product classification dataset for LM

        self.do_nsp = False  # train next sentence prediction with product classification dataset
        self.num_neg_sample = 1

        self.model_save_dir = None

        self.tokenizer_type = 'word_piece'
        self.lowercase = True
        self.vocab_size = 30000

