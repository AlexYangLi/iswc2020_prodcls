# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: transformers.py

@time: 2020/5/23 20:34

@desc:

"""

from transformers import BertTokenizerFast, RobertaTokenizerFast, XLNetTokenizer, AlbertTokenizer, GPT2TokenizerFast, \
    TransfoXLTokenizerFast, DistilBertTokenizerFast
from transformers import BertConfig, RobertaConfig, XLNetConfig, AlbertConfig, GPT2Config, TransfoXLConfig, \
    DistilBertConfig
from transformers import TFBertModel, TFRobertaModel, TFXLNetModel, TFAlbertModel, TFGPT2Model, TFTransfoXLModel, \
    TFDistilBertModel

from config import BERT_VOCAB_FILE, BERT_MERGE_FILE, BERT_CONFIG_FILE, BERT_MODEL_FILE


def get_bert_tokenizer(bert_model_type):
    if bert_model_type in ['bert-base-uncased', 'prod-bert-base-uncased', 'bert-base-cased', 'bert-large-uncased',
                           'tune_bert-base-uncased_nsp', 'bert-large-uncased-whole-word-masking',
                           'bert-large-uncased-whole-word-masking-finetuned-squad']:
        if '-cased' in bert_model_type:
            do_lower_case = False
        else:
            do_lower_case = True  # default
        return BertTokenizerFast(vocab_file=BERT_VOCAB_FILE[bert_model_type], do_lower_case=do_lower_case)
    elif bert_model_type in ['roberta-base', 'prod-roberta-base-cased', 'roberta-large', 'roberta-large-mnli',
                             'distilroberta-base']:
        return RobertaTokenizerFast(vocab_file=BERT_VOCAB_FILE[bert_model_type],
                                    merges_file=BERT_MERGE_FILE[bert_model_type],
                                    add_prefix_space=True)
    elif bert_model_type in ['xlnet-base-cased']:
        if '-uncased' in bert_model_type:
            do_lower_case = True
        else:
            do_lower_case = False  # default
        return XLNetTokenizer(vocab_file=BERT_VOCAB_FILE[bert_model_type], do_lower_case=do_lower_case)
    elif bert_model_type in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1']:
        return AlbertTokenizer(vocab_file=BERT_VOCAB_FILE[bert_model_type])
    elif bert_model_type in ['gpt2', 'gpt2-medium']:
        tokenizer = GPT2TokenizerFast(vocab_file=BERT_VOCAB_FILE[bert_model_type],
                                      merges_file=BERT_MERGE_FILE[bert_model_type],
                                      add_prefix_space=True)
        # https://github.com/huggingface/transformers/issues/3859
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif bert_model_type in ['transfo-xl']:
        return TransfoXLTokenizerFast(vocab_file=BERT_VOCAB_FILE[bert_model_type])
    elif bert_model_type in ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad']:
        if '-cased' in bert_model_type:
            do_lower_case = False
        else:
            do_lower_case = True  # default
        return DistilBertTokenizerFast(vocab_file=BERT_VOCAB_FILE[bert_model_type], do_lower_case=do_lower_case)
    else:
        raise ValueError(f'`bert_model_type` not understood: {bert_model_type}')


def get_bert_config(bert_model_type, output_hidden_states=False):
    if bert_model_type in ['bert-base-uncased', 'prod-bert-base-uncased', 'bert-base-cased', 'bert-large-uncased',
                           'tune_bert-base-uncased_nsp', 'bert-large-uncased-whole-word-masking',
                           'bert-large-uncased-whole-word-masking-finetuned-squad']:
        bert_config = BertConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['roberta-base', 'prod-roberta-base-cased', 'roberta-large', 'roberta-large-mnli',
                             'distilroberta-base']:
        bert_config = RobertaConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['xlnet-base-cased']:
        bert_config = XLNetConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1']:
        bert_config = AlbertConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['gpt2', 'gpt2-medium']:
        bert_config = GPT2Config.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['transfo-xl']:
        bert_config = TransfoXLConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    elif bert_model_type in ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad']:
        bert_config = DistilBertConfig.from_pretrained(BERT_CONFIG_FILE[bert_model_type])
    else:
        raise ValueError(f'`bert_model_type` not understood: {bert_model_type}')

    bert_config.output_hidden_states = output_hidden_states
    return bert_config


def get_transformer(bert_model_type, output_hidden_states=False):
    config = get_bert_config(bert_model_type, output_hidden_states)
    if bert_model_type in ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased',
                           'bert-large-uncased-whole-word-masking',
                           'bert-large-uncased-whole-word-masking-finetuned-squad']:
        return TFBertModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['prod-bert-base-uncased', 'tune_bert-base-uncased_nsp']:
        return TFBertModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config, from_pt=True)
    elif bert_model_type in ['roberta-base', 'roberta-large', 'roberta-large-mnli', 'distilroberta-base']:
        return TFRobertaModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['prod-roberta-base-cased']:
        return TFRobertaModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config, from_pt=True)
    elif bert_model_type in ['xlnet-base-cased']:
        return TFXLNetModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1']:
        return TFAlbertModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['gpt2', 'gpt2-medium']:
        return TFGPT2Model.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['transfo-xl']:
        return TFTransfoXLModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    elif bert_model_type in ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad']:
        return TFDistilBertModel.from_pretrained(BERT_MODEL_FILE[bert_model_type], config=config)
    else:
        raise ValueError(f'`bert_model_type` not understood: {bert_model_type}')
