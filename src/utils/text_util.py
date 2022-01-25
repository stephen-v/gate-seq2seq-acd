import pickle
import re

import json
import nltk
import os

from cachetools.func import lru_cache
from nltk.corpus import stopwords
import nltk
import yaml
from scipy.io.idl import AttrDict
import numpy as np
import jieba_fast as jieba
from tqdm import tqdm

from src.utils.const import UNK_NAME, PAD_NAME, PAD_VALUE, UNK_VALUE

tokenizer = nltk.TweetTokenizer(preserve_case=False)


def read_config(config_name):
    path = os.getcwd()
    path = os.path.join(path, 'conf', config_name)
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))


def _padding(sent, sequence_len):
    """
     convert sentence to index array
    """
    if len(sent) > sequence_len:
        sent = sent[:sequence_len]
    padding = sequence_len - len(sent)
    sent2idx = sent + [0] * padding
    return sent2idx, len(sent)


def padding(text, word2id, max_len):
    text_ids = [word2id.get(w, word2id[UNK_NAME]) for w in text]
    text_padding = _padding(text_ids, max_len)[0]
    return text_padding


def padding_cate(cate, cate2id, max_len):
    text_ids = [cate2id[w] for w in cate]
    text_padding = _padding(text_ids, max_len)[0]
    return text_padding
