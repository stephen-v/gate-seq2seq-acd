import os

from src.utils.text_util import read_config

data_configs = read_config('sys.config.yaml')


def processed_path(data_source):
    path = data_configs['data_config'][data_source]['processed']
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def output_path(data_source):
    path = data_configs['data_config'][data_source]['output']
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def yelp_pretrain_path():
    return data_configs['data_config']['glove_pretrain_yelp']


def amazon_pretrain_path():
    return data_configs['data_config']['glove_pretrain_amazon']


def raw_path(data_source):
    return data_configs['data_config'][data_source]['raw']


def word_embedding_path(data_source, word_dim, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_embedding_{}.pkl'.format(data_category, word_dim))


def cate_embedding_path(data_source, word_dim, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_cate_embedding_{}.pkl'.format(data_category, word_dim))


def train_data_path(data_source, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_train.pkl'.format(data_category))


def valid_data_path(data_source, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_valid.pkl'.format(data_category))


def test_data_path(data_source, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_test.pkl'.format(data_category))


def word2id_path(data_source, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_word2id.pkl'.format(data_category))


def cate2id_path(data_source, data_category):
    path = processed_path(data_source)
    return os.path.join(path, '{}_cate2id.pkl'.format(data_category))
