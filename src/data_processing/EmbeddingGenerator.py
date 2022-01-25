import os
import numpy as np
import pickle

from src.utils.file_utils import word_embedding_path, yelp_pretrain_path, amazon_pretrain_path, cate_embedding_path


class EmbeddingGenerator(object):
    def __init__(self, data_source, word_dim, data_category):
        self.word_dim = word_dim
        if data_category == 'Restaurants':
            self.pretrain_embedding_path = os.path.join(yelp_pretrain_path(), 'glove.yelp.{}d.txt'.format(word_dim))
        else:
            self.pretrain_embedding_path = os.path.join(amazon_pretrain_path(), 'reviews_embeddings.txt')
        self.processed_pretrain_embedding_path = word_embedding_path(data_source, word_dim, data_category)
        self.processed_pretrain_cate_embedding_path = cate_embedding_path(data_source, word_dim, data_category)

    def generate_embedding(self, word2id):
        print('start to process word embedding')
        embedding_dir = {}
        f = open(self.pretrain_embedding_path, encoding='utf-8')
        for i, line in enumerate(f):
            try:
                values = line.split()
                if len(values) < 10:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_dir[word] = coefs
            except Exception as e:
                print(e)
        f.close()
        total = len(embedding_dir)
        print('uniform_init...')
        rng = np.random.RandomState(1)
        a = 0.25
        embedding_matrix = rng.uniform(-a, a, size=(len(word2id), self.word_dim))
        find_word2vec = 0
        for i, word in enumerate(word2id):
            if word in embedding_dir:
                embedding_vector = embedding_dir[word]
                embedding_matrix[i] = embedding_vector
                find_word2vec += 1
            # else:
            #     print(word)
        print('pretrain word embedding has {} words'.format(total))
        print('find words {} '.format(find_word2vec))
        print('embeddings的shape为： {}'.format(np.shape(embedding_matrix)))
        pickle.dump(embedding_matrix, open(self.processed_pretrain_embedding_path, 'wb'))
        return embedding_matrix
