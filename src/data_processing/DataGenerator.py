import os.path
import pickle
from xml.dom.minidom import parse
import xml.dom.minidom
import nltk
from nltk.corpus import stopwords
import numpy as np
from src.utils.const import PAD_NAME, PAD_VALUE, UNK_NAME, UNK_VALUE, BOS_WORD_VALUE, BOS_WORD, EOS_WORD_VALUE, EOS_WORD
from src.utils.file_utils import train_data_path, valid_data_path, test_data_path, word2id_path, cate2id_path, raw_path

stopwords = set(stopwords.words('english'))


class DataGenerator(object):
    def __init__(self, data_source, data_category):
        self.data_source = data_source
        self.data_category = data_category
        self.raw_path = raw_path(data_source)

    def _parser_xml_(self, path, word2id, cate2id):
        DOMTree = xml.dom.minidom.parse(path)
        collection = DOMTree.documentElement
        sentences = collection.getElementsByTagName('sentence')
        data = []
        # word_tokenizer = nltk.TreebankWordTokenizer()
        word_tokenizer = nltk.TweetTokenizer(preserve_case=False)
        max_word, max_cates = 0, 0
        not_have_aspect_nums = 0
        for idx, sentence in enumerate(sentences):
            cate_list = []
            text = sentence.getElementsByTagName('text')
            text = text[0].childNodes[0].data.lower()
            text_tokens = []

            # aspect category
            cates = sentence.getElementsByTagName('Opinion')
            # word2id
            for word in word_tokenizer.tokenize(text):
                # if word in stopwords:
                #     continue
                if word not in word2id:
                    word2id[word] = len(word2id)
                text_tokens.append(word)
            if len(text_tokens) > max_word:
                max_word = len(text_tokens)
            # terms = sentence.getElementsByTagName('aspectTerm')
            for cate in cates:
                cate_str = cate.getAttribute('category')
                # the category only in train set will discard.
                if cate_str not in cate_list:
                    cate_list.append(cate_str)
                if cate_str not in cate2id:
                    cate2id[cate_str] = len(cate2id)

            if len(cate_list) == 0:
                not_have_aspect_nums += 1
                continue

            if len(cate_list) > max_cates:
                max_cates = len(cate_list)
            cate_list.append(EOS_WORD)

            single_label = 1 if len(cate_list) == 2 else 0
            data.append((idx, text, text_tokens, cate_list, single_label))
        single_label_nums = len([line for line in data if line[-1] == 1])
        print('{}-{}: total_data:{},data :{} , max word nums :{} ,max category nums:{} , single lable radio:{}'.
              format(self.data_source, path, idx + 1, len(data), max_word, max_cates, single_label_nums / len(data)))
        print('there are {} data have any aspect'.format(not_have_aspect_nums))
        return data, max_word, max_cates, word2id, cate2id

    def generate_data(self):
        word2id = {PAD_NAME: PAD_VALUE, UNK_NAME: UNK_VALUE}
        cate2id = {PAD_NAME: PAD_VALUE, UNK_NAME: UNK_VALUE, BOS_WORD: BOS_WORD_VALUE, EOS_WORD: EOS_WORD_VALUE}

        train_data, max_word, max_cates, word2id, cate2id = self._parser_xml_(
            os.path.join(self.raw_path, '{}_Train.xml'.format(self.data_category)),
            word2id, cate2id)

        test_data, max_word, max_cates, word2id, cate2id = self._parser_xml_(
            os.path.join(self.raw_path, '{}_Test.xml'.format(self.data_category)),
            word2id, cate2id)

        print('there are {} words and {} categories'.format(len(word2id) - 2, len(cate2id) - 4))
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        total = len(train_data)
        valid_radio = 0.1
        valid_data = train_data[:int(total * valid_radio)]
        train_data = train_data[int(total * valid_radio):]
        pickle.dump(train_data, open(train_data_path(self.data_source, self.data_category), 'wb'))
        pickle.dump(valid_data, open(valid_data_path(self.data_source, self.data_category), 'wb'))
        pickle.dump(test_data, open(test_data_path(self.data_source, self.data_category), 'wb'))
        pickle.dump(word2id, open(word2id_path(self.data_source, self.data_category), 'wb'))
        pickle.dump(cate2id, open(cate2id_path(self.data_source, self.data_category), 'wb'))
        return word2id, cate2id

