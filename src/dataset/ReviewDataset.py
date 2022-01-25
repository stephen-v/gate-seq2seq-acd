import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils.const import UNK_NAME, UNK_VALUE
from src.utils.file_utils import word2id_path
from src.utils.text_util import padding_cate, padding


class ReviewDataset(Dataset):
    def __init__(self, data, word2id, cate2id, padding_nums, cate_padding_nums):
        super(ReviewDataset, self).__init__()
        self.data = data
        self.word2id = word2id
        self.cate2id = cate2id
        self.padding_nums = padding_nums
        self.cate_padding_nums = cate_padding_nums

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        idx, review, review_token, categories, single_label = self.data[index]
        review = torch.tensor(padding(review_token, self.word2id, self.padding_nums))
        categories_label = torch.tensor(padding_cate(categories, self.cate2id, self.cate_padding_nums)).type(torch.LongTensor)
        single_label = torch.tensor(single_label).type(torch.FloatTensor)
        X = [review, categories_label, single_label]
        return X
