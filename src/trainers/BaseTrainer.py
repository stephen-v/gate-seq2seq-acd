import json
import os

import pickle
import numpy as np
from monai.data import worker_init_fn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.ReviewDataset import ReviewDataset
from src.models.BaseModel import BaseModel
from src.utils.const import MAX_PADDING, CATE_MAX_PADDING, EARLY_STOPPING, ModelName
from src.utils.file_utils import train_data_path, test_data_path, valid_data_path, output_path, word_embedding_path, \
    word2id_path, cate2id_path, cate_embedding_path
from src.utils.metrics import compute_f1, compute_precision, compute_recall
import torch

from src.utils.text_util import read_config


class BaseTrainer(object):

    def __init__(self, args):
        self.args = args
        self.data_source = args.data_source
        self.data_category = args.data_category
        self.train_data = pickle.load(open(train_data_path(self.data_source, self.data_category), 'rb'))
        self.test_data = pickle.load(open(test_data_path(self.data_source, self.data_category), 'rb'))
        self.valid_data = pickle.load(open(valid_data_path(self.data_source, self.data_category), 'rb'))
        self.word2id = pickle.load(open(word2id_path(self.data_source, self.data_category), 'rb'))
        self.cate2id = pickle.load(open(cate2id_path(self.data_source, self.data_category), 'rb'))

    def init_model_settings(self):
        self.model_settings = read_config('model.config.yaml').get(self.args.model_name.name, {})
        self.model_settings['device'] = self.args.device
        self.model_settings['num_words'] = len(self.word2id)
        self.model_settings['num_category'] = len(self.cate2id)

    def init_train_setting(self):
        self.train_settings = {}
        self.train_settings['lr'] = self.args.lr
        self.train_settings['l2'] = self.args.l2
        self.train_settings['batch_size'] = self.args.batch_size
        self.train_settings['epochs'] = self.args.epochs
        self.train_settings['interval'] = self.args.interval
        self.train_settings['device'] = self.args.device
        self.train_settings['num_workers'] = self.args.num_workers
        self.train_settings['model_name'] = self.args.model_name.name
        self.train_settings['data_source'] = self.args.data_source
        self.train_settings['data_category'] = self.args.data_category

    def load_dataset(self, data):
        pass

    def load_embeddings(self):
        embeddings = pickle.load(
            open(word_embedding_path(self.data_source, self.model_settings['word_dim'], self.data_category), 'rb'))
        return embeddings

    def get_data_loader(self, dataset, batch_size, num_workers):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          worker_init_fn=worker_init_fn)

    def create_model(self):
        self.init_train_setting()
        self.init_model_settings()
        self.print_settings(self.model_settings, 'model_settings')
        self.print_settings(self.train_settings, 'train_setting')
        return BaseModel(self.model_settings)

    def train(self):
        model = self.create_model()
        train_dataset = ReviewDataset(self.train_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        valid_dataset = ReviewDataset(self.valid_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        train_loader = self.get_data_loader(train_dataset, self.train_settings['batch_size'],
                                            self.train_settings['num_workers'])
        optimizer = AdamW(model.parameters(), lr=self.train_settings['lr'])
        best_f1, f1_score = 0, 0
        patient = 0
        for epoch in range(self.train_settings['epochs']):
            loss_total = 0
            with tqdm(ncols=200, total=len(train_dataset), desc='start to new epoch......') as pbar:
                for counter, batch_data in enumerate(train_loader):
                    optimizer.zero_grad()
                    model.train()
                    pbar.update(len(batch_data[0]))
                    loss = model(batch_data)
                    loss_total += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    if (counter + 1) % self.train_settings['interval'] == 0:
                        f1_score, _, _, _ = self._eval_(model, valid_dataset)
                        if f1_score > best_f1:
                            best_f1 = f1_score
                            self.save_model(model)
                            patient = 0
                        else:
                            patient += 1
                        pbar.set_description(
                            'epoch:{},loss:{},f1:{},best_f1:{}'.format(epoch, loss_total / counter, f1_score, best_f1))
            if patient >= EARLY_STOPPING:
                break

    def eval(self):
        model = self.load_model()
        test_dataset = ReviewDataset(self.test_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        f1, p, r, bad_cases = self._eval_(model, test_dataset)
        print('model:{} , f1 score:{} , p:{} ,r:{}'.format(model.get_model_name(), f1, p, r))

        test_data = [(idx, review, review_token, categories, single_label) for
                     (idx, review, review_token, categories, single_label) in self.test_data if single_label == 1]
        test_dataset = ReviewDataset(test_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        f1_1, p_1, r_1, bad_cases_1 = self._eval_(model, test_dataset)
        print(
            'model:{} , only have single label sentence, f1 score:{} , p:{} ,r:{}'.format(model.get_model_name(), f1_1,
                                                                                          p_1, r_1))

        test_data = [(idx, review, review_token, categories, single_label) for
                     (idx, review, review_token, categories, single_label) in self.test_data if single_label == 0]
        test_dataset = ReviewDataset(test_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        f1_2, p_2, r_2, bad_cases_2 = self._eval_(model, test_dataset)
        print('model:{} , only have multiple labels sentence, f1 score:{} , p:{} ,r:{}'.format(model.get_model_name(),
                                                                                               f1_2, p_2, r_2))

        self.__print_bad_cases__(bad_cases_1)
        self.__print_bad_cases__(bad_cases_2)

        return f1

    def __print_bad_cases__(self, bad_cases):
        print('----------------------')
        word2id = pickle.load(open(word2id_path(self.data_source, self.data_category), 'rb'))
        cate2id = pickle.load(open(cate2id_path(self.data_source, self.data_category), 'rb'))
        id2word = {v: k for k, v in word2id.items()}
        id2cate = {v: k for k, v in cate2id.items()}
        for line in bad_cases:
            review, real, predict, real_asp_num_label, predict_asp_num_label = line
            review_text = ' '.join([id2word[r.item()] for r in review.squeeze(0) if r.item() > 0])
            real_cate = ' '.join([id2cate[i + 4] for i, r in enumerate(real) if r > 0])
            predict_cate = ' '.join([id2cate[i + 4] for i, r in enumerate(predict) if r > 0])
            print('{},【{}】,【{}】,【{}】，【{}】'.format(review_text, real_cate, predict_cate, real_asp_num_label,
                                                  predict_asp_num_label))

    def _eval_(self, model, dataset):
        model = model.eval()
        data_loader = self.get_data_loader(dataset, batch_size=1, num_workers=0)
        predict_labels = []
        real_labels = []
        bad_cases = []
        for batch_data in data_loader:
            review, real_label, real_asp_num_label = model.assemble_tensor(batch_data)
            predict_label = model.make_label(model.predict(review))
            real_label = model.make_label(real_label[0].tolist())
            real_labels.append(real_label)
            predict_labels.append(predict_label)
            if not (real_label == predict_label).all():
                if self.train_settings['model_name'] == 'GSEQ':
                    predict_asp_num_label = model.predict_aspect_num(review)
                    bad_cases.append(
                        [review, real_label, predict_label, real_asp_num_label.item(), predict_asp_num_label.item()])
                else:
                    predict_asp_num_label = 'None'
                    bad_cases.append(
                        [review, real_label, predict_label, real_asp_num_label.item(), predict_asp_num_label])

        average = 'micro'
        real_labels = np.array(real_labels)
        predict_labels = np.array(predict_labels)
        f1_scores = compute_f1(real_labels, predict_labels, average=average)
        recall = compute_recall(real_labels, predict_labels, average=average)
        precision = compute_precision(real_labels, predict_labels, average=average)
        return f1_scores, precision, recall, bad_cases

    def predict(self):
        pass

    def save_model(self, model):
        out_path = os.path.join(output_path(self.data_source),
                                '{}_{}'.format(self.data_category, model.get_model_name()))
        torch.save(model.state_dict(), out_path)

    def load_model(self):
        model = self.create_model()
        out_path = os.path.join(output_path(self.data_source),
                                '{}_{}'.format(self.data_category, model.get_model_name()))
        model.load_state_dict(torch.load(out_path, map_location=model.device))
        return model

    def print_settings(self, settings, setting_name):
        print('======================================{}==================================================='.format(
            setting_name))
        data = json.dumps(settings, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
        print(data)
        print('======================================{}==================================================='.format(
            setting_name))

    def run(self, mode):
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        else:
            self.predict()
