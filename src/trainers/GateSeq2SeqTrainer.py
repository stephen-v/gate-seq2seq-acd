import sys

from torch.optim import Adam
from tqdm import tqdm

from src.dataset.ReviewDataset import ReviewDataset
from src.models.GateSeq2Seq import GateSeq2Seq
from src.models.Seq2Seq import Seq2Seq
from src.trainers.BaseTrainer import BaseTrainer
from src.utils.const import MAX_PADDING, CATE_MAX_PADDING, EARLY_STOPPING
from src.utils.metrics import compute_f1, compute_precision, compute_recall
import numpy as np


class GateSeq2SeqTrainer(BaseTrainer):
    def __init__(self, args):
        super(GateSeq2SeqTrainer, self).__init__(args)

    def create_model(self):
        super(GateSeq2SeqTrainer, self).create_model()
        word_embeddings_weight = self.load_embeddings()
        model = GateSeq2Seq(self.model_settings, word_embeddings_weight)
        return model.to(model.device)

    def init_model_settings(self):
        super(GateSeq2SeqTrainer, self).init_model_settings()
        self.model_settings['teacher_forcing_ratio'] = self.args.teacher_forcing_ratio
        self.model_settings['attn_hops'] = self.args.attn_hops
        self.model_settings['mlp_d'] = self.args.mlp_d
        self.model_settings['loss_weight'] = self.args.loss_weight
        self.model_settings['hidden_type'] = self.args.hidden_type
        self.model_settings['att'] = self.args.att

    def predict(self):
        test_dataset = ReviewDataset(self.test_data, self.word2id, self.cate2id, MAX_PADDING, CATE_MAX_PADDING)
        model = self.load_model()
        model = model.eval()
        data_loader = self.get_data_loader(test_dataset, batch_size=1, num_workers=0)
        predict_labels = []
        real_labels = []
        for batch_data in data_loader:
            review, real_label, label = model.assemble_tensor(batch_data)
            # predict_label = model.predict(review, real_label)
            predict_label = 1.0 if model.predict_aspect_num(review).item() > 0.5 else 0.0
            real_label = label.item()
            predict_labels.append(predict_label)
            real_labels.append(real_label)
        f1_scores = compute_f1(real_labels, predict_labels, average='binary')
        precision_scores = compute_precision(real_labels, predict_labels, average='binary')
        recall_scores = compute_recall(real_labels, predict_labels, average='binary')
        print('f1:{},precision:{},recall:{}'.format(f1_scores, precision_scores, recall_scores))


