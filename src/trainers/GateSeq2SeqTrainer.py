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
