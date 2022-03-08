import torch.nn as nn
import torch
from random import random

from src.models.BaseModel import BaseModel
from src.models.layers.Seq2Seq.Decoder import Decoder
from src.models.layers.Seq2Seq.RNNEncoder import RNNEncoder
from src.utils.const import BOS_WORD_VALUE, EOS_WORD_VALUE, CATE_MAX_PADDING, PAD_VALUE


class Seq2Seq(BaseModel):
    def __init__(self, model_settings, word_embeddings_weight):
        super().__init__(model_settings)
        hidden_dim = model_settings['hidden_dim']
        num_words = model_settings['num_words']
        word_dim = model_settings['word_dim']
        n_layers = model_settings['n_layers']
        num_category = model_settings['num_category']
        cate_dim = model_settings['cate_dim']
        dropout = model_settings['dropout']
        self.teacher_forcing_ratio = model_settings['teacher_forcing_ratio']
        self.encoder = RNNEncoder(hidden_dim, num_words, word_dim, n_layers, dropout, word_embeddings_weight,
                                  self.device)
        self.decoder = Decoder(num_category, cate_dim, hidden_dim, n_layers, dropout)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input):
        review, aspect_label, _ = self.assemble_tensor(input)
        batch_size = review.shape[0]
        hidden, encoder_outputs = self.encoder(review)
        input = torch.tensor([BOS_WORD_VALUE] * batch_size).to(self.device)
        mask, num_not_pad_tokens = torch.ones(batch_size, ).to(self.device), 0
        l = torch.tensor([0.0]).to(self.device)
        for y in aspect_label.permute(1, 0):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            teacher_force = random() < self.teacher_forcing_ratio
            top1 = output.argmax(1)
            input = y if teacher_force else top1
            l = l + (mask * self.criterion(output, y)).sum()
            num_not_pad_tokens += mask.sum().item()
            mask = mask * (y != PAD_VALUE).float()
        l = l / num_not_pad_tokens
        return l

    def predict(self, review):
        batch_size = review.shape[0]
        hidden, encoder_outputs = self.encoder(review)
        input = torch.tensor([BOS_WORD_VALUE] * batch_size).to(self.device)
        outputs = []
        for i in range(CATE_MAX_PADDING):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1)
            input = top1
            if i == 0 and top1.item() <= 3:
                top1 = torch.topk(output, k=2)[1][0][-1]
                outputs.append(top1.item())
                break
            if top1.item() == EOS_WORD_VALUE:
                break
            outputs.append(top1.item())
        return outputs

    def get_model_name(self):
        return 'Seq2Seq'
