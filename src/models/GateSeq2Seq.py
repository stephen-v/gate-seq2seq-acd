import torch.nn as nn
import torch
from random import random

from src.models.BaseModel import BaseModel
from src.models.layers.GateSeq2Seq.GateDecoder import GateDecoder
from src.models.layers.GateSeq2Seq.SelfAttentiveEncoder import SelfAttentiveEncoder
from src.models.layers.Seq2Seq.Decoder import Decoder
from src.models.layers.Seq2Seq.RNNEncoder import RNNEncoder
from src.utils.const import BOS_WORD_VALUE, EOS_WORD_VALUE, CATE_MAX_PADDING, PAD_VALUE, UNK_VALUE


class GateSeq2Seq(BaseModel):
    def __init__(self, model_settings, word_embeddings_weight):
        super().__init__(model_settings)
        hidden_dim = model_settings['hidden_dim']
        num_words = model_settings['num_words']
        word_dim = model_settings['word_dim']
        n_layers = model_settings['n_layers']
        num_category = model_settings['num_category']
        cate_dim = model_settings['cate_dim']
        dropout = model_settings['dropout']
        attn_hops = model_settings['attn_hops']
        mlp_d = model_settings['mlp_d']
        hidden_type = model_settings['hidden_type']
        self.att = model_settings['att']
        self.loss_weight = model_settings['loss_weight']
        self.teacher_forcing_ratio = model_settings['teacher_forcing_ratio']
        self.encoder = SelfAttentiveEncoder(hidden_dim, attn_hops, mlp_d, n_layers, num_words, word_dim,
                                            word_embeddings_weight, self.device, dropout)
        self.decoder = GateDecoder(num_category, cate_dim, hidden_dim, n_layers, dropout, hidden_type, self.att)
        self.asp_encoder = nn.Linear(hidden_dim * attn_hops * 2 + hidden_dim, hidden_dim)
        self.asp_fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.asp_encoder.weight)
        nn.init.xavier_uniform_(self.asp_fc.weight)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion1 = nn.BCELoss()
        self.hidden_type = model_settings['hidden_type']

    def forward(self, input):
        review, aspect_label, aspect_num_label = self.assemble_tensor(input)
        batch_size = review.shape[0]
        M, hidden, encoder_outputs, length_emb = self.encoder(review)
        asp_vector = self.asp_encoder(torch.cat([M.view(batch_size, -1), length_emb], dim=1).flatten(1, -1))
        asp_logit = torch.sigmoid(self.asp_fc(asp_vector))
        input = torch.tensor([BOS_WORD_VALUE] * batch_size).to(self.device)
        mask, num_not_pad_tokens = torch.ones(batch_size, ).to(self.device), 0
        l = torch.tensor([0.0]).to(self.device)
        for y in aspect_label.permute(1, 0):
            output, hidden = self.decoder(input, hidden, asp_vector, encoder_outputs)
            teacher_force = random() < self.teacher_forcing_ratio
            top1 = output.argmax(1)
            input = y if teacher_force else top1
            l = l + (mask * self.criterion(output, y)).sum()
            num_not_pad_tokens += mask.sum().item()
            mask = mask * (y != EOS_WORD_VALUE).float()
        l = l / num_not_pad_tokens
        asp_l = self.criterion1(asp_logit.squeeze(1), aspect_num_label)
        loss = self.loss_weight * l + (1 - self.loss_weight) * asp_l
        return loss

    def predict(self, review):
        batch_size = review.shape[0]
        M, hidden, encoder_outputs, length_emb = self.encoder(review)
        asp_vector = self.asp_encoder(torch.cat([M.view(batch_size, -1), length_emb], dim=1).flatten(1, -1))
        input = torch.tensor([BOS_WORD_VALUE] * batch_size).to(self.device)
        outputs = []
        for i in range(CATE_MAX_PADDING):
            output, hidden = self.decoder(input, hidden, asp_vector, encoder_outputs)
            top1 = output.argmax(1)
            if i == 0 and top1.item() <= 3:
                top1 = torch.topk(output, k=2)[1][0][-1]
                outputs.append(top1.item())
                break
            if top1.item() == EOS_WORD_VALUE:
                break
            input = top1
            outputs.append(top1.item())
        return outputs

    def predict_aspect_num(self, review):
        M, hidden, encoder_outputs, length_emb = self.encoder(review)
        asp_vector = self.asp_encoder(torch.cat([M.flatten(1, -1), length_emb], dim=1).flatten(1, -1))
        logit = torch.sigmoid(self.asp_fc(asp_vector))
        return logit

    def get_model_name(self):
        return 'GateSeq2Seq_{}_{}.pkl'.format(self.hidden_type, self.att)
