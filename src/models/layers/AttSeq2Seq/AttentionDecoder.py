from src.models.layers.Attention import Attention
from src.models.layers.Seq2Seq.Decoder import Decoder
import torch
import torch.nn as nn


class AttentionDecoder(Decoder):
    def __init__(self, num_category, emb_dim, hidden_dim, n_layers, dropout):
        super(AttentionDecoder, self).__init__(num_category, emb_dim, hidden_dim, n_layers, dropout)
        self.rnn = nn.GRU(emb_dim * 2, hidden_dim, n_layers, batch_first=True)
        self.attention = Attention(hidden_dim, hidden_dim)

    def forward(self, input, hidden, encoder_outputs):
        att = self.attention(hidden.squeeze(0), encoder_outputs)
        att_emb = torch.bmm(att.unsqueeze(1), encoder_outputs).squeeze(1)
        input = input.unsqueeze(-1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(torch.cat([embedded, att_emb.unsqueeze(1)], dim=-1), hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden
