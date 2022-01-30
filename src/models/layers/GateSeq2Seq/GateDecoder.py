import torch.nn as nn
import torch
import torch.nn.functional as F

from src.models.layers.Attention import Attention
from src.utils.const import MAX_PADDING, PAD_VALUE, UNK_VALUE, BOS_WORD_VALUE, EOS_WORD_VALUE
from src.utils.weight_init import apply_emb_weight


class GateDecoder(nn.Module):
    def __init__(self, num_category, cate_dim, hidden_dim, n_layers, dropout, hidden_type):
        super(GateDecoder, self).__init__()
        self.num_category = num_category
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_category, cate_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.rnn = nn.GRU(cate_dim, hidden_dim, self.n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_category)
        nn.init.xavier_uniform_(self.fc_out.weight)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.hidden_type = hidden_type

    def get_hidden(self, hidden, asp_vector):
        if self.hidden_type == 'gate':
            return self.get_input_gate(hidden.squeeze(0), asp_vector)
        if self.hidden_type == 'cat':
            return self.get_input_cat(hidden.squeeze(0), asp_vector)
        if self.hidden_type == 'asp':
            return asp_vector.unsqueeze(0)
        if self.hidden_type == 'hidden':
            return hidden

    def get_input_gate(self, hidden, asp_vector):
        g1 = torch.sigmoid(self.linear1(hidden) + self.linear2(asp_vector))
        hidden = g1 * hidden + (1 - g1) * asp_vector
        return hidden

    def get_input_cat(self, hidden, asp_vector):
        hidden = self.linear1(hidden) + self.linear2(asp_vector)
        return hidden

    def forward(self, input, hidden, asp_vector):
        input_emb = self.embedding(input)
        output, hidden = self.rnn(input_emb.unsqueeze(1), hidden)
        vec = self.get_hidden(hidden[-1], asp_vector)
        prediction = self.fc_out(vec)
        return prediction, hidden
