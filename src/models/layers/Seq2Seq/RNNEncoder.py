import torch.nn as nn
import torch

from src.utils.weight_init import apply_emb_weight
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RNNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_words, word_dim, n_layers, dropout, word_embeddings_weight, device):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_words, word_dim)
        self.embedding = apply_emb_weight(self.embedding, word_embeddings_weight, device)
        self.rnn = nn.GRU(word_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        length = torch.sum((inputs > 0).float(), dim=1).cpu()
        embedded = self.dropout(self.embedding(inputs))
        embedded = pack(embedded, length, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(embedded)
        outputs = unpack(outputs, batch_first=True)[0]
        return hidden, outputs
