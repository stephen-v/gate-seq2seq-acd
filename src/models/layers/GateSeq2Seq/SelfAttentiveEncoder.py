import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from src.utils.weight_init import apply_emb_weight


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, hidden_dim, attn_hops, mlp_d, n_layers, num_words, word_dim, word_embeddings_weight, device, dropout):
        super(SelfAttentiveEncoder, self).__init__()
        r = attn_hops
        d = mlp_d
        self.r = r
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_words, word_dim)
        self.embedding = apply_emb_weight(self.embedding, word_embeddings_weight, device)
        self.rnn = nn.GRU(word_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.length_embedding = nn.Embedding(80, word_dim)
        self.length_embedding = apply_emb_weight(self.length_embedding, None, device)
        self.Ws1 = nn.Parameter(torch.Tensor(1, d, hidden_dim * 2))
        self.Ws2 = nn.Parameter(torch.Tensor(1, r, d))
        nn.init.xavier_uniform_(self.Ws2)
        nn.init.xavier_uniform_(self.Ws1)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.hidden_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.xavier_uniform_(self.hidden_fc.weight)

    def forward(self, inputs):
        bs = len(inputs)
        emb = self.dropout(self.embedding(inputs))
        length = torch.sum((inputs > 0).long(), dim=1).cpu()
        emb = pack(emb, length, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(emb)  # (bs, n, 2u)
        outputs = unpack(outputs, batch_first=True)[0]
        length_emb = self.dropout(self.length_embedding(length.to(self.device)))
        n = outputs.size(1)  # sentence length
        H_T = torch.transpose(outputs, 2, 1).contiguous()  # (bs, 2u, n)
        A = torch.tanh(torch.bmm(self.Ws1.repeat(bs, 1, 1), H_T))  # (bs, d, n)
        A = torch.bmm(self.Ws2.repeat(bs, 1, 1), A)  # (bs, r, n)
        A = torch.softmax(A.view(-1, n), dim=-1).view(bs, -1, n)  # (bs, r, n)
        M = torch.bmm(A, outputs)
        hidden = hidden.view(self.n_layers, bs, -1)
        hidden = torch.tanh(self.hidden_fc(hidden))
        return M, hidden, outputs, length_emb
