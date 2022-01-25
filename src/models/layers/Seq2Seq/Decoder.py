import torch.nn as nn

from src.models.layers.Attention import Attention


class Decoder(nn.Module):
    def __init__(self, num_category, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.num_category = num_category
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_category, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_category)
        nn.init.xavier_uniform_(self.fc_out.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, _):
        input = input.unsqueeze(-1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden
