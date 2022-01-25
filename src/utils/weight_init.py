import math

import torch
import numpy as np
import torch.nn as nn


def apply_emb_weight(emb, weights, device):
    if type(emb) is not nn.Embedding:
        return nn.init.xavier_uniform_(emb)
    if weights is not None:
        weight = torch.from_numpy(weights).type(torch.FloatTensor).to(device)
        emb = nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = True
        emb.padding_idx = 0
    else:
        nn.init.xavier_uniform_(emb.weight.data)
        emb.weight.data[0] = torch.zeros_like(emb.weight.data[0])
    return emb
