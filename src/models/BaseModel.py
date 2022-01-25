import torch
import torch.nn as nn
import numpy as np

from src.utils.const import EOS_WORD_VALUE


class BaseModel(nn.Module):
    def __init__(self, model_settings):
        super(BaseModel, self).__init__()
        self.model_settings = model_settings
        self.device = model_settings['device']

    def forward(self, input):
        pass

    def predict(self, input):
        pass

    def compute_logit(self, input):
        pass

    def compute_loss(self, x, y):
        pass

    def assemble_tensor(self, input):
        review, aspect_label, label = input
        return review.to(self.device), aspect_label.to(self.device), label.to(self.device)

    def get_model_name(self):
        pass

    def make_label(self, l):
        length = self.model_settings['num_category'] - 4
        result = np.zeros(length)
        for i in l:
            if i > EOS_WORD_VALUE:
                result[i - 4] = 1
        return result
