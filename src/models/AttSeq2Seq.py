from src.models.Seq2Seq import Seq2Seq
from src.models.layers.AttSeq2Seq.AttentionDecoder import AttentionDecoder


class AttSeq2Seq(Seq2Seq):
    def __init__(self, model_settings, word_embeddings_weight):
        super(AttSeq2Seq, self).__init__(model_settings, word_embeddings_weight)
        hidden_dim = model_settings['hidden_dim']
        n_layers = model_settings['n_layers']
        num_category = model_settings['num_category']
        cate_dim = model_settings['cate_dim']
        dropout = model_settings['dropout']
        self.decoder = AttentionDecoder(num_category, cate_dim, hidden_dim, n_layers, dropout)

    def get_model_name(self):
        return 'AttSeq2Seq'
