from src.models.Seq2Seq import Seq2Seq
from src.trainers.BaseTrainer import BaseTrainer


class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, args):
        super(Seq2SeqTrainer, self).__init__(args)

    def create_model(self):
        super(Seq2SeqTrainer, self).create_model()
        word_embeddings_weight = self.load_embeddings()
        model = Seq2Seq(self.model_settings, word_embeddings_weight)
        return model.to(model.device)

    def init_model_settings(self):
        super(Seq2SeqTrainer, self).init_model_settings()
        self.model_settings['teacher_forcing_ratio'] = self.args.teacher_forcing_ratio
