import argparse

from monai.utils import set_determinism

from src.trainers.AttSeq2SeqTrainer import AttSeq2SeqTrainer
from src.trainers.GateSeq2SeqTrainer import GateSeq2SeqTrainer
from src.trainers.Seq2SeqTrainer import Seq2SeqTrainer

from src.utils.const import ModelName, SEED

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='GSEQ', type=ModelName)
    parser.add_argument('--data_source', default='2015', type=str)
    parser.add_argument('--data_category', default='Restaurants', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--l2', default=0.00005, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--loss_weight', default='0.8', type=float)
    parser.add_argument('--teacher_forcing_ratio', default=1, type=float)
    parser.add_argument('--mlp_d', default='50', type=int)
    parser.add_argument('--attn_hops', default='500', type=int)
    parser.add_argument('--hidden_type', default='gate', type=str, help='gate,hidden,cat')
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()
    set_determinism(seed=SEED)
    if args.model_name == ModelName.SEQ:
        Seq2SeqTrainer(args).run(args.mode)
    elif args.model_name == ModelName.GSEQ:
        GateSeq2SeqTrainer(args).run(args.mode)
    elif args.model_name == ModelName.ASEQ:
        AttSeq2SeqTrainer(args).run(args.mode)
