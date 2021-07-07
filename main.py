import argparse

from location.process import convert_crf_examples, get_training_set
from src.utils.model_utils import CRFModel
from src.utils.trainer import train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--other_lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--swa_start', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_grad_norm', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--attack_train', type=str, default='')
    opt = parser.parse_args()
    train_dataset = get_training_set()
    model = CRFModel(bert_dir='../bert/usr', num_tags=77, dropout_prob=0.1)
    train(opt, model, train_dataset)






