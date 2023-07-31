import argparse
import torch
import torch.optim as optim
from relation.CNN import CNN
from relation.callback import MyCallBack
from relation.data import load_data, get_data_iterator
from relation.config import Config
from relation.evaluate import evaluate_raw_text, metric
import torch.nn.functional as F
from fastNLP import Trainer, LossBase
import os

seed = 226
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--dataset', default='FinCorpus.CN', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='./pretrain_models/chinese-roberta-wwm-ext', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
args = parser.parse_args()
con = Config(args)


if __name__ == '__main__':
    model = CNN(con).to(device)
    data_bundle, rel_vocab = load_data(con.train_path, con.dev_path, con.test_path, con.rel_path)
    test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
    model.load_state_dict(torch.load(os.path.join(con.save_weights_dir, con.weights_save_name), map_location=device), False)
    print("-" * 5 + "Begin Testing" + "-" * 5)
    # precision, recall, f1_score = evaluate_raw_text(test_data, rel_vocab, con, model)
    precision, recall, f1_score = metric(test_data, rel_vocab, con, model)
    with open(os.path.join(con.save_logs_dir, con.log_save_name), 'a+') as f_log:
        f_log.write('batch size: {}, f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}\n\n'
                .format(con.batch_size, f1_score, precision, recall))
