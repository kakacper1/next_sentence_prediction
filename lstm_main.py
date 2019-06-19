import argparse
from datetime import datetime
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
#from src.mlstm_model import MatchLSTM
from src.utils import *
from src.lstm_models import LSTM_for_NSP,LSTM_for_SNLI
from torchtext import data
from torchtext.vocab import GloVe

# Ref.
# https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua
parser = argparse.ArgumentParser()

parser.add_argument('--task_type', type=str, default="nsp")  # "nli"
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--data_path', type=str, default='./res/data/train/wiki_swapped_new')
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--grad_max_norm', type=float, default=0.)  #

parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)

parser.add_argument('--slice_train', type=int, default=None)
parser.add_argument('--slice_val', type=int, default=None)
parser.add_argument('--slice_test', type=int, default=None)

parser.add_argument('--dropout', type=float, default=0.2)  #
parser.add_argument('--dropout_emb', type=float, default=0.)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=20)

parser.add_argument('--log_interval', type=int, default=10000)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)


def train_epoch(device, loader, model, epoch, optimizer, loss_func, config):
    model.train()
    train_loss = 0.
    example_count = 0
    correct = 0
    start_t = datetime.now()
    for batch_idx, ex in enumerate(loader):
        target = ex.label.to(device)
        optimizer.zero_grad()
        output = model(ex.premise[0], ex.premise[1], ex.hypothesis[0], ex.hypothesis[1])
        loss = loss_func(output, target)
        loss.backward()
        if config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.req_grad_params,
                                           config.grad_max_norm)
        optimizer.step()

        batch_loss = len(output) * loss.item()
        train_loss += batch_loss
        example_count += len(target)

        pred = torch.max(output, 1)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '{} Train Epoch {}, [{}/{} ({:.1f}%)],\tBatch Loss: {:.6f}' \
                .format(datetime.now(), epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(output))
            print(_progress)

    train_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} Train Epoch {}, Avg. Loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.
          format(datetime.now()-start_t, epoch, train_loss,
                 correct, len(loader.dataset), 100. * acc))
    return train_loss


def evaluate_epoch(device, loader, model, epoch, loss_func, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex.label.to(device)
            output = model(ex.premise[0], ex.premise[1], ex.hypothesis[0], ex.hypothesis[1])
            loss = loss_func(output, target)
            eval_loss += len(output) * loss.item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.1f}%)'.format(datetime.now()-start_t, mode,
                                             epoch, eval_loss,
                                             correct, len(loader.dataset),
                                             100. * acc))
    return eval_loss, acc


def main():
    # handle and display arguments
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    # handle cuda usage
    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # set a seed to ensure deterministic start
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # print type of execution
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')

    # to get the right dataset
    if args.task_type == "nsp":
        train, val, test, TEXT, LABELS = get_nsp_dataset(args)
    elif args.task_type == "snli":
        train, val, test, TEXT, LABELS = get_snli_dataset(args)


    # create batches:
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.premise), device=device, repeat=False)

    TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=args.embedding_dim))
    LABELS.build_vocab(train)


    print('#examples', len(train_iter.dataset), len(val_iter.dataset),
          len(test_iter.dataset))


    model = LSTM_for_SNLI(args, TEXT).to(device)

    optimizer = optim.Adam(model.req_grad_params, lr=args.lr,
                           betas=(0.9, 0.999), amsgrad=True)

    loss_func = nn.CrossEntropyLoss().to(device)

    best_loss = float('inf')
    best_acc = 0.
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(device, train_iter, model, epoch, optimizer, loss_func,
                    args)

        valid_loss, valid_acc = \
            evaluate_epoch(device, val_iter, model, epoch, loss_func,
                           'Valid')
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            best_epoch = epoch
        print('\tLowest Valid Loss {:.6f}, Acc. {:.1f}%, Epoch {}'.
              format(best_loss, 100 * best_acc, best_epoch))

        evaluate_epoch(device, test_iter, model, epoch, loss_func, 'Test')

        # learning rate decay
        for param_group in optimizer.param_groups:
            print('lr: {:.6f} -> {:.6f}'
                  .format(param_group['lr'], param_group['lr'] * args.lr_decay))
            param_group['lr'] *= args.lr_decay


if __name__ == '__main__':
    main()
