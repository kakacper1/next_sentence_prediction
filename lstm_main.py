import argparse
from datetime import datetime
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.utils import *
from src.lstm_models import LSTM_for_NSP,LSTM_for_SNLI
from torchtext import data
from torchtext.vocab import GloVe

# Ref.
# https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua
parser = argparse.ArgumentParser()

parser.add_argument('--task_type', type=str, default="nsp")  # "snli"
parser.add_argument('--seed', type=int, default=2019)
#parser.add_argument('--data_path', type=str, default='.res/data/wiki_pages/basic_50_50')
parser.add_argument('--data_path', type=str, default='.res/data/wiki_pages/swapped_33_33_33')
parser.add_argument('--num_classes', type=int, default=2)

# to pick the right model:
parser.add_argument('--model_name', type=str, default="lstm_snli") # bow_nsp , bow_snli

parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--lr_decay', type=float, default=0.96)
parser.add_argument('--grad_max_norm', type=float, default=0.)  #

parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)

parser.add_argument('--slice_train', type=int, default=None)
parser.add_argument('--slice_val', type=int, default=None)
parser.add_argument('--slice_test', type=int, default=None)

parser.add_argument('--dropout', type=float, default=0.25)  #
parser.add_argument('--dropout_emb', type=float, default=0.)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=25)

parser.add_argument('--log_interval', type=int, default=10000)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--patience', type=int, default=4)



parser.add_argument('--eed', type=bool, default=False)
parser.add_argument('--eed_swapped', type=str, default='./res/data/wiki_pages/swapped_33_33_33/test_50_50_true_swap.tsv')
parser.add_argument('--eed_regular', type=str, default='./res/data/wiki_pages/swapped_33_33_33/test_50_50_true_random.tsv')


parser.add_argument('--test_con', type=bool, default=True)
parser.add_argument('--con_data_path', type=str, default='./res/data/wiki_pages/swapped_33_33_33/test_con.tsv')

parser.add_argument('--test_rand', type=bool, default=True)
parser.add_argument('--rand_data_path', type=str, default='./res/data/wiki_pages/swapped_33_33_33/test_rand.tsv')

parser.add_argument('--test_swap', type=bool, default=True)
parser.add_argument('--swap_data_path', type=str, default='./res/data/wiki_pages/swapped_33_33_33/test_swap.tsv')



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
                 correct, len(loader.dataset), 100. * acc ))
    return train_loss, acc


def evaluate_epoch(device, loader, model, epoch, loss_func, mode, config, finish=False):
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

    if (epoch == config.epochs and mode == 'Test') or finish:
        # Draw confusion matrices:
        if config.task_type == 'snli':
            class_names = np.array(['entailment', 'contradiction', 'neutral'], dtype='<U10')
        elif config.task_type == 'nsp':
            class_names = np.array(['Consecutive', 'Random', ], dtype='<U10')

        np.set_printoptions(precision=2)

        # Plot normalized confusion
        if config.use_cuda:
            target = get_numpy(target)
            pred = get_numpy(pred)

        plot_confusion_matrix(target, pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.savefig(config.norm_conf_mat_path)

    return eval_loss, acc


def set_plots_model_names(now_str, args):
    model_path = "./res/models/model_" + args.model_name + "_%s.pt" % now_str
    learning_curve_path = "./res/plots/learning_curve_%s.svg" % now_str
    roc_curve_path = "./res/plots/roc_curve_%s.svg" % now_str
    conf_mat_path = "./res/plots/confusion_matrix_%s.svg" % now_str
    norm_conf_mat_path = "./res/plots/normalized_confusion_matrix_%s.svg" % now_str
    args_path = "./res/models/dict_" + args.model_name + "_%s" % now_str

    roc_curve_path_ext_swapped = "./res/plots/swapped_roc_curve_%s.svg" % now_str

    roc_curve_path_ext_regular = "./res/plots/regular_roc_curve_%s.svg" % now_str

    return model_path, learning_curve_path, roc_curve_path, conf_mat_path, norm_conf_mat_path, args_path, roc_curve_path_ext_swapped, roc_curve_path_ext_regular

def main():
    patience_counter = 0
    # handle and display arguments
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    # handling timestamp:
    cur_date = datetime.now()
    now_str = '%d-%d-%d_%d:%d' % (cur_date.year, cur_date.month, cur_date.day, cur_date.hour, cur_date.minute)
    model_path, learning_curve_path, roc_curve_path, conf_mat_path, norm_conf_mat_path, args_path, roc_curve_path_ext_swapped ,roc_curve_path_ext_regular = set_plots_model_names(
        now_str, args)
    args.norm_conf_mat_path = norm_conf_mat_path
    args.roc_curve_path_ext_swapped = roc_curve_path_ext_swapped
    args.roc_curve_path_ext_regular = roc_curve_path_ext_regular

    # handle cuda usage
    args.use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # set a seed to ensure deterministic start
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    # print type of execution
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if args.use_cuda else 'CPU')

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

    model = LSTM_for_SNLI(args, TEXT, LABELS).to(device)

    optimizer = optim.Adam(model.req_grad_params, lr=args.lr,
                           betas=(0.9, 0.999), amsgrad=True)

    loss_func = nn.CrossEntropyLoss().to(device)

    best_loss = float('inf')
    best_valid_acc = float('-inf')
    best_acc = 0.
    best_epoch = 0

    test_losses = []
    test_accuracies = []

    valid_losses = []
    valid_accuracies = []

    train_losses = []
    train_accuracies = []
    is_last_one = False

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_epoch(device, train_iter, model, epoch, optimizer, loss_func, args)
        train_losses.append( train_loss)
        train_accuracies.append( train_acc)

        valid_loss, valid_acc = evaluate_epoch(device, val_iter, model, epoch, loss_func, 'Valid', args)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        if valid_acc >= best_valid_acc:
            patience_counter = 0
            best_valid_acc = valid_acc
        else:
            patience_counter += 1

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            best_epoch = epoch
        print('\tLowest Valid Loss {:.6f}, Acc. {:.1f}%, Epoch {}'.
              format(best_loss, 100 * best_acc, best_epoch))
        if patience_counter > args.patience:
            is_last_one = True
        iter_test_loss, iter_test_accuracy = evaluate_epoch(device, test_iter, model, epoch, loss_func, 'Test', args, finish=is_last_one)
        test_losses.append(iter_test_loss)
        test_accuracies.append(iter_test_accuracy)

        # forced finish in case of overfitting
        if patience_counter > args.patience:
            print('Training terminated: PATIENCE exceeded')
            break

        # learning rate decay
        for param_group in optimizer.param_groups:
            print('lr: {:.6f} -> {:.6f}'
                  .format(param_group['lr'], param_group['lr'] * args.lr_decay))
            param_group['lr'] *= args.lr_decay

    # draw_results
    draw_learning_curve(train_accuracies, valid_accuracies, path=learning_curve_path)

    # works with torchtext bigger than 0.4.0 amd on dtu server there is 0.3.1
    # ask admins?

    # Save model
    #if args.save_model:
    #    torch.save(model, model_path)
    #    print('Model saved: ', str(model_path))


    # external dataset evaluation:

    if args.eed == True:

        evaluation_iter, eval_dataset = load_evaluation_dataset(TEXT, LABELS, path=args.eed_regular, )

        test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []

        ### Evaluate test set
        model.eval()
        for batch_idx, batch in enumerate(evaluation_iter):
            output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
            preds = torch.max(output, 1)[1]

            if (args.use_cuda):
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_targs += list(get_numpy(batch.label))
                test_preds += list(get_numpy(preds.data))
            else:
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_preds += list(preds.data.numpy())
                test_targs += list(batch.label.numpy())
        test_accuracy = accuracy_score(test_targs, test_preds)

        print("\nEvaluation set Acc:  %f" % (test_accuracy))
        print('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))

        y_test_preds = np.array(raw_outputs_class_two)
        y_test_targs = np.array(test_targs)

        draw_roc_curve(y_test_preds, y_test_targs, path=args.roc_curve_path_ext_regular)

    if args.eed == True:

        evaluation_iter, eval_dataset = load_evaluation_dataset(TEXT, LABELS, path=args.eed_swapped)

        test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []

        ### Evaluate test set
        model.eval()
        for batch_idx, batch in enumerate(evaluation_iter):
            output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
            preds = torch.max(output, 1)[1]

            if (args.use_cuda):
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_targs += list(get_numpy(batch.label))
                test_preds += list(get_numpy(preds.data))
            else:
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_preds += list(preds.data.numpy())
                test_targs += list(batch.label.numpy())
        test_accuracy = accuracy_score(test_targs, test_preds)
        print("\nEvaluation set Acc:  %f" % (test_accuracy))
        print('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))

        y_test_preds = np.array(raw_outputs_class_two)
        y_test_targs = np.array(test_targs)

        draw_roc_curve(y_test_preds, y_test_targs, path=args.roc_curve_path_ext_swapped  )

    if args.test_con == True:
        evaluation_iter, eval_dataset = load_evaluation_dataset(TEXT, LABELS, path=args.con_data_path)

        test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []

        ### Evaluate test set
        model.eval()
        for batch_idx, batch in enumerate(evaluation_iter):
            output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
            preds = torch.max(output, 1)[1]

            if (args.use_cuda):
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_targs += list(get_numpy(batch.label))
                test_preds += list(get_numpy(preds.data))
            else:
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_preds += list(preds.data.numpy())
                test_targs += list(batch.label.numpy())
        test_accuracy = accuracy_score(test_targs, test_preds)
        print("\nEvaluation of Consecutive data set Acc:  %f" % (test_accuracy))
        print('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))

    if args.test_rand == True:
        evaluation_iter, eval_dataset = load_evaluation_dataset(TEXT, LABELS, path=args.rand_data_path)

        test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []

        ### Evaluate test set
        model.eval()
        for batch_idx, batch in enumerate(evaluation_iter):
            output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
            preds = torch.max(output, 1)[1]

            if (args.use_cuda):
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_targs += list(get_numpy(batch.label))
                test_preds += list(get_numpy(preds.data))
            else:
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_preds += list(preds.data.numpy())
                test_targs += list(batch.label.numpy())
        test_accuracy = accuracy_score(test_targs, test_preds)
        print("\nEvaluation of Random data set Acc:  %f" % (test_accuracy))
        print('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))

    if args.test_swap == True:
        evaluation_iter, eval_dataset = load_evaluation_dataset(TEXT, LABELS, path=args.swap_data_path)

        test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []

        ### Evaluate test set
        model.eval()
        for batch_idx, batch in enumerate(evaluation_iter):
            output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
            preds = torch.max(output, 1)[1]

            if (args.use_cuda):
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_targs += list(get_numpy(batch.label))
                test_preds += list(get_numpy(preds.data))
            else:
                raw_outputs_class_one += list(get_numpy(output[:, 0]))
                raw_outputs_class_two += list(get_numpy(output[:, 1]))
                test_preds += list(preds.data.numpy())
                test_targs += list(batch.label.numpy())
        test_accuracy = accuracy_score(test_targs, test_preds)
        print("\nEvaluation of Swapped data set Acc:  %f" % (test_accuracy))
        print('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))


if __name__ == '__main__':
    main()
