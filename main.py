import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
from datetime import datetime
import time
from src.bow_model import BoWNet
import pprint
import torch
import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch.nn as nn
import torch.optim as optim

# dataset:
from torchtext import data, datasets
from torchtext.vocab import GloVe


# Ref.
# https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--data_path', type=str, default='./res/data/train/wiki_swapped_new')
parser.add_argument('--num_classes', type=int, default=2)


parser.add_argument('--slice_train', type=int, default=None)
parser.add_argument('--slice_val', type=int, default=None)
parser.add_argument('--slice_test', type=int, default=None)

parser.add_argument('--lr', type=float, default=0.000015)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--grad_max_norm', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)

parser.add_argument('--dropout_ln', type=float, default=0.3)
parser.add_argument('--dropout_emb', type=float, default=0.3)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_epochs_num', type=int, default=120)
parser.add_argument('--patience', type=int, default=4)

#parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--model_name', type=str, default="batch_size_512")


# this is part of the code to refactor: We could take out code from the main function and organize/merge it here
def train_epoch(device, loader, model, epoch, optimizer, loss_func, config):
    model.train()
    train_loss = 0.
    example_count = 0
    correct = 0
    start_t = datetime.now()
    for batch_idx, ex in enumerate(loader):
        target = ex[4].to(device)
        optimizer.zero_grad()
        output = model(ex[0], ex[1], ex[2], ex[3])
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

# this is part of the code to refactor: We could take out code from the main function and organize/merge it here
def evaluate_epoch(device, loader, model, epoch, loss_func, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex[4].to(device)
            output = model(ex[0], ex[1], ex[2], ex[3])
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


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    return x.cuda() #if use_cuda else x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    return x.cpu().data.numpy() #if use_cuda else x.data.numpy()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def get_wiki_dataset( args ):

    # define the columns that we want to process:
    TEXT = data.Field(sequential=True,
                      include_lengths=True,
                      use_vocab=True)
    LABELS = data.Field(sequential=False,
                        use_vocab=True,
                        pad_token=None,
                        unk_token=None)

    train_val_fields = [
        ('label', LABELS),  # process it as label
        ('premise', TEXT),  # process it as text
        ('hypothesis', TEXT)  # process it as text
    ]

    # Load data set:
    train, val, test = data.TabularDataset.splits(
        path=args.data_path, train='train.tsv',
        validation='dev.tsv', test='test.tsv', format='tsv',
        fields=train_val_fields,
        skip_header=True)

    # Slice data if needed:
    if args.slice_train is not None:
        train.examples = train.examples[:args.slice_train]
        assert (len(train) == args.slice_train), "Train data set does not equal"+str(args.slice_train)+"!"
    if args.slice_val is not None:
        val.examples = val.examples[:args.slice_val]
        assert (len(val) == args.slice_val), "Val data set does not equal" + str(args.slice_val) + "!"
    if args.slice_test is not None:
        test.examples = test.examples[:args.slice_test]
        assert (len(test) == args.slice_test), "Test data set does not equal" + str(args.slice_test) + "!"

    return train, val, test, TEXT, LABELS


def set_plots_model_names(now_str, args):
    model_path = "./res/models/" + args.model_name + "_%s.pt" % now_str
    learning_curve_path = "./res/plots/learning_curve_%s.svg" % now_str
    roc_curve_path = "./res/plots/roc_curve_%s.svg" % now_str
    conf_mat_path = "./res/plots/confusion_matrix_%s.svg" % now_str
    norm_conf_mat_path = "./res/plots/normalized_confusion_matrix_%s.svg" % now_str
    return model_path, learning_curve_path, roc_curve_path, conf_mat_path, norm_conf_mat_path


def draw_learning_curve(train_acc, valid_acc, path ='./res/plots'):
    epoch = np.arange(len(train_acc))

    plt.figure()
    plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
    plt.legend(['Train Acc', 'Val Acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    # save plot
    plt.savefig(path)


def draw_roc_curve(y_test_preds, y_test_targs, path='.res/plots'):

    fpr, tpr, thresholds = roc_curve(y_test_targs, y_test_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path)



def main():

    # handling arguments
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    # handling timestamp:
    cur_date = datetime.datetime.now()
    now_str = '%d-%d-%d_%d:%d' % (cur_date.year, cur_date.month, cur_date.day, cur_date.hour, cur_date.minute)
    model_path, learning_curve_path, roc_curve_path, conf_mat_path, norm_conf_mat_path = set_plots_model_names(now_str, args)

    # handling precessing time
    start = time.time()

    # handling cuda
    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pprint.PrettyPrinter().pprint('Using ' +  str(device) )

    # to make pytorch code deterministic
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        print('CUDA device_count {0}'.format(torch.cuda.device_count()) if use_cuda else 'CPU')

    # data load:
    #     with open(args.data_path, 'rb') as f:
    #     snli_dataset = pickle.load(f)

    train, val, test, TEXT, LABELS = get_wiki_dataset(args)

    # create batches:

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.premise), device=device, repeat=False)

    TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=args.embedding_dim))
    LABELS.build_vocab(train)

    batch = next(iter(train_iter))
    print("Tokenize premise:\n", batch.premise)
    print("Tokenize hypothesis:\n", batch.hypothesis)
    print("Entailment labels:\n", batch.label)

    model = BoWNet(args, TEXT).to(device)

    #    loss_func:
    criterion = nn.CrossEntropyLoss().to(device)

    #    diff of model.parameters() and model.req_grad_params:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    ## TRAIN:
    patience_counter = 0
    best_val_acc = 0
    iterations = 0

    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    losses = []

    for epoch in range(args.max_epochs_num):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        cur_loss = 0

        model.train()
        for batch_idx, batch in enumerate(train_iter):
            # clear gradient accumulators
            optimizer.zero_grad()
            iterations += 1

            # forward pass
            output = model(batch)

            # compute gradients given loss
            if use_cuda:
                batch_loss = criterion(output, get_variable(batch.label))
            #    batch_loss = criterion(output, batch.label))
            else:
                batch_loss = criterion(output, batch.label)
            batch_loss.backward()
            optimizer.step()

            cur_loss += batch_loss.detach().item()  # without item I was getting memory leak when introduction dropout
        losses.append(cur_loss / args.batch_size)

        model.eval()
        ### Evaluate training
        train_preds, train_targs = [], []
        for batch_idx, batch in enumerate(train_iter):

            output = model(batch)
            preds = torch.max(output, 1)[1]
            #         preds = output

            if (use_cuda):
                train_targs += list(get_numpy(batch.label))
                train_preds += list(get_numpy(preds.data))
            else:
                train_targs += list(batch.label.numpy())
                train_preds += list(preds.data.numpy())

        ### Evaluate validation
        val_preds, val_targs = [], []
        for batch_idx, batch in enumerate(val_iter):
            output = model(batch)
            preds = torch.max(output, 1)[1]

            if (use_cuda):
                val_targs += list(get_numpy(batch.label))
                val_preds += list(get_numpy(preds.data))
            else:
                val_targs += list(batch.label.numpy())
                val_preds += list(preds.data.numpy())

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

        if epoch % 1 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch + 1, losses[-1], train_acc_cur, valid_acc_cur))

        if valid_acc_cur >= best_val_acc:
            patience_counter = 0
            best_val_acc = valid_acc_cur
        else:
            patience_counter += 1

        # forced finish in case of overfitting
        if patience_counter > args.patience:
            print('Training terminated: PATIENCE exceeded')
            break

    # draw_results
    draw_learning_curve(path=learning_curve_path)

    # evaluation of test dataset
    test_targs, test_preds, raw_outputs_class_one, raw_outputs_class_two = [], [], [], []
    # Evaluate test set
    for batch_idx, batch in enumerate(test_iter):
        output = model(batch)
        preds = torch.max(output, 1)[1]

        if (use_cuda):
            raw_outputs_class_one += list(get_numpy(output[:, 0]))
            raw_outputs_class_two += list(get_numpy(output[:, 1]))
            test_targs += list(get_numpy(batch.label))
            test_preds += list(get_numpy(preds.data))
        else:
            test_preds += list(preds.data.numpy())
            test_targs += list(batch.label.numpy())


    print("\nTest set Acc:  %f" % (accuracy_score(test_targs, test_preds)))
    print('size of train set: %d; val: %d; test: %d' % (len(train), len(val), len(test)))

    # for epoch in range(1, args.epochs + 1):
    #     train_epoch(device, train_loader, model, epoch, optimizer, loss_func,args)

    # for param_group in optimizer.param_groups:
    #         print('lr: {:.6f} -> {:.6f}'
    #               .format(param_group['lr'], param_group['lr'] * args.lr_decay))
    #         param_group['lr'] *= args.lr_decay

    # Draw a roc curve

    y_test_preds = np.array(raw_outputs_class_two)
    y_test_targs = np.array(test_targs)

    draw_roc_curve(y_test_preds, y_test_targs, path=roc_curve_path)

    # Draw confusion matrices:

    class_names = np.array(['actual_next', 'not_next'], dtype='<U10')
    nsp_test_pred = test_preds
    nsp_test_target = test_targs

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(nsp_test_target, nsp_test_pred, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(conf_mat_path)
    # Plot normalized confusion
    plot_confusion_matrix(nsp_test_target, nsp_test_pred, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(norm_conf_mat_path)

    # Save model
    torch.save(model, model_path)

    # handling precessing time
    stop = time.time()
    run_time = stop - start
    print("It took: ", run_time, " seconds")


if __name__ == '__main__':
    main()
