import pprint
import argparse
import torch
from src.utils import *
from torchtext import data
from sklearn.metrics import accuracy_score
from torchtext.vocab import GloVe


# handle arguments
parser = argparse.ArgumentParser()

# arguments initialization:

parser.add_argument('--model_path', type=str, default='./res/models/model_lstm_snli_2019-6-20_12:53.pt')

parser.add_argument('--data_set_num', type=int, default=1)

parser.add_argument('--data_path', type=str, default='./res/data/eval/test_all_true')
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--task_type', type=str, default="snli")  # "nsp"

parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--load_args', type=bool, default=False)
parser.add_argument('--args_path', type=str, default='./res/models/')


parser.add_argument('--slice_train', type=int, default=None)
parser.add_argument('--slice_val', type=int, default=None)
parser.add_argument('--slice_test', type=int, default=None)

#parser.add_argument('--model_name', type=str, default='./res/data/wiki_pages/basic_50_50')


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    return x.cuda() #if use_cuda else x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    return x.cpu().data.numpy() #if use_cuda else x.data.numpy()



def main():

    # handling arguments
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)


    # setting device
    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pprint.PrettyPrinter().pprint('Using ' + str(device) )

    # initialize model
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    # to get the right dataset
    if args.task_type == "nsp":
        train, val, test, TEXT, LABELS = get_nsp_dataset(args)
    elif args.task_type == "snli":
        train, val, test, TEXT, LABELS = get_snli_dataset(args)

    # create batches:
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.premise), device=device, repeat=False)

    #TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=args.embedding_dim))
    #LABELS.build_vocab(train)

    print('#examples', len(train_iter.dataset), len(val_iter.dataset),
          len(test_iter.dataset))

    # assertion? batch = next(iter(evaluation_iter))

    #Run evaluation
    test_targs, test_preds = [], []
    ### Evaluate test set
    for batch_idx, batch in enumerate(test_iter):
        output = model(batch.premise[0], batch.premise[1], batch.hypothesis[0], batch.hypothesis[1])
        preds = torch.max(output, 1)[1]

        test_targs += list(batch.label.numpy())
        if (use_cuda):
            test_preds += list(get_numpy(preds.data))
        else:
            test_preds += list(preds.data.numpy())
    pprint.PrettyPrinter().pprint("\nEvaluation set Acc:  %f" % (accuracy_score(test_targs, test_preds)))
    pprint.PrettyPrinter().pprint('size of evaluation dataset: %d sentence pairs' % (len(test)))


if __name__ == '__main__':
    main()