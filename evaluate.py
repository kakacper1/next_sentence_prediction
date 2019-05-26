import pprint
import argparse
import torch

from torchtext import data
from sklearn.metrics import accuracy_score


# handle arguments
parser = argparse.ArgumentParser()

# arguments initialization:

parser.add_argument('--model_path', type=str, default='./res/models/emb_sum_50_50/pokus_model_2019-4-23_8:11.pt')

parser.add_argument('--data_set_num', type=int, default=1)

parser.add_argument('--data_path', type=str, default='./res/data/eval/test_all_true')

parser.add_argument('--yes_cuda', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=512)

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

    # parsing arguments
    args = parser.parse_args()

    # setting device
    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pprint.PrettyPrinter().pprint('Using ' + str(device) )

    # initialize model
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    # Load evaluation dataset
    TEXT = data.Field(sequential=True,
                      include_lengths=True,
                      use_vocab=True)

    LABELS = data.Field(sequential=False,
                        use_vocab=True,
                        pad_token=None,
                        unk_token=None)

    eval_fields = [
        ('label', LABELS),  # process it as label
        ('sentence_a', TEXT),  # process it as text
        ('sentence_b', TEXT)  # process it as text
    ]

    eval_dataset = data.TabularDataset(
        path='../resources/data/wiki_preprocessed/test_all_true.tsv',
        format='tsv',
        fields=eval_fields,
        skip_header=True)

    evaluation_iter = data.BucketIterator(
        dataset=eval_dataset, batch_size=args.batch_size)

    # assertion? batch = next(iter(evaluation_iter))

    #Run evaluation
    test_targs, test_preds = [], []
    ### Evaluate test set
    for batch_idx, batch in enumerate(evaluation_iter):
        output = model(batch)
        preds = torch.max(output, 1)[1]

        test_targs += list(batch.label.numpy())
        if (use_cuda):
            test_preds += list(get_numpy(preds.data))
        else:
            test_preds += list(preds.data.numpy())
    pprint.PrettyPrinter().pprint("\nEvaluation set Acc:  %f" % (accuracy_score(test_targs, test_preds)))
    pprint.PrettyPrinter().pprint('size of evaluation dataset: %d sentence pairs' % (len(eval_dataset)))


if __name__ == '__main__':
    main()