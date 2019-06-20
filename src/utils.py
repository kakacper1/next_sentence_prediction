import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchtext import data, datasets
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import sys
#import os

from src.bow_models import BoW_for_NSP, BoW_for_SNLI

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    return x.cuda() #if use_cuda else x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    return x.cpu().data.numpy() #if use_cuda else x.data.numpy()

def get_nsp_dataset( args ):
    """ get next_sentence prediction dataset """

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

    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #cwd  = os.getcwd()

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

    print("NSP Dataset loaded succesfully")
    return train, val, test, TEXT, LABELS


def get_requested_model(args, TEXT, device):

    """ In this project we have 2 types of model 2 types of task.
    All of that results in 4 different model to choose:
    - bow_nsp
    - lstm_nsp
    - bow_snli
    - lstm_snli
     This function helps to load the right model accordingly to the
     arguments from the input"""

    model = None

    if args.model_name == "bow_nsp":
            model = BoW_for_NSP(args, TEXT).to(device)
    elif args.model_name == "lstm_nsp":
            print('lstm_nsp Not ready yet')
            #model = LSTM_for_NSP(args, TEXT).to(device)
    elif args.model_name == "bow_snli":
            model = BoW_for_SNLI(args, TEXT).to(device)
    elif args.model_name == "lstm_snli":
            print('lstm_nsp Not ready yet')
            #model = LSTM_for_SNLI(args, TEXT).to(device)
    else:
        print("There is no model type called: ", args.model_type )

    if model is None:
        sys.exit("Model is None. Aborting script execution... ")

    return model


def get_snli_dataset( args ):
    """ get snli dataset """

    TEXT = datasets.nli.ParsedTextField(lower=True)
    LABELS = data.LabelField()

    train, val, test = datasets.SNLI.splits(TEXT, LABELS)

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

    print("SNLI Dataset loaded succesfully")
    return train, val, test, TEXT, LABELS


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


def load_evaluation_dataset( TEXT, LABELS, path):

    # define the columns that we want to process and how to process
   #TEXT = data.Field(sequential=True,
   #                  include_lengths=True,
   #                  use_vocab=True)
   #LABELS = data.Field(sequential=False,
   #                    use_vocab=True,
   #                    pad_token=None,
   #                    unk_token=None)

    eval_fields = [
        ('label', LABELS),  # process it as label
        ('sentence_a', TEXT),  # process it as text
        ('sentence_b', TEXT)  # process it as text
    ]

    eval_dataset = data.TabularDataset(
        path=path,
        format='tsv',
        fields=eval_fields,
        skip_header=True)

    evaluation_iter = data.BucketIterator(
        dataset=eval_dataset, batch_size=512)

    batch = next(iter(evaluation_iter))
    print(batch)

    return evaluation_iter, eval_dataset

