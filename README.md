# next-sentence-prediction

* [Python 3](https://www.python.org/downloads/)
* [PyTorch 1.0.1](https://pytorch.org/)

# Datasets:
* [Download snli_1.0.zip (90.2 MB)](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) and decompress snli_1.0_train.txt, snli_1.0_dev.txt and snli_1.0_test.txt to __data/__
    * More information can be found at [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
* For next sentence prediction task we have construct our own dataset that is composed of two sentences (chunk of text up to about 100 tokens) Premise and hypothesis. Based on these two sentences we predict if the sentences are consecutive or not.

# Word Embeddings
* [Download glove.840B.300d.zip (2.0 GB)](http://nlp.stanford.edu/data/glove.840B.300d.zip) and decompress glove.840B.300d.txt to __$HOME/common/__
    * See [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

# Running the code:

- to run SNLI task with BoW model:
main.py --task_type snli --model_name bow_snli --dropout_ln 0.2 --dropout_emb 0.2 --lr 0.0005

- to run NSP task with BoW model:
main.py --task_type nsp --model_name bow_nsp --num_classes 2 --slice_train 50000 --slice_val 1000 --slice_test 1000

- to run SNLI task with LSTM model:
main.py --task_type snli --model_name lstm_snli --dropout_ln 0.2 --dropout_emb 0.2 --lr 0.0005

- to run NSP task with LSTM model:
main.py --task_type nsp --model_name lstm_nsp --dropout_ln 0.2 --dropout_emb 0.2 --lr 0.0005

# Using DTU server:

1. Log in to the DTU server and Openxterm-LSF10-login-name. Then pick the GPU you would like to use:
```
$ k40sh 
#or
$ voltash
```
2. Then import necessary modules:
```
# Change directory:
$ cd /work3/s180011/next_sentence_prediction/
$ module load python3/3.6.2 numpy/1.13.1-python-3.6.2-openblas-0.2.20 matplotlib/2.0.2-python-3.6.2
```
3. Install missing libraries:
```
$ pip3 install --user torchtext
```
4. In the end run training
```
$ python3 main.py 
# if you get Cuda out of memory then check nvidia-smi
``` 

# Training time


# Results

