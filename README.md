# next-sentence-prediction

* [Python 3](https://www.python.org/downloads/)
* [PyTorch 1.0.1](https://pytorch.org/)

# Dataset
* [Download snli_1.0.zip (90.2 MB)](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) and decompress snli_1.0_train.txt, snli_1.0_dev.txt and snli_1.0_test.txt to __data/__
    * More information can be found at [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)

# Word Embeddings
* [Download glove.840B.300d.zip (2.0 GB)](http://nlp.stanford.edu/data/glove.840B.300d.zip) and decompress glove.840B.300d.txt to __$HOME/common/__
    * See [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

# Experiment
```

# Open xterm-LSF10-login-name and type:
$ k40sh or voltash
$ cd /work3/s180011/next_sentence_prediction/
# Then import necessary modules:
$ module load python3/3.6.2 numpy/1.13.1-python-3.6.2-openblas-0.2.20 matplotlib/2.0.2-python-3.6.2
$ pip3 install --user torchtext
# In the end run training
$ python3 main.py 
``` 

# Training time
* 156 minutes per training epoch w/ a NVIDIA Titan Xp GPU
* I plan to reduce the training time soon.

# Result
* Epoch 6
* Training loss: 0.361281, accuracy: 86.1% (mLSTM train accuracy: 92.0%)
* Dev loss: 0.392275, accuracy: 85.8% (mLSTM dev accuracy: 86.9%)
* Test loss: 0.397926, accuracy: 85.5% (mLSTM test accuracy: 86.1%)
