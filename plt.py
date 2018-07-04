from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, \
    TimeDistributed, Convolution1D, \
    MaxPooling1D, Flatten, Activation, Reshape, merge
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

maxlen = 100
batch_size = 200
nb_epoch = 100
hidden_dim = 120

kernel_size = 3
nb_filter = 60

cnn_module_num = 20
filters = 5
kernel_size = 3  # 3-gram
output_repre_dim = 240  # 200

# lstm
time_step = cnn_module_num
lstm_hidden_dim = 256


# f2 = open("隐喻动词_train.txt", 'r', encoding='utf8')
# test = pd.read_table(f2, header=None, sep="\t", quoting=3, error_bad_lines=False)


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev, y_dev_test = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']
        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    y_dev_test = y_dev
    # X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    # X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    # X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev, y_dev_test]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test2.pickle3')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    X_train, X_test, X_dev, y_train, y_dev, y_dev_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)


    avg_len=list(map(len,X_train))
    print(np.mean(avg_len))

    import matplotlib.pyplot as plt
    plt.hist(avg_len,bins=range(min(avg_len),max(avg_len)+1,1))
    plt.show()
