# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import keras

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, TimeDistributed,\
MaxPooling1D,Flatten,Activation,Convolution1D, Reshape,merge
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.model_selection import KFold
batch_size = 100
nb_epoch =20
hidden_dim = 120

kernel_size = 3
nb_filter = 60
maxlen=246

cnn_module_num = 20
filters = 5
kernel_size = 3  # 3-gram
output_repre_dim = 120  # 200

# lstm
time_step = cnn_module_num
lstm_hidden_dim = 256

# mlp
hidden_dim_1 = 20
output_dim = 3

# bgmlp
input_fea_dim = 20
bg_hidden_dim = 20  # 50


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
    X_train, X_test, X_dev, y_train, y_dev ,y_dev_test= [], [], [], [], [],[]
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

    y_dev_test=y_dev
    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
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
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    pickle_file = os.path.join('pickle', 'imdb_train_val_test2.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev, y_dev_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)
    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)



    main_input = Input(shape=(246,), dtype='int32', name='main_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(32)(x)
    # auxiliary_output = Dense(10, activation='sigmoid', name='aux_output')(lstm_out)
    # 额外的输入数据
    auxiliary_input = Input(shape=(246,), name='aux_input')
    '''
    #将LSTM得到的张量与额外输入的数据串联起来，这里是横向连接'''
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    # 建立一个深层连接的网络
    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_output = Dense(3, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1.])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # model.fit([X_train, X_train], [y_train], epochs=10, batch_size=128,verbose=2)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2)







