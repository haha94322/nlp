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
batch_size = 100
nb_epoch =20
hidden_dim = 120

kernel_size = 3
nb_filter = 60
maxlen=246

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

def create_model_cnn(max_features,num_features,W):
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    convolution = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid', activation='relu',\
                                strides=1,dilation_rate=1)(embedded)
    maxpooling = MaxPooling1D(pool_size=2)(convolution)
    maxpooling = Flatten()(maxpooling)
    # We add a vanilla hidden layer:
    dense = Dense(70)(maxpooling)  # best: 120
    dense = Dropout(0.25)(dense)  # best: 0.25
    dense = Activation('relu')(dense)
    output = Dense(output_repre_dim, activation='softmax')(dense)
    # model = Model(inputs=sequence, outputs=output)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #
    # return model
    return sequence,output


def create_model_gru():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = GRU(hidden_dim, recurrent_dropout=0.25) (embedded)  #0.7765237020316028
    output = Dense(3, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_lstm():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = LSTM(hidden_dim, recurrent_dropout=0.25) (embedded)   #0.7945823927765236
    output = Dense(3, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_2_gru():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)   #0.7844243792325056
    output = Dense(3, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_2_lstm():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)  #0.8058
    output = Dense(3, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def m():
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

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    return X_train, X_test, X_dev, y_train, y_dev, y_dev_test, max_features,num_features,W

import numpy as np
from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

cnn_module_num = 20
filters = 5
kernel_size = 3  # 3-gram
output_repre_dim = 3  # 200

# lstm
time_step = cnn_module_num
lstm_hidden_dim = 256

# mlp
hidden_dim_1 = 20
output_dim = 3

# bgmlp
input_fea_dim = 20
bg_hidden_dim = 20  # 50


if __name__ == '__main__':
    X_train, X_test, X_dev, y_train, y_dev, y_dev_test, max_features,num_features,W=m()
    import_all=[]
    #CNN
    sequence1 = Input(shape=(maxlen,), dtype='int32', name="CNN")
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, \
                         weights=[W], trainable=False)(sequence1)
    embedded = Dropout(0.25)(embedded)
    convolution = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid', activation='relu', \
                                strides=1, dilation_rate=1)(embedded)
    maxpooling = MaxPooling1D(pool_size=2)(convolution)
    maxpooling = Flatten()(maxpooling)
    # We add a vanilla hidden layer:
    dense = Dense(70)(maxpooling)  # best: 120
    dense = Dropout(0.25)(dense)  # best: 0.25
    dense = Activation('relu')(dense)
    output1 = Dense(output_repre_dim, activation='softmax')(dense)
    #LSTM
    sequence2 = Input(shape=(maxlen,), dtype='int32', name="LSTM")
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True, \
                         weights=[W], trainable=False)(sequence2)
    embedded = Dropout(0.25)(embedded)
    hidden = LSTM(hidden_dim, recurrent_dropout=0.25)(embedded)  # 0.7945823927765236
    output2 = Dense(output_repre_dim, activation='softmax')(hidden)
    # #GRU
    sequence3 = Input(shape=(maxlen, ), dtype='int32', name="GRU")
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence3)
    embedded = Dropout(0.25) (embedded)
    hidden = GRU(hidden_dim, recurrent_dropout=0.25) (embedded)  #0.7765237020316028
    output3 = Dense(output_repre_dim, activation='softmax') (hidden)
    # #2-LSTM
    sequence4 = Input(shape=(maxlen, ), dtype='int32', name="2-LSTM")
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence4)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)  #0.8058
    output4 = Dense(output_repre_dim, activation='softmax') (hidden)
    # #2-GRU
    sequence5 = Input(shape=(maxlen, ), dtype='int32', name="2-GRU")
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence5)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)   #0.7844243792325056
    output5 = Dense(output_repre_dim, activation='softmax') (hidden)

    hidden = keras.layers.concatenate([output1, output2,output3,output4,output5])
    # merged1 = merge([output1, output2,output3,output4,output5])  # , concat_axis=1)
    # print(merged1.shape)
    # reshaped = Reshape((output_repre_dim//3,3 ),\
    #                    input_shape=(output_repre_dim ,))(merged1)
    # hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(reshaped)
    hidden = Dense(10, activation='softmax')(hidden)
    # hidden = Dense(1)(hidden)
    hidden = Dropout(0.25)(hidden)
    # hidden = Activation('relu')(hidden)
    hidden = Dense(3)(hidden)
    output = Activation('softmax')(hidden)


    model = Model(inputs=[sequence1,sequence2,sequence3,sequence4,sequence5], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    checkpointer = ModelCheckpoint(filepath="CNN.hdf5", monitor='val_acc', verbose=1, save_best_only=True)


    model.fit({'CNN': X_train, 'LSTM': X_train ,'GRU': X_train,'2-LSTM': X_train,'2-GRU': X_train},\
              y_train, validation_data=[{'CNN': X_dev ,'LSTM': X_dev,'GRU': X_dev,'2-LSTM': X_dev,'2-GRU': X_dev}, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2, callbacks=[checkpointer])
    model = load_model('CNN.hdf5')
    y_pred = model.predict(X_dev, batch_size=batch_size)
    print(type(y_pred))

    y_pred = np.argmax(y_pred, axis=1)


    from sklearn.metrics import precision_score, recall_score, f1_score

    f1score = f1_score(y_dev_test, y_pred, average='micro')
    print(f1score)   #0.80587






