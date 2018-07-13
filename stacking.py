# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, TimeDistributed,\
MaxPooling1D,Flatten,Activation,Convolution1D
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
batch_size = 100
nb_epoch =20
epochs=20
hidden_dim = 120
nb_epoch_1=100

kernel_size = 3
nb_filter = 60

output_nu=3 #输出个数
n_fold = 5  # 交叉验证的折数

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

def create_model_cnn():
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
    output = Dense(output_nu, activation='softmax')(dense)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_gru():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = GRU(hidden_dim, recurrent_dropout=0.25) (embedded)  #0.7765237020316028
    output = Dense(output_nu, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_lstm():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = LSTM(hidden_dim, recurrent_dropout=0.25) (embedded)   #0.7945823927765236
    output = Dense(output_nu, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_2_gru():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)   #0.7844243792325056
    output = Dense(output_nu, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def create_model_2_lstm():
    sequence = Input(shape=(maxlen, ), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
                         weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)  #0.8058
    output = Dense(output_nu, activation='softmax') (hidden)
    model = Model(inputs=sequence, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

import numpy as np
from sklearn.cross_validation import KFold




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

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    # checkpointer = ModelCheckpoint(filepath="CNN.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

    kf = KFold(len(X_train), n_folds=n_fold)
    print(kf)
    from pandas.core.frame import DataFrame
    m=[create_model_cnn(),create_model_lstm(),create_model_gru(),\
                 create_model_2_lstm(),create_model_2_gru()]
    y_pred_train_all=[]
    y_pred_text_all=[]
    no=0
    for model in m:
        y_pred_train = []
        y_pred_text = []
        print(no)
        for train_index, dev_index in kf:
            # print(train_index.shape)
            # print(dev_index.shape)
            # print("TRAIN:", train_index, "DEV:", dev_index)
            no = no + 1
            x_train, Y_train = X_train[train_index], y_train[train_index]
            x_dev, Y_dev = X_train[dev_index], y_train[dev_index]
            checkpointer = ModelCheckpoint(filepath='./hdf5/CNN'+str(no)+'.hdf5', monitor='val_acc', \
                                           verbose=1, save_best_only=True)
            model.fit(x_train, Y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, \
                      epochs=nb_epoch,verbose=2,callbacks=[checkpointer])
            model = load_model('./hdf5/CNN'+str(no)+'.hdf5')
            # y_pred_train.append(np.argmax(model.predict(x_dev, batch_size=batch_size), axis=1))#下一层的训练集#
            y_pred_train.append(model.predict(x_dev, batch_size=batch_size))  # 下一层的训练集
            y_pred_text.append(model.predict(X_dev, batch_size=batch_size))  # 下一层的测试集

        y_pred_train_1 = np.mat(y_pred_train[0])
        for i in range(1, n_fold):
            y_pred_train1 = np.mat(y_pred_train[i])
            # print(y_pred_train1.shape)
            # print(y_pred_train1.shape)
            y_pred_train_1 = np.vstack((y_pred_train_1, y_pred_train1))  # 训练特征
        # print(y_pred_train_1.shape)
        y_pred_train_all.append(y_pred_train_1)     #将所有训练模型的训练结果存储
        # print(y_pred_train_1[0,0])
        y_pred_text_1 = np.mat(y_pred_text[0])
        for i in range(1, n_fold):
            y_pred_text1 = np.mat(y_pred_text[i])
            # print(y_pred_text1.shape)
            # print(y_pred_text1.shape)
            y_pred_text_1 = np.hstack((y_pred_text_1, y_pred_text1))  # 测试特征
        # print(y_pred_text_1.shape)
        y_pred_text_all.append(y_pred_text_1)#参数传递错误    将所有测试结果存储
        # print(y_pred_train_all, y_pred_text_all)

    y_pred_train_all_1 = np.mat(y_pred_train_all[0])
    for i in range(len(m)-1):
        y_pred_train_all1 = np.mat(y_pred_train_all[i])
        # print(y_pred_train1.shape)
        y_pred_train_all_1 = np.hstack((y_pred_train_all_1, y_pred_train_all1))  # 训练特征
    print(y_pred_train_all_1.shape)
    train_all=y_pred_train_all_1#  X*15   矩阵转换   形成X*3*5的形式    3是output_nu     5是折数



    y_pred_text_all_1 = np.mat(y_pred_text_all[0])
    for i in range(len(m)-1):
        y_pred_text_all1 = np.mat(y_pred_text_all[i])
        # print(y_pred_train1.shape)
        y_pred_text_all_1 = np.vstack((y_pred_text_all_1, y_pred_text_all1))  # 训练特征
    print(y_pred_text_all_1.shape)
    text_all_1=y_pred_text_all_1     #举证转换   形成X*3*5的形式    3是output_nu*5是折数
    # print(text_all_1)
    text_all_3=[]
    text_all=[]
    for j in range(len(text_all_1)):
        text_all_2 = []
        for i in range(3):
            text_all_2.append(np.mean((text_all_1[j,i], text_all_1[j,i + 1*output_nu], text_all_1[j,i + 2*output_nu], \
                                     text_all_1[j,i + 3*output_nu], text_all_1[j,i + 4*output_nu])))
        text_all_3 = np.mat(text_all_2)
        text_all.append(text_all_3)  #求测试集的平均值   并保存
        # print(text_all)
    # text_all = np.mat(text_all)
    # print(text_all.shape)
    # print(text_all)
    text_a = np.mat(text_all[0])
    for i in range(1, len(text_all_1)):
        text_a1 = np.mat(text_all[i])
        # print(y_pred_train1.shape)
        # print(y_pred_train1.shape)
        text_a = np.vstack((text_a, text_a1))
    print(text_a.shape)    #矩阵转换   形成X*3的形式    3是output_nu
    text_end_1=[]
    for i in range(len(m)):
        text_end_1.append(text_a[i*len(X_dev):(i+1)*len(X_dev),])
    # print(text_end_1)
    text_end = np.mat(text_end_1[0])
    for i in range(len(m)-1):
        text_end1 = np.mat(text_end_1[i])
        # print(y_pred_train1.shape)
        text_end = np.hstack((text_end, text_end1))  #矩阵转换   形成X*3*5的形式    3是output_nu   5是折数
    print(text_end.shape)    #一层数据处理    获取训练集和测试机的二阶特征
    print(len(m))
    print(type(train_all))

    input_layer = Input(shape=(len(m)*output_nu,))
    hidden = Dense(10, activation='softmax')(input_layer)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(3)(hidden)
    output = Activation('softmax')(hidden)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    checkpointer = ModelCheckpoint(filepath="./hdf5/end.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    model.fit(train_all, y_train, validation_data=[text_end, y_dev], batch_size=batch_size, epochs=nb_epoch_1,
              verbose=2, callbacks=[checkpointer])

    model = load_model('./hdf5/end.hdf5')
    y_pred = model.predict(text_end, batch_size=batch_size)
    print(type(y_pred))
    y_pred = np.argmax(y_pred, axis=1)
    from sklearn.metrics import precision_score, recall_score, f1_score

    f1score = f1_score(y_dev_test, y_pred, average='micro')
    print(f1score)



















