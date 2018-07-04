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
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, TimeDistributed,Convolution1D,\
MaxPooling1D,Flatten,Activation, Reshape,merge
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

maxlen = 100
batch_size = 200
nb_epoch =100
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
    print(maxlen)
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

    revs, W, word_idx_map, vocab, maxlen1 = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    # print(revs)#sentence
    # print(W)#lable
    # print(vocab)
    # print(maxlen)#max length
    # print(word_idx_map)#word code
    X_train, X_test, X_dev, y_train, y_dev, y_dev_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)
    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)
    print(max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)
    print(num_features)


    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen, ), dtype='int32')

    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,\
    #                      weights=[W], trainable=False) (sequence)
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,\
                         weights=[W], trainable=False) (sequence)    #3138
    embedded = Dropout(0.25) (embedded)

    #CNN
    # convolution = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid', activation='relu', strides=1, \
    #                             dilation_rate=1)(embedded)
    # maxpooling = MaxPooling1D(pool_size=2)(convolution)
    # maxpooling = Flatten()(maxpooling)
    # # We add a vanilla hidden layer:
    # dense = Dense(70)(maxpooling)  # best: 120
    # dense = Dropout(0.25)(dense)  # best: 0.25
    # dense = Activation('relu')(dense)
    # output = Dense(output_repre_dim, activation='softmax')(dense)
    # print(output.shape)
    # reshaped = Reshape((output_repre_dim//3 ,3 ), \
    #                    input_shape=(output_repre_dim ,))(output)
    # # biLSTM = Bidirectional(LSTM(lstm_hidden_dim, dropout=0.5))(reshaped)
    # hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(reshaped)


    # LSTM
    # hidden = LSTM(hidden_dim, recurrent_dropout=0.25,return_sequences=True) (embedded)
    hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)  #0.8058
    output = Dense(output_repre_dim, activation='softmax')(hidden)
    reshaped = Reshape((output_repre_dim//3 ,3 ), \
                       input_shape=(output_repre_dim ,))(output)
    convolution = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid', activation='relu', strides=1, \
                                dilation_rate=1)(reshaped)
    maxpooling = MaxPooling1D(pool_size=2)(convolution)
    maxpooling = Flatten()(maxpooling)
    # We add a vanilla hidden layer:
    dense = Dense(70)(maxpooling)  # best: 120
    dense = Dropout(0.25)(dense)  # best: 0.25
    dense = Activation('relu')(dense)

    # hidden = LSTM(hidden_dim, recurrent_dropout=0.25) (embedded)   #0.7945823927765236

    # GRU
    # hidden = GRU(hidden_dim, recurrent_dropout=0.25) (embedded)  #0.7765237020316028
    # hidden = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25))(embedded)   #0.7844243792325056
    # output = Dense(3, activation='softmax')(dense)
    output = Dense(3, activation='softmax') (dense)
    model = Model(inputs=sequence, outputs=output)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    checkpointer = ModelCheckpoint(filepath="CNN.hdf5",monitor='val_acc', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2, callbacks=[checkpointer])

    model = load_model('CNN.hdf5')
    # y_pred = model.predict(X_dev, batch_size=batch_size)
    y_pred = model.predict(X_dev, batch_size=batch_size)
    print(type(y_pred))
    # print(y_pred.shape) #为了输出
    # y_pred=y_pred.tolist()
    # y_pred = np.mat(y_pred)
    # print(y_pred[1,0])
    y_pred = np.argmax(y_pred, axis=1)
    # print(y_pred)
    # y_pred= np_utils.to_categorical(np.array(y_pred))
    # print(y_pred)
    # print(y_dev_test)

    from sklearn.metrics import precision_score, recall_score, f1_score

    f1score = f1_score(y_dev_test, y_pred, average='micro')
    print(f1score)

    # result_output = pd.DataFrame(data={"id": test[0], "Label": y_pred})
    #
    # # Use pandas to write the comma-separated output file
    # # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)
    #
    # result_output.to_csv("lstm11.csv", index=False, quoting=3)

    # from read import id,list
    # file_out_name = "test_train.txt"
    # file_out = open(file_out_name, 'w', encoding='utf8')
    # for i in range(len(list)):
    #     file_out.write(str(id[i]) + '\t' + list[i] + '\t'+str(y_pred[i,0])+'\t'+str(y_pred[i,1])+'\t'+str(y_pred[i,2])+'\n')
    # file_out.close()