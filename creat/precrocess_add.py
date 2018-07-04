from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle
import jieba
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict

# Read data from files
f1 = open("隐喻动词_train.txt", 'r', encoding='utf8')
train = pd.read_table(f1, header=None, sep="\t", quoting=3, error_bad_lines=False)
f2 = open("test1.txt", 'r', encoding='utf8')
test = pd.read_table(f2, header=None, sep="\t", quoting=3, error_bad_lines=False)
path="C:/Users/O邪恶的小胖哥O/Desktop/veb/train_add/train_add.txt"
f3 = open(path, 'r', encoding='utf8')
train_add = pd.read_table(f3, header=None, sep="\t", quoting=3, error_bad_lines=False)
# f2 = open("test.txt", 'r', encoding='utf8')
# test = pd.read_table(f2, header=None, sep="\t", quoting=3, error_bad_lines=False)
print(type(train))
print(test.shape)


def build_data_train_test(data_train, data_test, data_add ,train_ratio=0.90):
# def build_data_train_test(data_train, data_test ,train_ratio=0.9):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in range(len(data_train)):
        rev = data_train[i]
        y = int(train[2][i])
        orig_rev = ' '.join(rev)
        w = jieba.cut(orig_rev)
        words = ' '.join(w)
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)

    for i in range(len(data_add)):
        rev = data_add[i]
        y = int(train_add[2][i])
        orig_rev = ' '.join(rev)
        w = jieba.cut(orig_rev)
        words = ' '.join(w)
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 1}
        revs.append(datum)

    for i in range(len(data_test)):
        rev = data_test[i]
        orig_rev = ' '.join(rev)
        w = jieba.cut(orig_rev)
        words = ' '.join(w)
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab

def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % (unk_words))
    return word_vecs

def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
    W[0] = np.zeros((k, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))



    revs, vocab = build_data_train_test(train[1], test[1], train_add[1])
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))

    # word2vec GoogleNews
    # model_file = os.path.join('vector', 'GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # Glove Common Crawl
    model_file = os.path.join('vector', 'news12g_bdbk20g_nov90g_dim128.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    w2v = load_bin_vec(model, vocab)#词向量标记
    logging.info('word embeddings loaded!')
    logging.info('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    logging.info('extracted index from embeddings! ')


    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test2.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')
