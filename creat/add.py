import numpy as np
import pandas as pd

path="C:/Users/O邪恶的小胖哥O/Desktop/veb/train_add/train_add.txt"
f1 = open(path, 'r', encoding='utf8')
train = pd.read_table(f1, header=None, sep="\t", quoting=3, error_bad_lines=False)
f2 = open("add.txt", 'r', encoding='utf8')
add = pd.read_table(f2, header=None, sep="\t", quoting=3, error_bad_lines=False)
train=train.append(add)
train = np.mat(train)
t=1
test_train_id=[]
test_train_sent=[]
test_train_y=[]
for q in range(len(train)):
    t += 1
    test_train_id.append(train[q, 0])
    test_train_sent.append(train[q, 1])
    test_train_y.append(train[q, 2])

file_out_name= path
file_out = open(file_out_name, 'w', encoding='utf8')
for i in range(len(test_train_id)):
    file_out.write(str(test_train_id[i])+'\t'+test_train_sent[i]+'\t'+str(test_train_y[i])+'\n')
file_out.close()
