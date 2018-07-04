import numpy as np
import pandas as pd
f1 = open("test_train.txt", 'r', encoding='utf8')
train = pd.read_table(f1, header=None, sep="\t", quoting=3, error_bad_lines=False)
print(train.shape)
print(len(train))
print(type(train))
train = np.mat(train)
print(type(train))
print((train[1,4]))
t=1
test_train_id=[]
test_train_sent=[]
test_train_y=[]
for q in range(len(train)):
    if train[q,2]>0.5:
        t+=1
        test_train_id.append(train[q, 0])
        test_train_sent.append(train[q, 1])
        test_train_y.append(0)
    # elif train[q,3]>0.95:
    #     t+=1
    #     test_train_id.append(train[q, 0])
    #     test_train_sent.append(train[q, 1])
    #     test_train_y.append(1)
    # elif train[q,4]>0.95:
    #     t+=1
    #     test_train_id.append(train[q, 0])
    #     test_train_sent.append(train[q, 1])
    #     test_train_y.append(2)

print(t)
# test_train = np.mat(test_train)#需要转化为矩阵的模式
# test_train = np.mat(test_train)
# test_train = np.mat(test_train)
# print(test_train.shape)
# file_out_name= "train_add.txt"
file_out_name= "add.txt"
file_out = open(file_out_name, 'w', encoding='utf8')
for i in range(len(test_train_id)):
    file_out.write(str(test_train_id[i])+'\t'+test_train_sent[i]+'\t'+str(test_train_y[i])+'\n')
file_out.close()
