# -*- coding:utf-8 -*-
# list 可以直接存为txt
import numpy as np
path = 'C://Users/O邪恶的小胖哥O/Desktop/veb/txt/2.txt'
f_stop = open(path)
try:
    f_stop_text = f_stop.read()
finally:
    f_stop.close()
f_stop_seg_list = f_stop_text.split('\n')# “ ”和空格需要去掉


lines1 = []
lines2 = []
lines3 = []
lines4 = []
train =[]
f1 = open(path,'r')
lines3 = f1.read()  # 全部读出

lines2 = lines3.split(u'。')  # 逗号分隔
# print(lines2)
# print(len(lines2))
list8=[]
list=[]

for q in lines2:
    lines4=q.split(u'？')
    for p in lines4:
        if len(p) > 8:
            list8.append(p.strip())#？号分割


for q in list8:
    lines1=q.split(u'\n')#分割
    for p in lines1:
        if len(p) > 16:
            list.append(p.strip())
# print(list)
# for q in list8:
#     lines1=q.split(u'！')
#     for p in lines1:
#         if len(p) > 8:
#             list.append(p.strip())
list_length = len(list)
print(list_length)
# print(type(list))

# list= np.mat(list)
# list = np.transpose(list)
# print(list)
# print(type(list))
id=[]
for i in range(list_length):
    id.append(i)

# id= np.mat(id)
# id = np.transpose(id)


# test=np.hstack((id,list))
# print(test)
file_out_name= "test1.txt"
file_out = open(file_out_name, 'w', encoding='utf8')
for i in range(len(list)):
    file_out.write(str(id[i])+'\t'+list[i]+'\n')
file_out.close()






