
from xml.dom import minidom
import jieba
file_name = "隐喻情感_train.xml"
file_out_name= "隐喻情感_train.txt"
file_out = open(file_out_name, 'w', encoding='utf8')
dom = minidom.parse(file_name)
root = dom.documentElement
case = root.getElementsByTagName('metaphor')
for t in case:
	ID = t.getElementsByTagName('ID')[0].childNodes[0].data
	Sentence = t.getElementsByTagName('Sentence')[0].childNodes[0].data
	ans = jieba.cut(Sentence)
	for word in ans:
		print(word,' ',end="")
	print()
	Emo_Class = t.getElementsByTagName('Emo_Class')[0].childNodes[0].data
	#print(ID,Sentence,Label)
	file_out.write(ID+"\t"+Sentence+"\t"+Emo_Class+"\n")

file_out.close()