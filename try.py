import jieba
import torch
dic = {'越女':1, '采莲':3, '秋水':5, '畔':7}
input_para = '越女采莲秋水畔'
input_words = jieba.lcut(input_para)
print(input_words)
input_len = len(input_words)
input_lst = []
for input_word in input_words:
    lst = [dic[input_word]]
    input_lst.append(lst)
_input = torch.Tensor(input_lst)
print(_input)
article = str()
article = ''.join(input_para)
print(article)