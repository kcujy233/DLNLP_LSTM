import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx#以字为键
            self.idx2word[self.idx] = word#以数值为键
            self.idx += 1

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()#继承类，初始化映射表

    def get_data(self, path, batch_size=20):#读取文件，导入映射表
        # step 1
        with open(path, 'r', encoding="ANSI") as f:
            tokens = 0
            for line in f.readlines():
                #把一些无意义的空格、段落符给去掉

                line = line.replace(' ', '')
                line = line.replace('\u3000', '')
                line = line.replace('\t', '')
                #jieba
                words = jieba.lcut(line)
                # words = jieba.lcut(line) + ['<eos>']#相当于学习一下回车在哪
                tokens += len(words)
                for word in words:#构造彼此映射的关系
                    self.dictionary.add_word(word)
        # step 2
        ids = torch.LongTensor(tokens)#实例化一个LongTensor，命名为ids。遍历全部文本，根据映射表把单词转成索引，存入ids里
        token = 0
        with open(path, 'r', encoding="ANSI") as f:
            for line in f.readlines():
                line = line.replace(' ', '')
                line = line.replace('\u3000', '')
                line = line.replace('\t', '')
                words = jieba.lcut(line)
                # words = jieba.lcut(line) + ['<eos>']#相当于学习一下回车在哪
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]#把每个词对应的索引存在ids里
                    token += 1
        # step 3 根据batchsize重构成一个矩阵
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)#单词总数，每个单词的特征个数
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)#单词特征数，隐藏节点数，隐藏层数
        self.linear = nn.Linear(hidden_size, vocab_size)#全连接层

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)
    
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
batch_size = 50
seq_length = 30#序列长度，我认为是与前多少个词具有相关程度
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus = Corpus()#构造实例
ids = corpus.get_data('笑傲江湖.txt', batch_size)#获得数据
vocab_size = len(corpus.dictionary)#词总数

model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)

cost = nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))#参数矩阵初始化(h,c)

    for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):#打印循环中的进度条
        inputs = ids[:, i:i+seq_length].to(device)#训练集的输入
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)#训练集的结果

        states = [state.detach() for state in states]
        #detach返回一个新的tensor，相当于可以切断反向传播的计算
        outputs, states = model(inputs, states)
        loss = cost(outputs, targets.reshape(-1))

        model.zero_grad()#梯度清零
        loss.backward()#反向传播
        clip_grad_norm_(model.parameters(), 0.5)#避免梯度爆炸
        optimizer.step()

num_samples = 300#生成文本的长度，可以认为是包含单词的个数
article = str()#输出文本的容器

'''选择1个随即单词的输入'''
state = (torch.zeros(num_layers, 1, hidden_size).to(device),
        torch.zeros(num_layers, 1, hidden_size).to(device))#初始化参数
prob = torch.ones(vocab_size)#对应模型中的outputs，相当于单词的概率分布
#在字典中随机抽样单词作为开头
_input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
for i in range(num_samples):
    output, state = model(_input, state)
    prob = output.exp()
    word_id = torch.multinomial(prob, num_samples=1).item()

    _input.fill_(word_id)
    word = corpus.dictionary.idx2word[word_id]
    word = '\n' if word == '<eos>' else word
    article += word
print(article)

'''选择10个随即单词的输入'''
state = (torch.zeros(num_layers, 10, hidden_size).to(device),
        torch.zeros(num_layers, 10, hidden_size).to(device))#初始化参数
prob = torch.ones(vocab_size)#对应模型中的outputs，相当于单词的概率分布
#在字典中随机抽样单词作为开头
_input = torch.multinomial(prob, num_samples=10).unsqueeze(1).to(device)
for i in range(num_samples):
    output, state = model(_input, state)
    prob = output.exp()

    # word_id = torch.multinomial(prob, num_samples=1).item()
    word_id = torch.multinomial(prob, num_samples=1)
    _input = word_id
    # _input.fill_(word_id)
    # word = corpus.dictionary.idx2word[word_id]
    # word = '\n' if word == '<eos>' else word
    # article += word

    for j in word_id:
        word_value = j.item()
        word = corpus.dictionary.idx2word[word_value]
        word = '\n' if word == '<eos>' else word
        article += word
print(article)