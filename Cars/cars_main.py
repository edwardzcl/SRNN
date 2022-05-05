import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import jieba
import os
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
import time
from spiking_rnn import*

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分词
def tokenizer(text): 
    return [word for word in jieba.lcut(text) if word not in stop_words]

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt',encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

stop_words = get_stop_words()  # 加载停用词表    


text = data.Field(sequential=True,
                  lower=True,
                  tokenize=tokenizer,
                  stop_words=stop_words)
label = data.Field(sequential=False)


train, val = data.TabularDataset.splits(
    path='data/',
    skip_header=True,
    train='train.tsv',
    validation='validation.tsv',
    format='tsv',
    fields=[('index', None), ('label', label), ('text', text)],
)

print(train[2].text)
print(train[5].__dict__.keys())


#加载Google训练的词向量
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('data/myvector.vector', binary=False)


cache = 'data/.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(name='data/myvector.vector', cache=cache)
# 指定Vector缺失值的初始化方式，没有命中的token的初始化方式
#vectors.unk_init = nn.init.xavier_uniform_

text.build_vocab(train, val, vectors=vectors)#加入测试集的vertor

#text.build_vocab(train, val, vectors=Vectors(name='data/myvector.vector'))#加入测试集的vertor
label.build_vocab(train, val)

embedding_dim = text.vocab.vectors.size()[-1]
vectors = text.vocab.vectors

text.vocab.freqs.most_common(10)
print(text.vocab.vectors.shape)

batch_size=100

train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)), # 训练集设置batch_size,验证集整个集合用于测试
    )

vocab_size = len(text.vocab)
label_num = len(label.vocab)

batch = next(iter(train_iter))
data = batch.text
print(batch.text.shape)
print(batch.text)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers):
        super(BiLSTM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, 100)  # embedding之后的shape: torch.Size([200, 8, 300])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.fc1 = nn.Linear(100, embedding_dim) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(2*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(batch_size, seq_len)
        # 再提取词特征，输出形状为(batch_size, seq_len, embedding_dim)
        embeddings = self.word_embeddings(inputs)
        embeddings = self.fc1(embeddings)
        embeddings = torch.tanh(embeddings)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # outputs形状是(batch_size, seq_len, 2 * num_hiddens)
        #outputs = torch.cat((outputs[:, 0, :], outputs[:, -1, :]), -1) # (batch_size, 4*h)
        outs = self.decoder(outputs[:, -1, :])
        #outs = self.decoder(outputs)
        return outs


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers):
        super(VanillaRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, 100)  # embedding之后的shape: torch.Size([200, 8, 300])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.fc1 = nn.Linear(100, embedding_dim) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.RNN(input_size=embedding_dim, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(batch_size, seq_len)
        # 再提取词特征，输出形状为(batch_size, seq_len, embedding_dim)
        embeddings = self.word_embeddings(inputs)
        embeddings = self.fc1(embeddings)
        embeddings = torch.tanh(embeddings)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # outputs形状是(batch_size, seq_len, 2 * num_hiddens)
        outputs = torch.cat((outputs[:, 0, :], outputs[:, -1, :]), -1) # (batch_size, 4*h)
        #outs = self.decoder(outputs[:, -1, :])
        outs = self.decoder(outputs)
        return outs


class SRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers):
        super(SRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, 100)  # embedding之后的shape: torch.Size([200, 8, 300])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.spiking_layer = Quantize_Spike(0,4)

        self.fc1 = nn.Linear(100, embedding_dim) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        #self.fc1 = QuantLinear(embedding_dim, embedding_dim)
        #self.fc2 = nn.Linear(embedding_dim, 2*num_hiddens) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.fc2 = QuantLinear(embedding_dim, 2*num_hiddens)
        #self.recu_fc2 = nn.Linear(2*num_hiddens, 2*num_hiddens)
        self.recu_fc2 = QuantLinear(2*num_hiddens, 2*num_hiddens)
        self.fc3 = nn.Linear(2*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        #self.fc3 = QuantLinear(2*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.recu_fc3 = nn.Linear(2, 2)
        #self.recu_fc3 = QuantLinear(2, 2)

    def forward(self, inputs):
        h2_mem = h2_spike = h2_sumspike = torch.zeros(len(inputs[:,0]), 2*num_hiddens)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(len(inputs[:,0]), 2)

        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.word_embeddings(inputs) # (seq_len, batch_size, d)500*64*100
        embeddings = self.fc1(embeddings)
        embeddings = torch.relu(embeddings)
        embeddings = self.spiking_layer(embeddings)

        for seq_len in range(len(embeddings[0,:,0])): # simulation seq length
            #for step in range(time_window): # simulation time steps
                #x = embeddings[:,seq_len,:] > torch.rand(embeddings[:,seq_len,:].size()) # prob. firing
            x = embeddings[:,seq_len,:]

            h2_mem, h2_spike = mem_update(self.fc2, self.recu_fc2, x.float(), h2_mem, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike

            #h3_mem, h3_spike = mem_update(self.fc3, self.recu_fc3, h2_spike, h3_mem, h3_spike)
            #h3_sumspike = h3_sumspike + h3_spike

        #outs = h3_sumspike / time_window*len(embeddings[0,:,0])
        outs = self.fc3(h2_sumspike)

        #outs = self.decoder(outputs[:, -1, :])
        return outs


time_window = 20
embedding_dim, num_hiddens, num_layers = 144, 64, 1
#net = BiLSTM(vocab_size, embedding_dim, num_hiddens, num_layers)
#net = VanillaRNN(vocab_size, embedding_dim, num_hiddens, num_layers)
net = SRNN(vocab_size, embedding_dim, num_hiddens, num_layers)
print(net)     


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in net.parameters())
print('total parameters:', total_params)
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('training parameters:', total_trainable_params)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  #X转置 y为啥要减1
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, batch in enumerate(train_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  #X转置 y为啥要减1
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))

lr, num_epochs = 0.01, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
#loss = nn.MSELoss()
train(train_iter, val_iter, net, loss, optimizer, num_epochs)                   