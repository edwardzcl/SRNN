# 导包
import collections
import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F
from spiking_rnn import*
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_imdb(folder='train', data_root="./aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])# 评论文本字符串和01标签
    random.shuffle(data)
    return data

DATA_ROOT = "./"
data_root = os.path.join(DATA_ROOT, "aclImdb")
train_data, test_data = read_imdb('train', data_root), read_imdb('test', data_root)

# 打印训练数据中的前五个sample
for sample in train_data[:5]:
    print(sample[1], '\t', sample[0][:50])


def get_tokenized_imdb(data):# 将每行数据的进行空格切割,保留每个的单词
    '''
    @params:
        data: 数据的列表，列表中的每个元素为 [文本字符串，0/1标签] 二元组
    @return: 切分词后的文本的列表，列表中的每个元素为切分后的词序列
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):
    '''
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    '''
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 统计所有的数据
    return Vocab.Vocab(counter, min_freq=5)# 构建词汇表,这里最小出现次数是5

vocab = get_vocab_imdb(train_data)
print('# words in vocab:', len(vocab))
# print(vocab[:5])


def preprocess_imdb(data, vocab):
    '''
    @params:
        data: 同上，原始的读入数据
        vocab: 训练集上生成的词典
    @return:
        features: 单词下标序列，形状为 (n, max_l) 的整数张量
        labels: 情感标签，形状为 (n,) 的0/1整数张量
    '''
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    # 填充,这里是将每一行数据扩充500个特征的
    labels = torch.tensor([score for _, score in data])
    return features, labels


train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))# 相当于将函数参数是函数结果
# *号语法糖,解绑参数 
# 上面的代码等价于下面的注释代码
# train_features, train_labels = preprocess_imdb(train_data, vocab)
# test_features, test_labels = preprocess_imdb(test_data, vocab)
# train_set = Data.TensorDataset(train_features, train_labels)
# test_set = Data.TensorDataset(test_features, test_labels)

# len(train_set) = features.shape[0] or labels.shape[0]
# train_set[index] = (features[index], labels[index])

batch_size = 100
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter))#391个批次,每个批次64个样本
# 这是对的


class BiLSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 100)# 映射长度,这里是降维度的作用
        self.fc1 = nn.Linear(100, embed_size) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        
        # encoder-decoder framework
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)# 双向循环网络
        self.decoder = nn.Linear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # 循环神经网络最后的隐藏状态可以用来表示一句话
    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.embedding(inputs.permute(1, 0)) # (seq_len, batch_size, d)500*64*100
        embeddings = self.fc1(embeddings)

        embeddings = torch.tanh(embeddings)
        #embeddings = (embeddings-embeddings.min())/(embeddings.max()-embeddings.min())
        # print(embeddings.shape)
        # rnn.LSTM 返回输出、隐藏状态和记忆单元，格式如 outputs, (h, c)
        outputs, _ = self.encoder(embeddings) # (seq_len, batch_size, 2*h)每一个输出,然后将第一次输出和最后一次输出拼接
        #print(outputs.shape)# 如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)
        encoding = torch.cat((outputs[0], outputs[-1]), -1) # (batch_size, 4*h)
        outs = self.decoder(encoding) # (batch_size, 2)
        return outs


class VanillaRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 100)# 映射长度,这里是降维度的作用
        self.fc1 = nn.Linear(100, embed_size) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        
        # encoder-decoder framework
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.RNN(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)# 双向循环网络
        self.decoder = nn.Linear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # 循环神经网络最后的隐藏状态可以用来表示一句话
    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.embedding(inputs.permute(1, 0)) # (seq_len, batch_size, d)500*64*100
        embeddings = self.fc1(embeddings)
        embeddings = torch.tanh(embeddings)
        #embeddings = (embeddings-embeddings.min())/(embeddings.max()-embeddings.min())
        # print(embeddings.shape)
        # rnn.LSTM 返回输出、隐藏状态和记忆单元，格式如 outputs, (h, c)
        outputs, _ = self.encoder(embeddings) # (seq_len, batch_size, 2*h)每一个输出,然后将第一次输出和最后一次输出拼接
        #print(outputs.shape)# 如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)
        encoding = torch.cat((outputs[0], outputs[-1]), -1) # (batch_size, 4*h)
        outs = self.decoder(encoding) # (batch_size, 2)
        return outs


class SRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(SRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 100)# 映射长度,这里是降维度的作用
        self.spiking_layer = Quantize_Spike(0,4)
        self.fc1 = nn.Linear(100, embed_size) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        #self.fc1 = QuantLinear(embed_size, embed_size)
        #self.fc2 = nn.Linear(embed_size, 4*num_hiddens) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.fc2 = QuantLinear(embed_size, 4*num_hiddens)
        #self.recu_fc2 = nn.Linear(4*num_hiddens, 4*num_hiddens)
        self.recu_fc2 = QuantLinear(4*num_hiddens, 4*num_hiddens)
        self.fc3 = nn.Linear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        #self.fc3 = QuantLinear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.recu_fc3 = nn.Linear(2, 2)
        #self.recu_fc3 = QuantLinear(2, 2)


    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        h2_mem = h2_spike = h2_sumspike = torch.zeros(len(inputs[:,0]), 4*num_hiddens, device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(len(inputs[:,0]), 2, device=device)
        num_sops = torch.zeros(1, device=device)

        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.embedding(inputs.permute(1, 0)) # (seq_len, batch_size, d)500*64*100
        embeddings = self.fc1(embeddings)
        embeddings = torch.relu(embeddings)
        embeddings = self.spiking_layer(embeddings)

        for seq_len in range(len(embeddings[:,0,0])): # simulation seq length
            #for step in range(time_window): # simulation time steps
                #x = embeddings[seq_len,:,:] > torch.rand(embeddings[seq_len,:,:].size(), device=device) # prob. firing
            x = embeddings[seq_len,:,:]
            h2_mem, h2_spike, sops = mem_update(self.fc2, self.recu_fc2, x.float(), h2_mem, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike
            num_sops = num_sops + sops / len(h2_sumspike)
            #h3_mem, h3_spike = mem_update(self.fc3, self.recu_fc3, h2_spike, h3_mem, h3_spike)
            #h3_sumspike = h3_sumspike + h3_spike


        #outs = h3_sumspike / time_window
        outs = self.fc3(h2_sumspike)
        num_spikes = torch.sum(h2_sumspike) / len(h2_sumspike)

        return outs, num_spikes, num_sops


embed_size, num_hiddens, num_layers = 288, 64, 1
time_window = 20
#net = BiLSTM(vocab, embed_size, num_hiddens, num_layers)
#net = VanillaRNN(vocab, embed_size, num_hiddens, num_layers)
net = SRNN(vocab, embed_size, num_hiddens, num_layers)
print(net)

#cache_dir = "F:\python\scnn-master\glove6B"
cache_dir = "./glove6B"
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=cache_dir)

def load_pretrained_embedding(words, pretrained_vocab):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
    @return:
        embed: 加载到的词向量
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为len*100维度
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]# 将每个词语用训练的语言模型理解
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    # print(embed.shape),在词典中寻找相匹配的词向量
    #embed = (embed -embed.min()) / (embed.max() - embed.min())
    return embed

net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = True # 直接加载预训练好的, 所以不需要更新它


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in net.parameters())
print('total parameters:', total_params)
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('training parameters:', total_trainable_params)

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                outs, num_spikes, num_sops = net(X.to(device))
                acc_sum += (outs.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n, num_spikes, num_sops

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat, num_spikes_train, num_sops_train = net(X)
            l = loss(y_hat, y) # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()# 优化方法
            train_l_sum += l.cpu().item()# 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc, num_spikes_test, num_sops_test = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        print("number of spikes per sample in this test batch: ",  num_spikes_test)
        print("number of sops per sample in this test batch: ",  num_sops_test)      

            #X = X.to(device)
            #y = torch.zeros(batch_size, 2).scatter_(1, y.view(-1, 1), 1)
            #y = y.to(device)
            #y_hat = net(X)
            #l = loss(y_hat, y) # 交叉熵损失函数
            #optimizer.zero_grad()
            #l.backward()
            #optimizer.step()# 优化方法
            #train_l_sum += l.cpu().item()# 进入cpu中统计
            #_, predicted = y_hat.cpu().max(1)
            #train_acc_sum += float(predicted.eq(y).sum().item())
            #n += y.shape[0]
            #batch_count += 1


lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
#loss = nn.MSELoss()

train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

def predict_sentiment(net, vocab, sentence):
    '''
    @params：
        net: 训练好的模型
        vocab: 在该数据集上创建的词典，用于将给定的单词序转换为单词下标的序列，从而输入模型
        sentence: 需要分析情感的文本，以单词序列的形式给出
    @return: 预测的结果，positive 为正面情绪文本，negative 为负面情绪文本
    '''
    device = list(net.parameters())[0].device # 读取模型所在的环境
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    object_class = net(sentence.view((1, -1)))
    label = torch.argmax(object_class[0], dim=1)# 这里输入之后,进入embedding,进入lstm,进入全连接层,输出结果
    return 'positive' if label.item() == 1 else 'negative'

print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))

query_num = 0
while query_num<=100:
    #用户输入，操作
    print("python 用户输入操作")
 
    # input(提示字符串)，函数阻塞程序，并提醒用户输入字符串
    instr = input("please input the string: ")
    print("your input >> " + instr)
    instr = instr.split()
    print("This query is evaluated as: ", predict_sentiment(net, vocab, instr))
    query_num +=1
