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


def read_imdb(folder='train', data_root="F:\python\scnn-master\aclImdb"):
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
    # # 手动过滤低频词
    # filtered_counter = {token: freq for token, freq in counter.items() if freq >= 5}
    # return Vocab.Vocab(filtered_counter)

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




class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        '''
        @params:
            x: 输入，形状为 (batch_size, n_channels, seq_len) 的张量
        @return: 时序最大池化后的结果，形状为 (batch_size, n_channels, 1) 的张量
        '''
        return F.max_pool1d(x, kernel_size=x.shape[2]) # kenerl_size=seq_len


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            kernel_sizes: 卷积核大小列表:文本上的卷积神经网络通常都会采用不同的卷积核,用来采集不同的尺度信息
            num_channels: 卷积通道数列表
        '''
        super(TextCNN, self).__init__()
        #  因为有比较多的词都是没有的,使用不变的话,他们就都是0了,所以分为两层
        self.embedding = nn.Embedding(len(vocab), 100) # 参与训练的嵌入层
        #self.constant_embedding = nn.Embedding(len(vocab), embed_size) # 不参与训练的嵌入层
        self.fc1 = nn.Linear(100, embed_size)
        
        self.pool = GlobalMaxPool1d() # 时序最大池化层没有权重，所以可以共用一个实例
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        # 并没有使用python自带的model的包,不会让梯度传播到这些成员变量上
        #for c, k in zip(num_channels, kernel_sizes):
        #    self.convs.append(nn.Conv1d(in_channels = 2*embed_size, 
        #                                out_channels = c, 
        #                                kernel_size = k))# 因为使用了两层嵌入层

        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embed_size, 
                                        out_channels = c, 
                                        kernel_size = k))# 因为使用了两层嵌入层    
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.dropout = nn.Dropout(0.5) # 丢弃层用于防止过拟合

    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outputs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        #embeddings = torch.cat((
        #    self.embedding(inputs), 
        #    self.constant_embedding(inputs)), dim=2) # (batch_size, seq_len, 2*embed_size)64*500*100
        embeddings = self.embedding(inputs) # (batch_size, seq_len, 2*embed_size)64*500*100
        embeddings = self.fc1(embeddings)
        # 根据一维卷积层要求的输入格式，需要将张量进行转置
        embeddings = embeddings.permute(0, 2, 1) # (batch_size, 2*embed_size, seq_len)# 64*100*500
        # 嵌入层计算
        # 卷积层
        encoding = torch.cat([
            self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # encoding = []
        # for conv in self.convs:
        #     out = conv(embeddings) # (batch_size, out_channels, seq_len-kernel_size+1)
        #     out = self.pool(F.relu(out)) # (batch_size, out_channels, 1)
        #     encoding.append(out.squeeze(-1)) # (batch_size, out_channels)
        # encoding = torch.cat(encoding) # (batch_size, out_channels_sum)
        
        # 应用丢弃法后使用全连接层得到输出
        # print(encoding.shape)64*30,3个卷积核,每个卷积核输出通道是100
        outputs = self.decoder(self.dropout(encoding))#0.5的可能性归0
        return outputs






embed_size, kernel_sizes, num_channels = 288, [3, 4, 5], [80, 80, 80]
embed_size = 100
# 嵌入维度,卷积核,通道数
net = TextCNN(vocab, embed_size, kernel_sizes, num_channels)

print(net)




embed_size, num_hiddens, num_layers = 288, 256, 1
time_window = 1


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

print(net)

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()# 优化方法
            train_l_sum += l.cpu().item()# 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

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
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)# 这里输入之后,进入embedding,进入lstm,进入全连接层,输出结果
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



