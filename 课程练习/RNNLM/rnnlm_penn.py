from doctest import Example
from typing import Text
import torch

SEQ_LEN = 5
HIDDEN_SIZE = 1024
EMBEDDING_SIZE = 256 
BATCH_SIZE = 256
LR = 0.001
EPOCH_NUM = 10
PATH = "FNNLM\penn\sample-big.txt"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Dictionary:
    def __init__(self, path:str, embedding_size:int):
        self.word2index_dict = dict()
        self.index2word_dict = dict()
        self.path = path
        self.embedding_size = embedding_size

    def make_dict(self):
        print('making dictionary from {} ...'.format(self.path)) 
        word_set = set()
        with open(self.path, 'r' ,encoding='utf-8') as file:
            for line in file:
                word_set = word_set.union(line.split())
        for index, word in enumerate(word_set):
            self.word2index_dict[word] = index + 3
            self.index2word_dict[index + 3] = word
        self.word2index_dict['<pad>'] = 0
        self.index2word_dict[0] = '<pad>'
        self.word2index_dict['<sos>'] = 1
        self.index2word_dict[1] = '<sos>'
        self.word2index_dict['<eos>'] = 2
        self.index2word_dict[2] = '<eos>'
        # <unk> 在数据集中已经存在
        self.dict_len = len(self.word2index_dict)
        print('dictionary size: {}'.format(self.dict_len))
        return self.word2index_dict, self.index2word_dict

class Batch:
    def __init__(self, path, batch_size, seq_len, dictionary:Dictionary):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_batch_list = []
        self.target_batch_list = []
        self.count = 0
        self.dictionary = dictionary
    
    def make_batch(self, skip_short=False):
        input_batch_temp = []
        target_batch_temp = []
        print('making batch from {} ...'.format(self.path)) 
        with open(self.path, 'r' ,encoding='utf-8') as file:
            for line in file:
                tokens = line.split()
                if len(tokens) < self.seq_len:
                    if skip_short:
                        continue #将小于seq_len的样本丢�?
                    tokens = ['<pad>'] * (self.seq_len - len(tokens)) + tokens
                for i in range(0, len(tokens)-self.seq_len + 1) :
                    input_batch_temp.append(list(self.dictionary.word2index_dict[token] if token in self.dictionary.word2index_dict.keys() else self.dictionary.word2index_dict['<unk>'] 
                                                 for token in tokens[i:i+self.seq_len-1]))
                    target_batch_temp.append(self.dictionary.word2index_dict[tokens[i+self.seq_len-1]] if tokens[i+self.seq_len-1] in self.dictionary.word2index_dict.keys() else self.dictionary.word2index_dict['<unk>'])
                    if len(input_batch_temp) == self.batch_size:
                        self.input_batch_list.append(input_batch_temp)
                        self.target_batch_list.append(target_batch_temp)
                        input_batch_temp = []
                        target_batch_temp = []
        # if len(input_batch_temp) != 0: 
        #     self.input_batch_list.append(input_batch_temp)
        #     self.target_batch_list.append(target_batch_temp)
        self.batch_num = len(self.input_batch_list)
        print("sample num: {}. batch size: {}. batch num:{}".format(
                 (len(self.input_batch_list)-1) * self.batch_size + len(self.input_batch_list[-1]),
                 self.batch_size,
                 len(self.input_batch_list)))
        self.train_set = zip(torch.LongTensor(self.input_batch_list).to(DEVICE), torch.LongTensor(self.target_batch_list).to(DEVICE))
        return self.input_batch_list, self.target_batch_list
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.input_batch_list):
            self.count+=1
            return torch.LongTensor(self.input_batch_list[self.count-1]).to(DEVICE), torch.LongTensor(self.target_batch_list[self.count-1]).to(DEVICE)
        else:
            self.count = 0
            raise StopIteration

class TextRNN(torch.nn.Module):
    def __init__(self, dict_len, embedding_size, hidden_size):
        super(TextRNN, self).__init__()
        self.embedding = torch.nn.Embedding(dict_len, embedding_size, padding_idx=0).to(DEVICE)
        self.rnn = torch.nn.RNN(embedding_size, hidden_size).to(DEVICE)
        self.fc = torch.nn.Linear(hidden_size, dict_len, bias=True).to(DEVICE)
    
    def forward(self, X:torch.Tensor):
        X = self.embedding(X)
        X = X.transpose(0,1) # [seq_len, batch, embedding_size]
        outputs, hiddens = self.rnn(X)
        outputs:torch.Tensor = outputs.transpose(0,1)[:,-1,:] # [batch, embedding_size] 或者用hiddens
        prediction = self.fc(outputs)
        return prediction

def train(model:TextRNN, batch:Batch, epoch_num:int, dictionary:Dictionary):
    # criterion = torch.nn.NLLLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epoch_num):
        step = 1
        for inputs, targets in batch:
            optimizer.zero_grad()
            outputs:torch.Tensor = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('epoch: {}. step: {}. loss:{}'.format(epoch, step, loss))
            if step % 1000 == 0:
                input_sen = list()
                for input in inputs.tolist():
                    input_sen.append(' '.join([dictionary.index2word_dict[token] for token in input]))
                target_sen = [dictionary.index2word_dict[target] for target in targets.tolist()]
                hypothesis_sen = [dictionary.index2word_dict[output] for output in outputs.max(-1)[1].tolist()]
                for i, example in enumerate(zip(input_sen, target_sen, hypothesis_sen)):
                    print('{} input: {} \n'.format(i, example[0]) + 
                    '{} target: {} \n'.format(i, example[1]) +  
                    '{} hypothesis: {}'.format(i, example[2]))
            step += 1

if __name__ == '__main__':
    dictionary = Dictionary(PATH, EMBEDDING_SIZE)
    dictionary.make_dict()
    batch = Batch(PATH, BATCH_SIZE, SEQ_LEN, dictionary)
    batch.make_batch()
    model = TextRNN(dictionary.dict_len, EMBEDDING_SIZE, HIDDEN_SIZE)
    train(model, batch, EPOCH_NUM, dictionary)


