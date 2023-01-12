

import torch

SEQ_LEN = 10
HIDDEN_SIZE = 1024
EMBEDDING_SIZE = 256 
BATCH_SIZE = 256
LR = 0.001
EPOCH_NUM = 20
PATH = "FNNLM\penn\sample-big.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # <unk> 已经存在
        self.dict_len = len(self.word2index_dict)
        print('dictionary size: {}'.format(self.dict_len))
        return self.word2index_dict, self.index2word_dict
    
class Batch:
    def __init__(self, path, batch_size, seq_len):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_batch_list = []
        self.target_batch_list = []
        self.count = 0
    
    def make_batch(self, dictionary:Dictionary):
        input_batch_temp = []
        target_batch_temp = []
        print('making batch from {} ...'.format(self.path)) 
        with open(self.path, 'r' ,encoding='utf-8') as file:
            for line in file:
                tokens = line.split()
                if len(tokens) < self.seq_len:
                    continue #将小于seq_len的样本丢掉
                    tokens = ['<pad>'] * (self.seq_len - len(tokens)) + tokens
                for i in range(0, len(tokens)-self.seq_len + 1) :
                    input_batch_temp.append(list(dictionary.word2index_dict[token] if token in dictionary.word2index_dict.keys() else dictionary.word2index_dict['<unk>'] 
                                                 for token in tokens[i:i+self.seq_len-1]))
                    target_batch_temp.append(dictionary.word2index_dict[tokens[i+self.seq_len-1]] if tokens[i+self.seq_len-1] in dictionary.word2index_dict.keys() else dictionary.word2index_dict['<unk>'])
                    if len(input_batch_temp) == self.batch_size:
                        self.input_batch_list.append(input_batch_temp)
                        self.target_batch_list.append(target_batch_temp)
                        input_batch_temp = []
                        target_batch_temp = []
        # if len(input_batch_temp) != 0: 
        #     self.input_batch_list.append(input_batch_temp)
        #     self.target_batch_list.append(target_batch_temp)
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
       
class NNLM(torch.nn.Module):
    def __init__(self, seq_len, hidden_size, embedding_size, dict_len):
        super(NNLM,self).__init__()
        self.embedding = torch.nn.Embedding(dict_len, embedding_size, padding_idx=0).to(DEVICE)
        self.fc1 = torch.nn.Linear((seq_len-1) * embedding_size, hidden_size, bias=True).to(DEVICE)
        self.fc2 = torch.nn.Linear(hidden_size, dict_len, bias=True).to(DEVICE)
        self.fc3 = torch.nn.Linear((seq_len-1) * embedding_size, dict_len).to(DEVICE)
    
    def forward(self, X:torch.Tensor):
        X = self.embedding(X) # X : [batch_size, len, embedding_size]
        # 维度转换
        # [batch_size, len * embedding_size]
        batch_size, seq_len, embedding_size = X.size()
        X = X.view(batch_size, seq_len * embedding_size)
        Y = torch.tanh(self.fc1(X))
        output = torch.softmax(self.fc2(Y) + self.fc3(X), dim=1)
        return output

def train(model, batch:Batch, epoch_num:int):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epoch_num):
        step = 1
        for inputs, targets in batch:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(torch.log(output), targets)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('epoch: {}. step: {}. loss:{}'.format(epoch, step, loss))
            step += 1

if __name__ == '__main__':
    dictionary = Dictionary(PATH, EMBEDDING_SIZE)
    dictionary.make_dict()
    batch = Batch(PATH, BATCH_SIZE, SEQ_LEN)
    batch.make_batch(dictionary)
    model = NNLM(SEQ_LEN, HIDDEN_SIZE, EMBEDDING_SIZE, dictionary.dict_len)
    train(model, batch, EPOCH_NUM)

   
    

