import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim


# 指定设备编号
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_batch(path, word2number_dict, batch_size, n_step, device):
    """
    输入：
        path: 数据路径
        word2number_dict: 单词->索引字典
        batch_size: 批大小
        n_step: RNN上下文窗口大小
        device: 设备，CPU使用："cpu"，GPU使用"cuda:0"
    输出：
        训练数据批次
    """

    text = open(path, 'r', encoding='utf-8')

    input_batch = []
    target_batch = []
    all_input_batch = []
    all_target_batch = []

    for i, sent in enumerate(text):
        tokens = sent.strip().replace('|', '')
        tokens = [t for t in tokens]

        # 为每条句子添加<sos>和<eos>符号
        # tokens = ['<sos>'] + tokens
        # tokens.append('<eos>')

        # 为长度不足n_step的句子补上<pad>符号
        if len(tokens) <= n_step:
            tokens = ["<pad>"] * (n_step + 1 - len(tokens)) + tokens

        for word_index in range(len(tokens) - n_step):
            # print(tokens[word_index: word_index + n_step], '------>', tokens[word_index + n_step])

            # 输入是前n_step-1个单词
            input = [word2number_dict[n] if n in word2number_dict else word2number_dict['<unk>']
                     for n in tokens[word_index: word_index + n_step]]

            # 目标是最后一个单词
            if tokens[word_index + n_step] in word2number_dict:
                target = word2number_dict[tokens[word_index + n_step]]
            else:
                target = word2number_dict['<unk>']
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    text.close()

    return torch.LongTensor(all_input_batch).to(device), torch.LongTensor(all_target_batch).to(device)


def make_dict(train_path):
    """
    输入：
        train_path: 训练数据路径
    输出：
        单词->索引和索引->单词的转换字典
    """
    text = open(train_path, 'r', encoding='utf-8')
    word_list = set()

    for line in text:
        # tokens = line.strip().split(" ")
        tokens = [t for t in line]
        word_list = word_list.union(set(tokens))

    text.close()

    word_list = list(sorted(word_list))
    word2number_dict = {w: i + 3 for i, w in enumerate(word_list)}
    number2word_dict = {i + 3: w for i, w in enumerate(word_list)}

    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk>"] = 1
    number2word_dict[1] = "<unk>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    # word2number_dict["<eos>"] = 3
    # number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.input_emb = nn.Embedding(n_class, emb_size, padding_idx=0)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=1)
        self.output_emb = nn.Linear(hidden_size, n_class)
        self.input_dropout = nn.Dropout(p=0.1)
        self.hidden_dropout = nn.Dropout(p=0.1)
        self.output_dropout = nn.Dropout(p=0.1)

    def forward(self, input_batch):
        # input_batch : [n_step, batch_size, embeding size]
        input_batch = self.input_emb(input_batch).transpose(0, 1)
        input_batch = self.input_dropout(input_batch)

        # outputs : [n_step, batch_size, num_directions(=1) * hidden_size]
        outputs, hidden = self.rnn(input_batch)
        outputs = self.hidden_dropout(outputs)

        # prediction : [batch_size, n_class]
        prediction = self.output_emb(outputs[-1])
        prediction = self.output_dropout(prediction)

        return prediction


def train_rnnlm(model, train_inputs, train_targets, valid_inputs, valid_targets):
    """
    输入：
        model: RNN模型
        train_inputs: 训练输入批次
        train_targets: 训练目标批次
        valid_inputs: 校验输入批次
        valid_targets: 校验目标批次
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training

    batch_number = len(train_inputs)
    train_set = list(zip(train_inputs, train_targets))

    for epoch in range(n_epoch):
        count_batch = 0
        random.shuffle(train_set)

        for input_batch, target_batch in train_set:

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 50 == 0:
                print('epoch', '%04d,' % (epoch + 1), 'step', f'{count_batch + 1}/{batch_number},',
                      'loss:', '{:.6f},'.format(loss.item()), 'ppl:', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            count_batch += 1

        # 每轮执行一次校验
        model.eval()
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(valid_inputs, valid_targets):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'\nValidating at epoch', '%04d:' % (epoch + 1), 'loss:',
                  '{:.6f},'.format(total_loss / count_loss),
                  'ppl:', '{:.6}'.format(math.exp(total_loss / count_loss)))
        print('-' * 80)
        model.train()

        # if (epoch + 1) % save_interval == 0:
        #     if not os.path.exists('models'):
        #         os.makedirs('models')
        #     torch.save(model, f'models/epoch_{epoch + 1}.ckpt')


def test(model, word2number, number2word):
    MAX_LEN = 20
    model.eval()
    raw_sequence = '月色'
    sequence = [word2number[s] for s in raw_sequence]
    sequence = [word2number['<sos>']] + sequence
    with torch.no_grad():
        while len(sequence) <= MAX_LEN : # and sequence[-1] = word2number['eos']
            predictions = model(torch.LongTensor([sequence]).to(device))
            predictions = predictions.data.max(1, keepdim=True)[1]
            sequence.append(predictions.tolist()[-1][-1])
    print(' '.join([str(number2word[n]) for n in sequence]))


if __name__ == '__main__':
    n_step = 5  # RNN上下文长度
    emb_size = 200  # 词向量维度
    hidden_size = 200  # 隐藏层维度
    batch_size = 256  # 批大小
    learn_rate = 0.001  # 学习率
    n_epoch = 5  # 最大训练轮数
    save_interval = 1  # 保存模型的间隔轮数
    train_path = 'C:/Users/13475/Desktop/NLP/RNNLM/古诗/train.txt'  # 训练文件路径
    valid_path = 'C:/Users/13475/Desktop/NLP/RNNLM/古诗/valid.txt'  # 训练文件路径

    # 根据训练数据构建词表
    word2number_dict, number2word_dict = make_dict(train_path)
    n_class = len(word2number_dict)
    print("Vocabulary size:", n_class)

    # 准备训练数据
    train_inputs, train_targets = make_batch(
        train_path, word2number_dict, batch_size, n_step, device)
    print("Number of training batches:", len(train_inputs))

    # 准备校验数据
    valid_inputs, valid_targets = make_batch(
        valid_path, word2number_dict, 1, n_step, device)
    print("Number of validation batches:", len(valid_inputs))

    # 准备模型
    model = TextRNN()
    model.to(device)
    print(model)

    # 执行训练
    print("Start Training")
    train_rnnlm(model, train_inputs, train_targets, valid_inputs, valid_targets)

    # model = torch.load('models/epoch_30.ckpt', map_location='cpu')
    # model.to(device)

    # 生成句子
    print("Start Generating")
    test(model, word2number_dict, number2word_dict)
