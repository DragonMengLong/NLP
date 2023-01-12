import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.input_emb = nn.Embedding(n_class, emb_size, padding_idx=0)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=hidden_size)
        self.output_emb = nn.Linear(hidden_size, n_class)

    def forward(self, input_batch):

        # input_batch : [n_step, batch_size, embeding size]
        input_batch = self.input_emb(input_batch).transpose(0, 1)

        # outputs : [n_step, batch_size, num_directions(=1) * hidden_size]
        outputs, hidden = self.rnn(input_batch)
        # prediction : [batch_size, n_class]
        prediction = self.output_emb(outputs[-1])

        return prediction


if __name__ == '__main__':
    n_step = 2  # 上下文长度
    emb_size = 10  # 输入词向量维度
    hidden_size = 10  # 隐藏层维度

    sentences = ["i like dogs", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}

    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for step in range(200):

        # input_batch : [batch_size, n_step, n_class]
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 20 == 0:
            print('step:', '%04d' % (step + 1),
                  'loss =', '{:.6f}'.format(loss))

            # Predict
            predictions = model(input_batch).data.max(
                1, keepdim=True)[1]
            for input_sent, prediction in zip([sen.split()[:2] for sen in sentences], [number_dict[n.item()] for n in predictions.squeeze()]):
                print(input_sent, '->', prediction)
