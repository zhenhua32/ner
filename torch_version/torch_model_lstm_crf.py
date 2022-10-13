import json
import os
import time

import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import load_vocabulary

# 加载词汇表
vocab_char_path = os.path.join(root_path, "./data/vocab_char.txt")
vocab_bioattr_path = os.path.join(root_path, "./data/vocab_bioattr.txt")
w2i_char, i2w_char = load_vocabulary(vocab_char_path)
w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)

# 创建数据集
train_input_file = os.path.join(root_path, "./data/train/input.seq.char")
train_output_file = os.path.join(root_path, "./data/train/output.seq.bioattr")
test_input_file = os.path.join(root_path, "./data/test/input.seq.char")
test_output_file = os.path.join(root_path, "./data/test/output.seq.bioattr")


class MyDataset(Dataset):
    def __init__(self, input_seq_path: str, output_seq_path: str, w2i_char: dict, w2i_bio: dict) -> None:
        super().__init__()
        print("加载 ", input_seq_path)
        print("加载 ", output_seq_path)

        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 按行处理, 将里面的单词替换成单词索引
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)

        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 同上, 将标签也替换成索引
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)

        # 简单验证下数据, 总行数要相同, 每行的单词数也要相同
        print("行数", len(inputs_seq), len(outputs_seq))
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq

    def __len__(self):
        return len(self.inputs_seq)

    def __getitem__(self, index):
        input_seq = self.inputs_seq[index]
        output_seq = self.outputs_seq[index]
        return input_seq, output_seq


def collate_fn(batch_data, w2i_char=w2i_char, w2i_bio=w2i_bio):
    """
    需要把序列填充到同一个长度
    """
    inputs_seq_batch, outputs_seq_batch = map(list, zip(*batch_data))
    inputs_seq_len_batch = [len(x) for x in inputs_seq_batch]

    # 获取最大序列长度, 全部填充到同样的长度
    max_seq_len = max(inputs_seq_len_batch)
    for seq in inputs_seq_batch:
        seq.extend([w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
    for seq in outputs_seq_batch:
        seq.extend([w2i_bio["O"]] * (max_seq_len - len(seq)))

    return (
        torch.tensor(inputs_seq_batch),
        torch.tensor(outputs_seq_batch),
        torch.tensor(inputs_seq_len_batch),
    )


train_dataset = MyDataset(train_input_file, train_output_file, w2i_char, w2i_bio)
test_dataset = MyDataset(test_input_file, test_output_file, w2i_char, w2i_bio)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)


# 创建模型
class MyModel(nn.Module):
    def __init__(
        self, num_embeddings=len(w2i_char), embedding_dim=300, hidden_size=300, output_size=len(w2i_bio)
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * hidden_size, output_size)
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        emb = self.embedding(x)
        # h0 = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        # c0 = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        output, (final_hidden_state, final_cell_state) = self.lstm(emb)
        output = self.linear(output)
        output = self.softmax(output)
        return output


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel()
model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(dataloader: DataLoader, model: MyModel, loss_fn: nn.CrossEntropyLoss, optimizer: optim.Adam):
    model.train()
    size = len(dataloader.dataset)
    for batch, (inputs_seq_batch, outputs_seq_batch, inputs_seq_len_batch) in enumerate(dataloader):
        inputs_seq_batch = inputs_seq_batch.to(device)
        outputs_seq_batch = outputs_seq_batch.to(device)
        inputs_seq_len_batch = inputs_seq_len_batch.to(device)
        pred = model(inputs_seq_batch)
        # print("inputs_seq_batch", inputs_seq_batch.shape)
        # print("pred", pred.shape)
        # print("outputs_seq_batch", outputs_seq_batch.shape, outputs_seq_batch.dtype)
        # print("pred", pred.shape, pred.dtype)
        # print(pred)
        # print(outputs_seq_batch)
        # 求损失的时候, 需要形状是 (N, C, other) 的, N 是 batch_size, C 是类别数
        pred = torch.transpose(pred, 1, 2)
        # print("pred", pred.shape, pred.dtype)
        loss = loss_fn(pred, outputs_seq_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(inputs_seq_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")
