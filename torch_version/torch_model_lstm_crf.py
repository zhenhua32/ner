import json
import os
import time
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchcrf import CRF

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
sys.path.append(root_path)


from utils import load_vocabulary, cal_f1_score, extract_kvpairs_in_bio


# 固定随机种子
random.seed(32)
np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed(32)
torch.cuda.manual_seed_all(32)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载词汇表
vocab_char_path = os.path.join(root_path, "./data/resume-zh/vocab_char.txt")
vocab_bioattr_path = os.path.join(root_path, "./data/resume-zh/vocab_bioattr.txt")
w2i_char, i2w_char = load_vocabulary(vocab_char_path)
w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)

# 创建数据集
train_input_file = os.path.join(root_path, "./data/resume-zh/train.input.seq.char")
train_output_file = os.path.join(root_path, "./data/resume-zh/train.output.seq.bioattr")
test_input_file = os.path.join(root_path, "./data/resume-zh/dev.input.seq.char")
test_output_file = os.path.join(root_path, "./data/resume-zh/dev.output.seq.bioattr")


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
        """
        必须要牢记, 返回的数据不能是个引用, 要隔离开, 不然后续的 collate_fn 函数会修改内容
        """
        input_seq = self.inputs_seq[index].copy()
        output_seq = self.outputs_seq[index].copy()
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
        # seq.extend([w2i_bio["O"]] * (max_seq_len - len(seq)))
        seq.extend([-1] * (max_seq_len - len(seq)))

    return (
        torch.tensor(inputs_seq_batch),
        torch.tensor(outputs_seq_batch),
        torch.tensor(inputs_seq_len_batch),
    )


train_dataset = MyDataset(train_input_file, train_output_file, w2i_char, w2i_bio)
test_dataset = MyDataset(test_input_file, test_output_file, w2i_char, w2i_bio)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 计算权重
weight_dict = dict((x, 0) for x in i2w_bio.keys())
for input_seq, output_seq in train_dataset:
    for x in output_seq:
        weight_dict[x] += 1
print("权重字典: ", weight_dict)
summed = sum(weight_dict.values())
weights = torch.tensor([x / summed for x in weight_dict.values()]).to(device)
weights = 1.0 / weights
print("权重: ", weights)


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
        self.crf = CRF(output_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x, y=None):
        emb = self.embedding(x)
        output, (final_hidden_state, final_cell_state) = self.lstm(emb)
        output = self.linear(output)
        output = F.softmax(output, dim=-1)
        # 使用 crf
        if y is not None:
            # 根据 -1 生成 mask, 类型是 ByteTensor
            mask = (y != -1).byte()
            loss = self.crf(output, y, mask=mask)
            return (output, -1 * loss)

        return (output, None)


model = MyModel()
model.to(device)
print(model)

# 那用 CRF 的话, 就不用这个损失函数了
# 这个 ignore_index=-1, 就是前面填充成批次的最大长度的时候, 填充的值
loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=weights, reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader: DataLoader, model: MyModel, loss_fn: nn.CrossEntropyLoss, optimizer: optim.Adam):
    model.train()
    size = len(dataloader.dataset)
    for batch, (inputs_seq_batch, outputs_seq_batch, inputs_seq_len_batch) in enumerate(dataloader):
        inputs_seq_batch = inputs_seq_batch.to(device)
        outputs_seq_batch = outputs_seq_batch.to(device)
        inputs_seq_len_batch = inputs_seq_len_batch.to(device)
        pred, loss = model(inputs_seq_batch, outputs_seq_batch)
        # print("inputs_seq_batch", inputs_seq_batch.shape)
        # print("pred", pred.shape)
        # print("outputs_seq_batch", outputs_seq_batch.shape, outputs_seq_batch.dtype)
        # print("pred", pred.shape, pred.dtype)
        # print(pred)
        # print(outputs_seq_batch)
        # 求损失的时候, 需要形状是 (N, C, other) 的, N 是 batch_size, C 是类别数
        pred = torch.transpose(pred, 1, 2)
        # print("pred", pred.shape, pred.dtype)
        # loss = loss_fn(pred, outputs_seq_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(inputs_seq_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: MyModel):
    model.eval()
    preds_kvpair = []
    golds_kvpair = []

    with torch.no_grad():
        for batch, (inputs_seq_batch, outputs_seq_batch, inputs_seq_len_batch) in enumerate(dataloader):
            inputs_seq_batch = inputs_seq_batch.to(device)
            outputs_seq_batch = outputs_seq_batch.to(device)
            inputs_seq_len_batch = inputs_seq_len_batch.to(device)
            # preds_seq_batch = model(inputs_seq_batch).argmax(-1).cpu().numpy()
            # 使用 crf 解码
            pred, _ = model(inputs_seq_batch, outputs_seq_batch)
            preds_seq_batch = model.crf.decode(pred, mask=(inputs_seq_batch != -1).byte())

            inputs_seq_batch = inputs_seq_batch.cpu().numpy()
            outputs_seq_batch = outputs_seq_batch.cpu().numpy()
            inputs_seq_len_batch = inputs_seq_len_batch.cpu().numpy()

            for pred_seq, gold_seq, input_seq, l in zip(
                preds_seq_batch,
                outputs_seq_batch,
                inputs_seq_batch,
                inputs_seq_len_batch,
            ):
                # 从数字索引转换成标签
                pred_seq = [i2w_bio[i] for i in pred_seq[:l]]
                gold_seq = [i2w_bio[i] for i in gold_seq[:l]]
                char_seq = [i2w_char[i] for i in input_seq[:l]]
                pred_kvpair = extract_kvpairs_in_bio(pred_seq, char_seq)
                gold_kvpair = extract_kvpairs_in_bio(gold_seq, char_seq)

                preds_kvpair.append(pred_kvpair)
                golds_kvpair.append(gold_kvpair)

        p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)

        print("前几行的预测结果和标签: ")
        print(preds_kvpair[:3])
        print(golds_kvpair[:3])
        print("Valid Samples: {}".format(len(preds_kvpair)))
        print("Valid P/R/F1: {} / {} / {}".format(round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)))


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")
