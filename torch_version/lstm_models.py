import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class LSTMModel(nn.Module):
    """
    ner 的 lstm 模型
    """

    def __init__(
        self, num_embeddings: int, output_size: int, embedding_dim: int = 300, hidden_size: int = 300, weights=None
    ) -> None:
        """
        num_embeddings: 词汇表的大小
        output_size: 输出的标签的个数
        embedding_dim: 词向量的维度
        hidden_size: lstm 的隐藏层的维度
        weights: 损失函数的权重, 一般用于解决类别不平衡的问题
        """
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
        self.weights = weights

    def forward(self, x, y=None):
        # x shape: (batch_size, seq_len)
        # y shape: (batch_size, seq_len)
        # emb shape: (batch_size, seq_len, embedding_dim)
        emb = self.embedding(x)
        # output shape: (batch_size, seq_len, 2 * hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(emb)
        # output shape: (batch_size, seq_len, output_size)
        output = self.linear(output)
        # output shape: (batch_size, seq_len, output_size)
        output = F.softmax(output, dim=-1)
        # 如果有 y, 就计算下 loss
        if y is not None:
            # 求损失的时候, 需要形状是 (N, C, other) 的, N 是 batch_size, C 是类别数
            loss = F.cross_entropy(torch.transpose(output, 1, 2), y, ignore_index=-1, weight=self.weights)
            return (output, loss)

        return (output, None)

    def predict(self, x, return_numpy=False):
        output, _ = self.forward(x)
        # output shape: (batch_size, seq_len, output_size)
        # pred shape: (batch_size, seq_len)
        pred = output.argmax(dim=-1)
        if return_numpy:
            return pred.detach().cpu().numpy()
        return pred


class LSTMCRFModel(nn.Module):
    """
    在 LSTMModel 的基础上加入 CRF
    """

    def __init__(self, num_embeddings: int, output_size: int, embedding_dim=300, hidden_size=300) -> None:
        """
        num_embeddings: 词汇表的大小
        output_size: 输出的标签的个数
        embedding_dim: 词向量的维度
        hidden_size: lstm 的隐藏层的维度
        """
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
        # x shape: (batch_size, seq_len)
        # emb shape: (batch_size, seq_len, embedding_dim)
        emb = self.embedding(x)
        # output shape: (batch_size, seq_len, 2 * hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(emb)
        # output shape: (batch_size, seq_len, output_size)
        output = self.linear(output)
        output = F.softmax(output, dim=-1)
        # 使用 crf
        if y is not None:
            # 根据 -1 生成 mask, 类型是 ByteTensor
            mask = (y != -1).byte()
            # crf 的 loss 是负的
            loss = self.crf(output, y, mask=mask)
            return (output, -1 * loss)

        return (output, None)

    def predict(self, x, return_numpy=False):
        output, _ = self.forward(x)
        # output shape: (batch_size, seq_len, output_size)
        # pred shape: (batch_size, seq_len)
        mask = (x != -1).byte()
        # 使用 crf 解码
        pred = self.crf.decode(output, mask=mask)
        if return_numpy:
            return np.array(pred)
        return torch.tensor(pred)
