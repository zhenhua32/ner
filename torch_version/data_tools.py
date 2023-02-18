import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed():
    """
    固定随机种子
    """
    random.seed(32)
    np.random.seed(32)
    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    torch.cuda.manual_seed_all(32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def padding_to_batch_max_len(batch_data: list, w2i_char: dict, w2i_bio: dict):
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


def calc_weight_dict(train_dataset: Dataset, i2w_bio: dict, device: torch.device):
    """
    计算权重
    """
    weight_dict = dict((x, 0) for x in i2w_bio.keys())
    for input_seq, output_seq in train_dataset:
        for x in output_seq:
            weight_dict[x] += 1
    print("权重字典: ", weight_dict)
    summed = sum(weight_dict.values())
    weights = torch.tensor([x / summed for x in weight_dict.values()]).to(device)
    weights = 1.0 / weights
    print("权重: ", weights)
