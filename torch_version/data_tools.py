import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


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


class LSTMDataset(Dataset):
    """
    LSTM 用的词表是直接映射的. 目前看 data/vocab_char.txt 也是小写的
    如果用在新的数据集上, 可以考虑用训练集生成下, 或者网上找个比较全的
    """

    def __init__(self, input_seq_path: str, output_seq_path: str, w2i_char: dict, w2i_bio: dict, do_lower=True) -> None:
        """
        input_seq_path: 输入序列的路径
        output_seq_path: 输出标签的路径
        w2i_char: 单词到索引的映射
        w2i_bio: 标签到索引的映射
        do_lower: 是否转换成小写
        """
        super().__init__()
        print("加载 ", input_seq_path)
        print("加载 ", output_seq_path)

        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 按行处理, 将里面的单词替换成单词索引
                if do_lower:
                    line = line.lower()
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


class BertDataset(Dataset):
    """
    bert 显然是用的自己的分词器了, 且不需要 collate_fn
    """

    def __init__(
        self, input_seq_path: str, output_seq_path: str, w2i_bio: dict, bert_path: str, max_length: int, do_lower=True
    ) -> None:
        """
        input_seq_path: 输入序列的路径
        output_seq_path: 输出标签的路径
        w2i_bio: 标签到索引的映射
        bert_path: bert 的路径
        max_length: 最大序列长度
        do_lower: 是否转换成小写
        """
        super().__init__()
        print("加载 ", input_seq_path)
        print("加载 ", output_seq_path)

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(bert_path)

        inputs_seq = []
        inputs_seq_len = []  # 每个输入序列的长度
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if do_lower:
                    line = line.lower()
                inputs_seq.append("".join(line.split(" ")))
                inputs_seq_len.append(len(line.split(" ")))

        # 批量分词
        # 损失点性能, 就不用动态的批次最大长度了
        tokenized_inputs: BatchEncoding = self.tokenizer(
            inputs_seq,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 同上, 将标签也替换成索引
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)

        # 需要对齐 inputs_seq 和 outputs_seq, 因为 bert 的分词器会在前后增加
        # 最终所有的数据都在这个 data_list 里面
        self.data_list = self.tokenize_and_align_labels(tokenized_inputs, outputs_seq)
        self.data_list["inputs_seq_len"] = inputs_seq_len

        # 简单验证下数据, 总行数要相同, 每行的单词数也要相同
        print("行数", len(inputs_seq), len(outputs_seq))
        assert len(self.data_list["input_ids"]) == len(self.data_list["labels"])
        for input_ids, labels in zip(self.data_list["input_ids"], self.data_list["labels"]):
            assert len(input_ids) == len(labels)

        self.w2i_bio = w2i_bio

    @staticmethod
    def tokenize_and_align_labels(tokenized_inputs: BatchEncoding, outputs_seq: list):
        """
        返回一个对齐后的标签序列, 这是为训练使用的
        outputs_seq: 标签序列
        """
        labels = []
        # 对齐标签时, 将无效的位置设置为该值
        special_token_id = -100
        for i, label in enumerate(outputs_seq):
            # 获取每个 token 对应的单词索引, 因为 bert 的分词可能会把一个单词分成多个 token, 比如 "apple" -> "app", "##le
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            # 上一个 token 对应的单词索引
            previous_word_idx = None
            # 新的标签序列
            label_ids = []
            # Set the special tokens to special token id
            for word_idx in word_ids:
                # Node 的就是特殊 token, 比如开头的 [CLS], 结尾的 [SEP]
                if word_idx is None:
                    label_ids.append(special_token_id)
                # 如果当前 token 对应的单词索引和上一个不同, 说明是一个新的单词, 那么就把当前的标签加入到新的标签序列中
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # 否则就是一个单词的多个 token, 那么也是当作特殊 token 处理
                else:
                    # 这里有两种选择, 即可以当作特殊 token 处理, 也可以当作上一个 token 的标签
                    label_ids.append(label[word_idx])
                    # label_ids.append(special_token_id)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def __len__(self):
        return len(self.data_list["input_ids"])

    def __getitem__(self, index):
        """
        没有后续的 collate_fn, 所以直接返回 tensor 类型, 且不需要用 .copy() 复制
        """
        input_ids = torch.tensor(self.data_list["input_ids"][index])
        attention_mask = torch.tensor(self.data_list["attention_mask"][index])
        token_type_ids = torch.tensor(self.data_list["token_type_ids"][index])
        inputs_seq_len = torch.tensor(self.data_list["inputs_seq_len"][index])
        output_seq = torch.tensor(self.data_list["labels"][index])
        return input_ids, attention_mask, token_type_ids, inputs_seq_len, output_seq
