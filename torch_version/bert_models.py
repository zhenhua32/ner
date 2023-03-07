import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoTokenizer, BertModel, BertConfig, BertForTokenClassification


class BertNerModel(nn.Module):
    """
    ner 的 bert 模型
    """

    def __init__(self, output_size: int, bert_path: str) -> None:
        """
        output_size: 输出的标签的个数
        bert_path: bert 的预训练模型的路径
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config: BertConfig = self.bert.config
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # input_ids shape: (batch_size, seq_len)
        # attention_mask shape: (batch_size, seq_len)
        # token_type_ids shape: (batch_size, seq_len)
        # labels shape: (batch_size, seq_len)
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # sequence_output shape: (batch_size, seq_len, hidden_size)
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        # logits shape: (batch_size, seq_len, output_size)
        logits = self.linear(sequence_output)

        if labels is not None:
            # loss shape: (1,)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
            return (logits, loss)

        return (logits, None)

    def predict(self, input_ids, attention_mask, token_type_ids, return_numpy=False):
        logits, _ = self.forward(input_ids, attention_mask, token_type_ids)
        # logits shape: (batch_size, seq_len, output_size)
        # pred shape: (batch_size, seq_len)
        pred = logits.argmax(dim=-1)
        if return_numpy:
            return pred.detach().cpu().numpy()
        return pred


class BertNerCRFModel(nn.Module):
    """
    ner 的 bert + crf 模型
    """

    def __init__(self, output_size: int, bert_path: str) -> None:
        """
        output_size: 输出的标签的个数
        bert_path: bert 的预训练模型的路径
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config: BertConfig = self.bert.config
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_config.hidden_size, output_size)
        self.crf = CRF(output_size, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # input_ids shape: (batch_size, seq_len)
        # attention_mask shape: (batch_size, seq_len)
        # token_type_ids shape: (batch_size, seq_len)
        # labels shape: (batch_size, seq_len)
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # sequence_output shape: (batch_size, seq_len, hidden_size)
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        # logits shape: (batch_size, seq_len, output_size)
        logits = self.linear(sequence_output)

        if labels is not None:
            # loss shape: (1,)
            # TODO: 还在想着怎么把 -100 的位置去掉
            labels_mask = labels == -100
            temp_labels = torch.clone(labels)
            temp_labels[labels_mask] = 0
            loss = -self.crf(logits, temp_labels, mask=attention_mask.bool(), reduction="mean")
            return (logits, loss)

        return (logits, None)

    def predict(self, input_ids, attention_mask, token_type_ids, return_numpy=False):
        logits, _ = self.forward(input_ids, attention_mask, token_type_ids)
        # logits shape: (batch_size, seq_len, output_size)
        # pred shape: (batch_size, seq_len)
        pred = self.crf.decode(logits, attention_mask.bool())
        if return_numpy:
            return np.array(pred)
        return torch.tensor(pred)
