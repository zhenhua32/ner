import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoTokenizer, BertModel, BertConfig


class BertNerModel(nn.Module):
    """
    ner 的 bert 模型
    """

    def __init__(self, output_size: int, bert_path) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config: BertConfig = self.bert.config
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
        # logits shape: (batch_size, seq_len, output_size)
        logits = self.linear(sequence_output)

        if labels is not None:
            # loss shape: (1,)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-1
            )
            return (logits, loss)

        return (logits, None)
