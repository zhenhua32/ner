import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoTokenizer, BertModel, BertConfig


def tokenize_and_align_labels(examples, tokenizer):
    """
    TODO: token 需要特殊处理, 还没实验
    Mapping all tokens to their corresponding word with the word_ids method.
    Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function.
    Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class BertNerModel(nn.Module):
    """
    ner 的 bert 模型
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
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

        if labels:
            # loss shape: (1,)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-1
            )
            return (logits, loss)

        return (logits, None)
