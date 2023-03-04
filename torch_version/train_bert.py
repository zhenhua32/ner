import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils_tools import load_vocabulary
from data_tools import set_seed, BertDataset
from bert_models import BertNerModel, BertNerCRFModel
from bert_train_step import train, test


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载词汇表
vocab_bioattr_path = os.path.join(root_path, "./data/resume-zh/vocab_bioattr.txt")
w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)

# 创建数据集
train_input_file = os.path.join(root_path, "./data/resume-zh/train.input.seq.char")
train_output_file = os.path.join(root_path, "./data/resume-zh/train.output.seq.bioattr")
test_input_file = os.path.join(root_path, "./data/resume-zh/dev.input.seq.char")
test_output_file = os.path.join(root_path, "./data/resume-zh/dev.output.seq.bioattr")

bert_path = r"D:\code\pretrain_model_dir\bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
collate_fn = None
train_dataset = BertDataset(train_input_file, train_output_file, w2i_bio, bert_path, max_length=64)
test_dataset = BertDataset(test_input_file, test_output_file, w2i_bio, bert_path, max_length=64)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)


# flag
use_crf = False

if use_crf:
    model = BertNerCRFModel(
        output_size=len(w2i_bio),
        bert_path=bert_path,
    )
else:
    model = BertNerModel(
        output_size=len(w2i_bio),
        bert_path=bert_path,
    )
model.to(device)
print(model)

bert_params = [param for name, param in model.named_parameters() if "bert" in name]
other_params = [param for name, param in model.named_parameters() if "bert" not in name]

optimizer = torch.optim.AdamW([
    {'params': bert_params, 'lr': 5e-5},
    {'params': other_params, 'lr': 1e-3},
])

epochs = 100
best_f1 = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer, device)
    p, r, f1 = test(test_dataloader, model, device, use_crf, tokenizer, i2w_bio)
    if f1 > best_f1:
        # 保存模型
        model_file_name = "bert_model.pt" if not use_crf else "bert_crf_model.pt"
        model_path = os.path.join(root_path, f"./ckpt/{model_file_name}")
        torch.save(model.state_dict(), model_path)
        best_f1 = f1
print("Done!")
print("best_f1: ", best_f1)
