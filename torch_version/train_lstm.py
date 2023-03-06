import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import transformers

from utils_tools import load_vocabulary
from data_tools import set_seed, LSTMDataset, padding_to_batch_max_len, calc_weight_dict
from lstm_models import LSTMModel, LSTMCRFModel
from lstm_train_step import train, test


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载词汇表
vocab_char_path = os.path.join(root_path, "./data/resume-zh/vocab.txt")
vocab_bioattr_path = os.path.join(root_path, "./data/resume-zh/vocab_bioattr.txt")
w2i_char, i2w_char = load_vocabulary(vocab_char_path)
w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)

# 创建数据集
train_input_file = os.path.join(root_path, "./data/resume-zh-one/train.csv")
train_output_file = None
test_input_file = os.path.join(root_path, "./data/resume-zh-one/dev.csv")
test_output_file = None

collate_fn = partial(padding_to_batch_max_len, w2i_char=w2i_char, w2i_bio=w2i_bio)
train_dataset = LSTMDataset(train_input_file, train_output_file, w2i_char, w2i_bio)
test_dataset = LSTMDataset(test_input_file, test_output_file, w2i_char, w2i_bio)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

weights = calc_weight_dict(train_dataset, i2w_bio, device)

# flag
use_crf = True

if use_crf:
    model = LSTMCRFModel(
        num_embeddings=len(w2i_char),
        output_size=len(w2i_bio),
        embedding_dim=300,
        hidden_size=300,
    )
else:
    model = LSTMModel(
        num_embeddings=len(w2i_char),
        output_size=len(w2i_bio),
        embedding_dim=300,
        hidden_size=300,
        weights=weights,  # 加了权重起步快, 但在 100 轮后结果更差些
    )
model.to(device)
print(model)

epochs = 20
max_train_steps = epochs * len(train_dataloader)
lr_warmup_steps = max_train_steps // 20

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=lr_warmup_steps,
#     num_training_steps=max_train_steps,
#     num_cycles=2,
# )
if use_crf:
    # crf 的起步也太慢了, 第一轮的结果就比较差
    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=max_train_steps,
    #     num_cycles=0.5,
    # )
    scheduler = None
else:
    # 对于 lstm 来说, 不需要 warmup, 第一个 epoch 就比较高了
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
        num_cycles=0.5,
    )

best_f1 = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer, device)
    p, r, f1 = test(test_dataloader, model, device, use_crf, i2w_char, i2w_bio)
    if f1 > best_f1:
        # 保存模型
        model_file_name = "lstm_model.pt" if not use_crf else "lstm_crf_model.pt"
        model_path = os.path.join(root_path, f"./ckpt/{model_file_name}")
        torch.save(model.state_dict(), model_path)
        best_f1 = f1
    # 在每轮结束后调整学习率
    if scheduler is not None:
        scheduler.step()
print("Done!")
print("best_f1: ", best_f1)
