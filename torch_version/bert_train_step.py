import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer

from utils_tools import extract_kvpairs_in_bio, cal_f1_score


def train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim.Adam,
    device: torch.device,
):
    """
    训练 bert 模型的过程
    """
    model.train()
    # 总的样本数
    size = len(dataloader.dataset)
    print(f"训练集样本数: {size}")
    for batch, (input_ids, attention_mask, token_type_ids, inputs_seq_len, output_seq) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        inputs_seq_len = inputs_seq_len.to(device)
        output_seq = output_seq.to(device)
        pred, loss = model(input_ids, attention_mask, token_type_ids, output_seq)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    use_crf: bool,
    tokenizer: BertTokenizer,
    i2w_bio: dict,
):
    """
    验证 bert 模型的过程
    """
    model.eval()
    size = len(dataloader.dataset)
    print(f"验证集样本数: {size}")
    preds_kvpair = []
    golds_kvpair = []

    with torch.no_grad():
        for batch, (input_ids, attention_mask, token_type_ids, inputs_seq_len, output_seq) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            inputs_seq_len = inputs_seq_len.to(device)  # 输入序列长度
            output_seq = output_seq.to(device)

            if use_crf:
                # 使用 crf 解码
                pred, _ = model(input_ids, attention_mask, token_type_ids, output_seq)
                # 这里是 0, 表示 padding 的位置
                preds_seq_batch = model.crf.decode(pred, mask=attention_mask.byte())
            else:
                pred, _ = model(input_ids, attention_mask, token_type_ids, output_seq)
                preds_seq_batch = pred.argmax(-1).cpu().numpy()

            input_ids = input_ids.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            token_type_ids = token_type_ids.cpu().numpy()
            inputs_seq_len = inputs_seq_len.cpu().numpy()
            output_seq = output_seq.cpu().numpy()

            for pred_seq, gold_seq, input_seq, l in zip(
                preds_seq_batch,
                output_seq,
                input_ids,
                inputs_seq_len,
            ):
                # 从数字索引转换成标签
                pred_seq = [i2w_bio[i] for i in pred_seq]
                gold_seq = [i2w_bio[i] for i in gold_seq]
                # 从 id 转换成 tokens
                char_seq = tokenizer.convert_ids_to_tokens(input_seq)
                # 以 [SEP] 为界限, 只需要前面的部分. 同时去掉第一个字符 [CLS]
                sep_index = char_seq.index("[SEP]")
                pred_seq = pred_seq[1:sep_index]
                gold_seq = gold_seq[1:sep_index]
                char_seq = char_seq[1:sep_index]

                # TODO: 这里需要验证下
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

        return p, r, f1
