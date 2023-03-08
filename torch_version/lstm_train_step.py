import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils_tools import extract_kvpairs_in_bio, cal_f1_score


def train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim.Adam,
    device: torch.device,
):
    """
    训练 lstm 模型的过程
    """
    model.train()
    # 总的样本数
    size = len(dataloader.dataset)
    print(f"训练集样本数: {size}")
    for batch, (inputs_seq_batch, outputs_seq_batch, inputs_seq_len_batch) in enumerate(dataloader):
        inputs_seq_batch = inputs_seq_batch.to(device)
        outputs_seq_batch = outputs_seq_batch.to(device)
        inputs_seq_len_batch = inputs_seq_len_batch.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with torch.autocast(device):
            pred, loss = model(inputs_seq_batch, outputs_seq_batch)

        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(inputs_seq_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    use_crf: bool,
    i2w_char: dict,
    i2w_bio: dict,
):
    """
    验证 lstm 模型的过程
    """
    model.eval()
    size = len(dataloader.dataset)
    print(f"验证集样本数: {size}")
    preds_kvpair = []
    golds_kvpair = []

    with torch.no_grad():
        for batch, (inputs_seq_batch, outputs_seq_batch, inputs_seq_len_batch) in enumerate(dataloader):
            inputs_seq_batch = inputs_seq_batch.to(device)
            outputs_seq_batch = outputs_seq_batch.to(device)
            inputs_seq_len_batch = inputs_seq_len_batch.to(device)

            with torch.autocast(device):
                if use_crf:
                    # 使用 crf 解码
                    pred, _ = model(inputs_seq_batch, outputs_seq_batch)
                    preds_seq_batch = model.crf.decode(pred, mask=(inputs_seq_batch != -1).byte())
                else:
                    pred, _ = model(inputs_seq_batch, outputs_seq_batch)
                    preds_seq_batch = pred.argmax(-1).cpu().numpy()

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

        return p, r, f1
