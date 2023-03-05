import os
import json

import torch
from transformers import BertTokenizerFast

from utils_tools import load_vocabulary, extract_kvpairs_in_bio
from bert_models import BertNerModel, BertNerCRFModel

# 使用 os.path.abspath("") 可获取当前脚本所在的目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
vocab_bioattr_path = os.path.join(root_path, "./data/resume-zh/vocab_bioattr.txt")
bert_model_path = os.path.join(root_path, "./ckpt/bert_model.pt")
crf_model_path = os.path.join(root_path, "./ckpt/bert_crf_model.pt")
# TODO: 这个不应该存在, 应该选择合并, 只留下配置文件
bert_path = r"D:\code\pretrain_model_dir\bert-base-chinese"


class BertPredictor:
    def __init__(self, vocab_bioattr_path, model_path, use_crf=False, device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(bert_path)

        if use_crf:
            model = BertNerCRFModel(
                output_size=len(w2i_bio),
                # TODO: 没必要加载原始的 bert 变量, 可以只用配置构建一个空的 bert 模型
                bert_path=bert_path,
            )
        else:
            model = BertNerModel(
                output_size=len(w2i_bio),
                bert_path=bert_path,
            )

        # 加载模型
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)

        self.w2i_bio = w2i_bio
        self.i2w_bio = i2w_bio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def predict(self, text: str, verbose=False):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            y = self.model.predict(**inputs).squeeze(0).cpu().numpy()
        if verbose:
            print("y.shape: ", y.shape)
            print("y: ", y)
        y = [self.i2w_bio[x] for x in y]
        if verbose:
            print("y: ", y)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # 从 BIO 序列中提取出 K-V 对
        kvpairs = extract_kvpairs_in_bio(y, tokens)
        print(kvpairs)

        return kvpairs


def main():
    bert_model = BertPredictor(vocab_bioattr_path, bert_model_path, use_crf=False)
    crf_model = BertPredictor(vocab_bioattr_path, crf_model_path, use_crf=True)

    text = "姓名: 张三 性别: 男 年龄: 25 籍贯: 北京, 职位: 机器学习工程师, 就职于霍格沃茨学校"
    print("bert 模型预测结果:")
    bert_model.predict(text, verbose=True)
    print("bert_crf 模型预测结果:")
    crf_model.predict(text, verbose=True)


if __name__ == "__main__":
    main()
