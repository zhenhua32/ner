import os
import json

import torch

from utils_tools import load_vocabulary, extract_kvpairs_in_bio
from lstm_models import LSTMModel, LSTMCRFModel

# 使用 os.path.abspath("") 可获取当前脚本所在的目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
vocab_char_path = os.path.join(root_path, "./data/resume-zh/vocab_char.txt")
vocab_bioattr_path = os.path.join(root_path, "./data/resume-zh/vocab_bioattr.txt")
lstm_model_path = os.path.join(root_path, "./ckpt/lstm_model.pt")
crf_model_path = os.path.join(root_path, "./ckpt/lstm_crf_model.pt")


class LSTMPredictor:
    """
    lstm 模型预测类
    """
    def __init__(self, vocab_char_path, vocab_bioattr_path, model_path, use_crf=False, device=None) -> None:
        """
        vocab_char_path: 词汇表路径
        vocab_bioattr_path: BIO标签表路径
        model_path: 模型路径
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        w2i_char, i2w_char = load_vocabulary(vocab_char_path)
        w2i_bio, i2w_bio = load_vocabulary(vocab_bioattr_path)

        if use_crf:
            model = LSTMCRFModel(
                num_embeddings=len(w2i_char),
                output_size=len(w2i_bio),
                # TODO: 后面两个参数可以通过配置文件读取
                embedding_dim=300,
                hidden_size=300,
            )
        else:
            model = LSTMModel(
                num_embeddings=len(w2i_char),
                output_size=len(w2i_bio),
                embedding_dim=300,
                hidden_size=300,
            )

        # 加载模型
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)

        self.w2i_char = w2i_char
        self.i2w_char = i2w_char
        self.w2i_bio = w2i_bio
        self.i2w_bio = i2w_bio
        self.model = model
        self.device = device

    def predict(self, text: str, verbose=False):
        """
        text: 输入文本
        """
        x = [self.w2i_char.get(w, 0) for w in text]
        x = torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            y = self.model.predict(x).squeeze(0).cpu().numpy()
        if verbose:
            print("y.shape: ", y.shape)
            print("y: ", y)
        y = [self.i2w_bio[x] for x in y]
        if verbose:
            print("y: ", y)

        # 从 BIO 序列中提取出 K-V 对
        kvpairs = extract_kvpairs_in_bio(y, list(text))
        if verbose:
            print("kvpairs: ", kvpairs)

        return kvpairs


def main():
    lstm_model = LSTMPredictor(vocab_char_path, vocab_bioattr_path, lstm_model_path, use_crf=False)
    crf_model = LSTMPredictor(vocab_char_path, vocab_bioattr_path, crf_model_path, use_crf=True)

    text = "姓名: 张三 性别: 男 年龄: 25 籍贯: 北京, 职位: 机器学习工程师, 就职于霍格沃茨学校"
    print("lstm 模型预测结果:")
    lstm_model.predict(text, verbose=True)
    print("lstm_crf 模型预测结果:")
    crf_model.predict(text, verbose=True)


if __name__ == "__main__":
    main()
