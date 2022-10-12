import random
import numpy as np

##########################
####### Vocabulary #######
##########################


def load_vocabulary(path):
    """
    加载词汇表, 按行读取, 返回 word=>index 和 index=>word 的字典
    """
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w


#########################################
####### DataProcessor 1: for LSTM #######
#########################################


class DataProcessor_LSTM(object):
    """
    处理 LSTM 的数据
    """

    def __init__(
        self, input_seq_path: str, output_seq_path: str, w2i_char: dict, w2i_bio: dict, shuffling: bool = False
    ):
        """
        input_seq_path: 输入文件的路径
        output_seq_path: 标签文件的路径
        w2i_char: word=>index 的字典
        w2i_bio: label=>index 的字典
        shuffling: 是否乱序
        """
        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 按行处理, 将里面的单词替换成单词索引
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)

        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                # 同上, 将标签也替换成索引
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)

        # 简单验证下数据, 总行数要相同, 每行的单词数也要相同
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        # 生成索引数组
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)

    def refresh(self):
        """
        重置数组, 就是重新初始化了索引位置和标记
        """
        if self.shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size: int):
        """
        batch_size: 批次大小
        获取一批次的数据
        """
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            # 当前索引位置
            p = self.ps[self.pointer]
            # 加一条数据
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps):
                self.end_flag = True

        # 获取最大序列长度, 全部填充到同样的长度
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))

        return (
            np.array(inputs_seq_batch, dtype="int32"),
            np.array(inputs_seq_len_batch, dtype="int32"),
            np.array(outputs_seq_batch, dtype="int32"),
        )


#########################################
####### DataProcessor 2: for BERT #######
#########################################


class DataProcessor_BERT(object):
    def __init__(self, input_seq_path, output_seq_path, w2i_char, w2i_bio, shuffling=False):

        with open(input_seq_path, "r", encoding="utf-8") as f:
            lines1 = f.read().strip().split("\n")
        with open(output_seq_path, "r", encoding="utf-8") as f:
            lines2 = f.read().strip().split("\n")

        inputs_seq = []
        outputs_seq = []
        for line1, line2 in zip(lines1, lines2):
            words = []
            bios = []
            for word, bio in zip(line1.split(" "), line2.split(" ")):
                if word != "[SPA]":
                    words.append(word)
                    bios.append(bio)

            words.insert(0, "[CLS]")
            words.append("[SEP]")
            seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in words]
            inputs_seq.append(seq)

            bios.insert(0, "O")
            bios.append("O")
            seq = [w2i_bio[bio] for bio in bios]
            outputs_seq.append(seq)

        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)) + " shuffling: " + str(shuffling))

    def refresh(self):
        if self.shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_mask_batch = []
        inputs_segment_batch = []
        outputs_seq_batch = []
        lens = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            l = len(self.inputs_seq[p])
            inputs_mask_batch.append([1] * l)
            inputs_segment_batch.append([0] * l)
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            lens.append(l)
            self.pointer += 1
            if self.pointer >= len(self.ps):
                self.end_flag = True

        max_seq_len = max(lens)
        for input_seq, input_mask, input_segment, output_seq, l in zip(
            inputs_seq_batch, inputs_mask_batch, inputs_segment_batch, outputs_seq_batch, lens
        ):
            input_seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - l))
            input_mask.extend([0] * (max_seq_len - l))
            input_segment.extend([0] * (max_seq_len - l))
            output_seq.extend([self.w2i_bio["O"]] * (max_seq_len - l))

        return (
            np.array(inputs_seq_batch, dtype="int32"),
            np.array(inputs_mask_batch, dtype="int32"),
            np.array(inputs_segment_batch, dtype="int32"),
            np.array(outputs_seq_batch, dtype="int32"),
        )


###################################################
####### DataProcessor 3: for Multitask-LSTM #######
###################################################


class DataProcessor_MTL_LSTM(object):
    def __init__(
        self, input_seq_path, output_seq_bio_path, output_seq_attr_path, w2i_char, w2i_bio, w2i_attr, shuffling=False
    ):

        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)

        outputs_seq_bio = []
        with open(output_seq_bio_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq_bio.append(seq)

        outputs_seq_attr = []
        with open(output_seq_attr_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_attr[word] for word in line.split(" ")]
                outputs_seq_attr.append(seq)

        assert len(inputs_seq) == len(outputs_seq_bio)
        assert all(
            len(input_seq) == len(output_seq_bio) for input_seq, output_seq_bio in zip(inputs_seq, outputs_seq_bio)
        )
        assert len(inputs_seq) == len(outputs_seq_attr)
        assert all(
            len(input_seq) == len(output_seq_attr) for input_seq, output_seq_attr in zip(inputs_seq, outputs_seq_attr)
        )

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.w2i_attr = w2i_attr
        self.inputs_seq = inputs_seq
        self.outputs_seq_bio = outputs_seq_bio
        self.outputs_seq_attr = outputs_seq_attr
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)

    def refresh(self):
        if self.shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_bio_batch = []
        outputs_seq_attr_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_bio_batch.append(self.outputs_seq_bio[p].copy())
            outputs_seq_attr_batch.append(self.outputs_seq_attr[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps):
                self.end_flag = True

        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_bio_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_attr_batch:
            seq.extend([self.w2i_attr["null"]] * (max_seq_len - len(seq)))

        return (
            np.array(inputs_seq_batch, dtype="int32"),
            np.array(inputs_seq_len_batch, dtype="int32"),
            np.array(outputs_seq_bio_batch, dtype="int32"),
            np.array(outputs_seq_attr_batch, dtype="int32"),
        )


###################################################
####### DataProcessor 4: for Multitask-BERT #######
###################################################


class DataProcessor_MTL_BERT(object):
    def __init__(
        self, input_seq_path, output_seq_bio_path, output_seq_attr_path, w2i_char, w2i_bio, w2i_attr, shuffling=False
    ):

        with open(input_seq_path, "r", encoding="utf-8") as f:
            lines1 = f.read().strip().split("\n")
        with open(output_seq_bio_path, "r", encoding="utf-8") as f:
            lines2 = f.read().strip().split("\n")
        with open(output_seq_attr_path, "r", encoding="utf-8") as f:
            lines3 = f.read().strip().split("\n")

        inputs_seq = []
        outputs_seq_bio = []
        outputs_seq_attr = []
        for line1, line2, line3 in zip(lines1, lines2, lines3):
            words = []
            bios = []
            attrs = []
            for word, bio, attr in zip(line1.split(" "), line2.split(" "), line3.split(" ")):
                if word != "[SPA]":
                    words.append(word)
                    bios.append(bio)
                    attrs.append(attr)

            words.insert(0, "[CLS]")
            words.append("[SEP]")
            seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in words]
            inputs_seq.append(seq)

            bios.insert(0, "O")
            bios.append("O")
            seq = [w2i_bio[bio] for bio in bios]
            outputs_seq_bio.append(seq)

            attrs.insert(0, "null")
            attrs.append("null")
            seq = [w2i_attr[attr] for attr in attrs]
            outputs_seq_attr.append(seq)

        assert len(inputs_seq) == len(outputs_seq_bio)
        assert all(
            len(input_seq) == len(output_seq_bio) for input_seq, output_seq_bio in zip(inputs_seq, outputs_seq_bio)
        )
        assert len(inputs_seq) == len(outputs_seq_attr)
        assert all(
            len(input_seq) == len(output_seq_attr) for input_seq, output_seq_attr in zip(inputs_seq, outputs_seq_attr)
        )

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.w2i_attr = w2i_attr
        self.inputs_seq = inputs_seq
        self.outputs_seq_bio = outputs_seq_bio
        self.outputs_seq_attr = outputs_seq_attr
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)) + " shuffling: " + str(shuffling))

    def refresh(self):
        if self.shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_mask_batch = []
        inputs_segment_batch = []
        outputs_seq_bio_batch = []
        outputs_seq_attr_batch = []
        lens = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            l = len(self.inputs_seq[p])
            inputs_mask_batch.append([1] * l)
            inputs_segment_batch.append([0] * l)
            outputs_seq_bio_batch.append(self.outputs_seq_bio[p].copy())
            outputs_seq_attr_batch.append(self.outputs_seq_attr[p].copy())
            lens.append(l)
            self.pointer += 1
            if self.pointer >= len(self.ps):
                self.end_flag = True

        max_seq_len = max(lens)
        for input_seq, input_mask, input_segment, output_seq_bio, output_seq_attr, l in zip(
            inputs_seq_batch,
            inputs_mask_batch,
            inputs_segment_batch,
            outputs_seq_bio_batch,
            outputs_seq_attr_batch,
            lens,
        ):
            input_seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - l))
            input_mask.extend([0] * (max_seq_len - l))
            input_segment.extend([0] * (max_seq_len - l))
            output_seq_bio.extend([self.w2i_bio["O"]] * (max_seq_len - l))
            output_seq_attr.extend([self.w2i_attr["null"]] * (max_seq_len - l))

        return (
            np.array(inputs_seq_batch, dtype="int32"),
            np.array(inputs_mask_batch, dtype="int32"),
            np.array(inputs_segment_batch, dtype="int32"),
            np.array(outputs_seq_bio_batch, dtype="int32"),
            np.array(outputs_seq_attr_batch, dtype="int32"),
        )


####################################################################
####### DataProcessor 5: for Multitask-LSTM-WordLevelFeature #######
####################################################################


class DataProcessor_MTL_LSTM_WLF(object):
    def __init__(
        self,
        input_seq_char_path,
        input_seq_word_path,
        output_seq_bio_path,
        output_seq_attr_path,
        w2i_char,
        w2i_word,
        w2i_bio,
        w2i_attr,
        shuffling=False,
    ):

        inputs_seq_char = []
        with open(input_seq_char_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq_char.append(seq)

        inputs_seq_word = []
        with open(input_seq_word_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = []
                for word in line.split(" "):
                    i = w2i_word[word] if word in w2i_word else w2i_word["[UNK]"]
                    if word == "[SPA]":
                        seq.append(i)
                    else:
                        seq.extend([i] * len(word))
                inputs_seq_word.append(seq)

        outputs_seq_bio = []
        with open(output_seq_bio_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq_bio.append(seq)

        outputs_seq_attr = []
        with open(output_seq_attr_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_attr[word] for word in line.split(" ")]
                outputs_seq_attr.append(seq)

        assert len(inputs_seq_char) == len(inputs_seq_word)
        assert all(
            len(input_seq_char) == len(input_seq_word)
            for input_seq_char, input_seq_word in zip(inputs_seq_char, inputs_seq_word)
        )
        assert len(inputs_seq_char) == len(outputs_seq_bio)
        assert all(
            len(input_seq_char) == len(output_seq_bio)
            for input_seq_char, output_seq_bio in zip(inputs_seq_char, outputs_seq_bio)
        )
        assert len(inputs_seq_char) == len(outputs_seq_attr)
        assert all(
            len(input_seq_char) == len(output_seq_attr)
            for input_seq_char, output_seq_attr in zip(inputs_seq_char, outputs_seq_attr)
        )

        self.w2i_char = w2i_char
        self.w2i_word = w2i_word
        self.w2i_bio = w2i_bio
        self.w2i_attr = w2i_attr
        self.inputs_seq_char = inputs_seq_char
        self.inputs_seq_word = inputs_seq_word
        self.outputs_seq_bio = outputs_seq_bio
        self.outputs_seq_attr = outputs_seq_attr
        self.ps = list(range(len(inputs_seq_char)))
        self.shuffling = shuffling
        if shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq_char)), "shuffling:", shuffling)

    def refresh(self):
        if self.shuffling:
            random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_char_batch = []
        inputs_seq_word_batch = []
        inputs_seq_len_batch = []
        outputs_seq_bio_batch = []
        outputs_seq_attr_batch = []

        while (len(inputs_seq_char_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_char_batch.append(self.inputs_seq_char[p].copy())
            inputs_seq_word_batch.append(self.inputs_seq_word[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq_char[p]))
            outputs_seq_bio_batch.append(self.outputs_seq_bio[p].copy())
            outputs_seq_attr_batch.append(self.outputs_seq_attr[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps):
                self.end_flag = True

        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_char_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in inputs_seq_word_batch:
            seq.extend([self.w2i_word["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_bio_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_attr_batch:
            seq.extend([self.w2i_attr["null"]] * (max_seq_len - len(seq)))

        return (
            np.array(inputs_seq_char_batch, dtype="int32"),
            np.array(inputs_seq_word_batch, dtype="int32"),
            np.array(inputs_seq_len_batch, dtype="int32"),
            np.array(outputs_seq_bio_batch, dtype="int32"),
            np.array(outputs_seq_attr_batch, dtype="int32"),
        )


######################################
####### extract_kvpairs_by_bio #######
######################################


def extract_kvpairs_in_bio(bio_seq: list, word_seq: list):
    """
    bio_seq: 标签的数组
    word_seq: 单词的数组
    抽取 ner 标签
    """
    assert len(bio_seq) == len(word_seq)
    pairs = set()  # 是 (标签, 文本) 的集合
    pre_bio = "O"  # 上一个标签
    v = ""  # 当前的文本
    for i, bio in enumerate(bio_seq):
        if bio == "O":
            # 当遇到下一个 O 时, 把前面的标签放进去
            if v != "":
                pairs.add((pre_bio[2:], v))
            v = ""
        elif bio[0] == "B":
            # 当遇到下一个 B 时, 和上面一样, 都是开启新的标签
            if v != "":
                pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif bio[0] == "I":
            # 如果是 O 或者标签不一致. (有点奇怪, 会遇到 (B_水果, I_蔬菜) 这种情况吗? 或者是 (O, I_蔬菜), (I_水果, I_蔬菜)
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "":
                    pairs.add((pre_bio[2:], v))
                v = ""
            else:
                # 继续将文本加进去
                v += word_seq[i]
        pre_bio = bio
    # 最后再加一遍
    if v != "":
        pairs.add((pre_bio[2:], v))
    return pairs


def extract_kvpairs_in_bioes(bio_seq, word_seq, attr_seq):
    assert len(bio_seq) == len(word_seq) == len(attr_seq)
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]
        attr = attr_seq[i]
        if bio == "O":
            v = ""
        elif bio == "S":
            v = word
            pairs.add((attr, v))
            v = ""
        elif bio == "B":
            v = word
        elif bio == "I":
            if v != "":
                v += word
        elif bio == "E":
            if v != "":
                v += word
                pairs.add((attr, v))
            v = ""
    return pairs


############################
####### cal_f1_score #######
############################


def cal_f1_score(preds, golds):
    """
    手动计算 F1
    """
    assert len(preds) == len(golds)
    p_sum = 0  # 总数
    r_sum = 0  # 总数
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        # 对于每个预测的标签, 在 label 中就 + 1
        for label in pred:
            if label in gold:
                hits += 1
    # 精准率就是 对的个数 / 总的预测出来的个数
    p = hits / p_sum if p_sum > 0 else 0
    # 召回率就是 对的个数 / 总的标签的个数
    r = hits / r_sum if r_sum > 0 else 0
    # f1 就是 (2pr)/(p+r)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1
