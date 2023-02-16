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
