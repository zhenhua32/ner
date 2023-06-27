# 命名实体识别实践与探索

完整文档：https://zhuanlan.zhihu.com/p/166496466

最近在做命名实体识别（Named Entity Recognition, NER）的工作，也就是序列标注（Sequence Tagging），老 NLP task 了，虽然之前也做过但是想细致地捋一下，看一下自从有了 LSTM+CRF 之后，NER 在做些什么，顺便记录一下最近的工作，中间有些经验和想法，有什么就记点什么

## 还是先放结论

命名实体识别虽然是一个历史悠久的老任务了，但是自从 2015 年有人使用了 BI-LSTM-CRF 模型之后，这个模型和这个任务简直是郎才女貌，天造地设，轮不到任何妖怪来反对。直到后来出现了 BERT。在这里放两个问题：

1. 2015-2019 年，BERT 出现之前 4 年的时间，命名实体识别就只有 BI-LSTM-CRF 了吗？
2. 2019 年 BERT 出现之后，命名实体识别就只有 BERT-CRF（或者 BERT-LSTM-CRF）了吗？

经过我不完善也不成熟的调研之后，好像的确是的，一个能打的都没有

既然模型打不动了，然后我找了找 ACL2020 做 NER 的论文，看看现在的 NER 还在做哪些事情，主要分几个方面

- 多特征：实体识别不是一个特别复杂的任务，不需要太深入的模型，那么就是加特征，特征越多效果越好，所以字特征、词特征、词性特征、句法特征、KG 表征等等的就一个个加吧，甚至有些中文 NER 任务里还加入了拼音特征、笔画特征。。？心有多大，特征就有多多
- 多任务：很多时候做 NER 的目的并不仅是为了 NER，而是一个大任务下的子任务，比如信息抽取、问答系统等等的，如果要做一个端到端的模型，那么就需要根据自己的需求和场景，做成一个多任务模型，把 NER 作为其中一个子任务；另外，单独的 NER 本身也可以做成多任务，比如一个用来识别实体，一个用来判断实体类型
- 时令大杂烩：把当下比较流行的深度学习话题或方法跟 NER 结合一下，比如结合强化学习的 NER、结合 few-shot learning 的 NER、结合多模态信息的 NER、结合跨语种学习的 NER 等等的，具体就不提了

所以沿着上述思路，就在一个中文 NER 任务上做一些实践，写一些模型。都列在下面了，首先是 LSTM-CRF 和 BERT-CRF，然后 Cascade 开头的是几个多任务模型（因为实体类型比较多，把 NER 拆成两个任务，一个用来识别实体，另一个用来判断实体类型），后面的几个模型里，WLF 指的是 Word Level Feature（即在原本字级别的序列标注任务上加入词级别的表征），WOL 指的是 Weight of Loss（即在 loss 函数方面通过设置权重来权衡 Precision 与 Recall，以达到提高 F1 的目的）

![](https://pic2.zhimg.com/80/v2-3062da7d38adce1213af496239f04bd9_720w.jpg)

# 关于数据集

**这个原始 git 里的数据集质量不行, 不能用**, 只能跑起来而已. 原作者的知乎文章里也说到了.

> 环境：Python3, Tensorflow1.12
>
> 数据：一个电商场景下商品标题中的实体识别，因为是工作中的数据，并且通过远程监督弱标注的质量也一般，完整数据就不放了。但是我 sample 了一些数据留在 git 里了，为了直接 git clone 完，代码原地就能跑，方便你我他

# 复刻

原始代码是用 Tensorflow1.12 实现的.
用 pytorch 和 tensorflow 2 重新实现.

先从最简单的开始实现.

- [] BiLSTM
- [] BiLSTM + CRF

# 导出成 onnx 模型

```bash
python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/dense/kernel:0

python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/dense/bias:0,projection/transitions:0

python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/transitions:0,projection/Softmax:0
```

环境变量 LD_LIBRARY_PATH 是动态库查找的路径.

```bash
spark-submit --conf spark.executorEnv.LD_LIBRARY_PATH=/usr/local/lib my_app.py
```

- [conda 使用动态链接库的小秘密](https://zhuanlan.zhihu.com/p/101027069)
- [conda 虚拟环境中添加临时环境变量 LD_LIBRARY_PATH(解决/usr/lib/libstdc++.so.6: version `GLIBCXX_3.4.20’ not found)](https://blog.csdn.net/jillar/article/details/116494270)
- [【pytorch】/libstdc++.so.6: version `CXXABI_1.3.11‘ not found](https://blog.csdn.net/answer3664/article/details/108648139)

```bash
libstdc++.so.6 cxxabi not found
strings /usr/lib/libstdc++.so.6 | grep CXXABI
strings /usr/lib64/libstdc++.so.6 | grep CXXABI
# 可以检查下这几个路径下的
```