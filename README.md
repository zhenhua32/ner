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

python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/transitions:0,projection/Softmax:0,projection/cond_2/ReverseSequence_1:0
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

导出时有报错, 完整的日志如下:

```log
rojection/transitions:0,projection/Softmax:0,projection/cond_2/ReverseSequence_1:0
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
C:\Anaconda3\envs\ner\lib\site-packages\tensorflow\python\framework\dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
C:\Anaconda3\envs\ner\lib\runpy.py:125: RuntimeWarning: 'tf2onnx.convert' found in sys.modules after import of package 'tf2onnx', but prior to execution of 'tf2onnx.convert'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
2023-06-28 21:28:10,189 - INFO - Using tensorflow=1.13.1, onnx=1.14.0, tf2onnx=1.14.0/8f8d49
2023-06-28 21:28:10,190 - INFO - Using opset <onnx, 15>
2023-06-28 21:28:10,568 - INFO - Computed 0 values for constant folding
2023-06-28 21:28:10,982 - ERROR - Tensorflow op [projection/cond_2/sub/Switch: Switch] is not supported
2023-06-28 21:28:10,983 - ERROR - Tensorflow op [projection/cond_2/Slice/Switch: Switch] is not supported
2023-06-28 21:28:10,986 - ERROR - Tensorflow op [projection/cond_2/ExpandDims_1/Switch: Switch] is not supported
2023-06-28 21:28:11,010 - ERROR - Unsupported ops: Counter({'Switch': 3})
2023-06-28 21:28:11,031 - INFO - Optimizing ONNX model
2023-06-28 21:28:11,955 - INFO - After optimization: Cast -12 (38->26), Concat -3 (12->9), Const -118 (154->36), Expand -2 (6->4), Gather +2 (3->5), Identity -19 (19->0), Reshape -9 (16->7), Squeeze -5 (11->6), Transpose -3 (9->6), Unsqueeze -9 (18->9)
2023-06-28 21:28:11,995 - INFO - 
2023-06-28 21:28:11,995 - INFO - Successfully converted TensorFlow model model.ckpt.batch8.meta to ONNX
2023-06-28 21:28:11,996 - INFO - Model inputs: ['inputs_seq:0', 'inputs_seq_len:0']
2023-06-28 21:28:11,996 - INFO - Model outputs: ['projection/transitions:0', 'projection/Softmax:0', 'projection/cond_2/ReverseSequence_1:0']
2023-06-28 21:28:11,996 - INFO - ONNX model is saved at model.onnx
```

`Switch] is not supported` 估计是因为 crf 解码时有个条件判断.


然后用 ONNX 推理就会报错, 加载模型时就出错了.

```log
Traceback (most recent call last):
  File ".\predict.py", line 4, in <module>
    session = onnxruntime.InferenceSession('./ckpt/model.onnx')
  File "C:\Anaconda3\envs\ner\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 360, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "C:\Anaconda3\envs\ner\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 397, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from ./ckpt/model.onnx failed:This is an invalid model. In Node, ("projection/cond_2/ExpandDims_1/Switch", Switch, "", -1) : ("projection/transitions:0": tensor(float),"projection/Equal_2:0": tensor(bool),) -> ("projection/cond_2/ExpandDims_1/Switch:0","projection/cond_2/ExpandDims_1/Switch:1",) , Error No Op registered for Switch with domain_version of 15
```
