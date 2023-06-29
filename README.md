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

python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/dense/BiasAdd:0,projection/Softmax:0,projection/transitions:0

python -m tf2onnx.convert --checkpoint model.ckpt.batch8.meta --output model.onnx --inputs inputs_seq:0,inputs_seq_len:0 --outputs projection/transitions:0,projection/Softmax:0,projection/cond_2/ReverseSequence:0
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

# 用 numpy 实现 crf decode

您好，感谢您使用必应搜索。根据您的搜索词，我为您找到了以下信息：

## numpy crf decode

numpy crf decode 是指在 numpy 中使用条件随机场（CRF）模型进行序列标注任务的解码过程。CRF 是一种概率图模型，可以用于建模序列数据的条件概率分布 ¹。

要在 numpy 中实现 crf decode，有以下几个步骤：

- 步骤一：定义 CRF 模型的参数。CRF 模型的参数包括状态特征权重矩阵 W 和转移特征权重矩阵 T。W 的形状为 (n_states, n_features)，T 的形状为 (n_states, n_states)。例如：

```python
import numpy as np
n_states = 3 # number of possible tags
n_features = 4 # number of input features
W = np.random.randn(n_states, n_features) # state feature weights
T = np.random.randn(n_states, n_states) # transition feature weights
```

- 步骤二：定义输入序列的特征矩阵 X 和真实标签序列 y。X 的形状为 (n_steps, n_features)，y 的形状为 (n_steps,)。例如：

```python
n_steps = 5 # length of the input sequence
X = np.random.randn(n_steps, n_features) # input feature matrix
y = np.array([0, 1, 2, 1, 0]) # true label sequence
```

- 步骤三：定义 Viterbi 算法的函数。Viterbi 算法是一种动态规划算法，可以用于寻找最可能的标签序列 ²。Viterbi 算法的函数接受 W，T 和 X 作为输入，返回最优标签序列 y_pred 和最大对数似然值 score。例如：

```python
def viterbi_decode(W, T, X):
    n_steps, n_features = X.shape
    n_states = W.shape[0]
    # Initialize the score matrix and the backpointer matrix
    score = np.zeros((n_steps, n_states))
    backpointer = np.zeros((n_steps, n_states), dtype=np.int32)
    # Compute the score and the backpointer for the first step
    score[0] = W.dot(X[0])
    backpointer[0] = -1
    # Loop over the remaining steps
    for i in range(1, n_steps):
        # Compute the score and the backpointer for the current step
        score[i] = W.dot(X[i]) + T + score[i-1][:, None]
        backpointer[i] = np.argmax(score[i], axis=0)
        # Normalize the score to avoid numerical issues
        score[i] -= np.max(score[i])
    # Trace back the optimal path from the last step
    y_pred = np.zeros(n_steps, dtype=np.int32)
    y_pred[-1] = np.argmax(score[-1])
    for i in range(n_steps-2, -1, -1):
        y_pred[i] = backpointer[i+1][y_pred[i+1]]
    # Return the optimal path and the maximum log-likelihood
    return y_pred, np.max(score[-1])
```

- 步骤四：调用 Viterbi 算法的函数，得到最优标签序列 y_pred 和最大对数似然值 score。例如：

```python
y_pred, score = viterbi_decode(W, T, X)
print(y_pred)
print(score)
```

以下是一些相关的网页链接：

- [NumPy decode() function - Studytonight](https://www.studytonight.com/numpy/numpy-decode-function)
- [numpy.char.decode — NumPy v1.25 Manual](https://numpy.org/doc/stable/reference/generated/numpy.char.decode.html)
- [numpy.chararray.decode — NumPy v1.25 Manual](https://numpy.org/doc/stable/reference/generated/numpy.chararray.decode.html)
- [条件随机场（CRF）原理及应用 - CSDN 博客](https://blog.csdn.net/qq_16234613/article/details/79451511)
- [Viterbi 算法详解 - CSDN 博客](https://blog.csdn.net/qq_19707521/article/details/78479532)

希望这些信息对您有帮助。如果您还有其他问题，欢迎继续使用必应搜索。😊

源: 与必应的对话， 2023/6/28
(1) numpy.char.decode — NumPy v1.25 Manual. https://numpy.org/doc/stable/reference/generated/numpy.char.decode.html.
(2) numpy.chararray.decode — NumPy v1.25 Manual. https://numpy.org/doc/stable/reference/generated/numpy.chararray.decode.html.
(3) NumPy decode() function - Studytonight. https://www.studytonight.com/numpy/numpy-decode-function.

# spark --archives

您好，感谢您使用必应搜索。根据您的搜索词，我为您找到了以下信息：

## pyspark --archives

pyspark --archives 是指在提交 pyspark 作业时，使用 --archives 参数来传递一些压缩文件，这些文件会被解压到每个执行器的工作目录中。支持的压缩文件格式有 .jar, .tar.gz, .tgz 和 .zip。您可以通过在文件名后加 # 来指定解压后的目录名，例如 file.zip#directory。这个参数是实验性的，自 Spark 3.1.0 版本开始支持 ¹。

要在 pyspark 中获取 --archives 传过来的文件的路径，有以下几种方法：

- 方法一：使用 SparkFiles 模块来获取文件的绝对路径。SparkFiles 模块提供了一个 get () 函数，可以根据文件名或目录名来返回文件的绝对路径 ²。例如：

```python
from pyspark import SparkFiles
# Suppose we submit the job with --archives file.zip#directory
file_path = SparkFiles.get("file.zip") # get the path of the zip file
dir_path = SparkFiles.get("directory") # get the path of the unzipped directory
```

- 方法二：使用 os 模块来获取文件的相对路径。os 模块提供了一个 getcwd () 函数，可以返回当前工作目录的路径 ³。例如：

```python
import os
# Suppose we submit the job with --archives file.zip#directory
cwd = os.getcwd() # get the current working directory
file_path = os.path.join(cwd, "file.zip") # get the relative path of the zip file
dir_path = os.path.join(cwd, "directory") # get the relative path of the unzipped directory
```

以下是一些相关的网页链接：

- [Configuration - Spark 3.4.1 Documentation - Apache Spark](https://spark.apache.org/docs/latest/configuration.html)
- [What's the difference between --archives, --files, py-files in pyspark ...](https://stackoverflow.com/questions/38066318/whats-the-difference-between-archives-files-py-files-in-pyspark-job-argum)
- [PySpark Archives - Spark By {Examples}](https://sparkbyexamples.com/category/pyspark/)

希望这些信息对您有帮助。如果您还有其他问题，欢迎继续使用必应搜索。😊

源: 与必应的对话， 2023/6/28
(1) What's the difference between --archives, --files, py-files in pyspark .... https://stackoverflow.com/questions/38066318/whats-the-difference-between-archives-files-py-files-in-pyspark-job-argum.
(2) Configuration - Spark 3.4.1 Documentation - Apache Spark. https://spark.apache.org/docs/latest/configuration.html.
(3) PySpark Archives - Spark By {Examples}. https://sparkbyexamples.com/category/pyspark/.

# pip list

```
Package              Version
-------------------- ---------
absl-py              1.4.0
appdirs              1.4.3
astor                0.8.1
attrs                23.1.0
black                19.10b0
certifi              2022.12.7
charset-normalizer   3.1.0
click                8.1.3
colorama             0.4.6
coloredlogs          15.0.1
flake8               3.7.9
flatbuffers          2.0.7
gast                 0.5.4
grpcio               1.56.0
h5py                 3.8.0
humanfriendly        10.0
idna                 3.4
importlib-metadata   6.7.0
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.2
Markdown             3.4.3
MarkupSafe           2.1.3
mock                 5.0.2
mpmath               1.3.0
numpy                1.21.6
onnx                 1.14.0
onnxruntime          1.14.1
packaging            23.1
pathspec             0.7.0
pip                  22.3.1
protobuf             3.20.0
pyreadline           2.1
regex                2020.2.20
requests             2.31.0
setuptools           65.6.3
six                  1.16.0
sympy                1.10.1
tensorboard          1.13.1
tensorflow           1.13.1
tensorflow-estimator 1.13.0
termcolor            2.3.0
tf2onnx              1.14.0
toml                 0.10.0
typed-ast            1.4.1
typing_extensions    4.6.3
urllib3              2.0.3
Werkzeug             2.2.3
wheel                0.38.4
wincertstore         0.2
zipp                 3.15.0
```

