import logging
import tensorflow as tf
import numpy as np
import os

from model_lstm_crf import MyModel
from utils import DataProcessor_LSTM as DataProcessor
from utils import load_vocabulary
from utils import extract_kvpairs_in_bio
from utils import cal_f1_score

# set logging
log_file_path = "./ckpt/run.log"
if os.path.exists(log_file_path):
    os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info("loading vocab...")

# 词表文件和标签文件
w2i_char, i2w_char = load_vocabulary("./data/vocab_char.txt")
w2i_bio, i2w_bio = load_vocabulary("./data/vocab_bioattr.txt")

logger.info("loading data...")

# 输入文件是分成两个的, 一个是输入序列, 一个是输出序列
data_processor_train = DataProcessor(
    "./data/train/input.seq.char",
    "./data/train/output.seq.bioattr",
    w2i_char,
    w2i_bio,
    shuffling=True,
)

data_processor_valid = DataProcessor(
    "./data/test/input.seq.char",
    "./data/test/output.seq.bioattr",
    w2i_char,
    w2i_bio,
    shuffling=True,
)

logger.info("building model...")

model = MyModel(
    embedding_dim=300,
    hidden_dim=300,
    vocab_size_char=len(w2i_char),
    vocab_size_bio=len(w2i_bio),
    use_crf=True,
)

# 计算参数总量
logger.info("model params:")
params_num_all = 0
for variable in tf.trainable_variables():
    params_num = 1
    for dim in variable.shape:
        params_num *= dim
    params_num_all += params_num
    logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
logger.info("all params num: " + str(params_num_all))

logger.info("start training...")
# GPU 选项, 限制下显存, 不要一下子占完
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)

    epoches = 0
    losses = []
    batches = 0
    best_f1 = 0
    batch_size = 32

    while epoches < 20:
        # 获取这一批次的数据
        (
            inputs_seq_batch,
            inputs_seq_len_batch,
            outputs_seq_batch,
        ) = data_processor_train.get_batch(batch_size)

        # 模型的输入字典
        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.outputs_seq: outputs_seq_batch,
        }

        # 第一个批次的时候检查下数据
        if batches == 0:
            logger.info("###### shape of a batch #######")
            logger.info("input_seq: " + str(inputs_seq_batch.shape))
            logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
            logger.info("output_seq: " + str(outputs_seq_batch.shape))
            logger.info("###### preview a sample #######")
            logger.info(
                "input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]])
            )
            logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
            logger.info(
                "output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0]])
            )
            logger.info("###############################")

        # 获取损失
        loss, _ = sess.run([model.loss, model.train_op], feed_dict)
        losses.append(loss)
        batches += 1

        # 一个轮次跑完了, 开始下一个
        if data_processor_train.end_flag:
            data_processor_train.refresh()
            epoches += 1

        def valid(data_processor, max_batches=None, batch_size=1024):
            preds_kvpair = []
            golds_kvpair = []
            batches_sample = 0

            while True:
                (
                    inputs_seq_batch,
                    inputs_seq_len_batch,
                    outputs_seq_batch,
                ) = data_processor.get_batch(batch_size)

                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq: outputs_seq_batch,
                }

                # 预测的序列标签的批次
                preds_seq_batch = sess.run(model.outputs, feed_dict)

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

                if data_processor.end_flag:
                    data_processor.refresh()
                    break

                batches_sample += 1
                if (max_batches is not None) and (batches_sample >= max_batches):
                    break

            p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)

            logger.info("Valid Samples: {}".format(len(preds_kvpair)))
            logger.info(
                "Valid P/R/F1: {} / {} / {}".format(
                    round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)
                )
            )

            return (p, r, f1)

        if batches % 100 == 0:
            logger.info("")
            logger.info("Epoches: {}".format(epoches))
            logger.info("Batches: {}".format(batches))
            # 报告下平均损失
            logger.info("Loss: {}".format(sum(losses) / len(losses)))
            losses = []

            ckpt_save_path = "./ckpt/model.ckpt.batch{}".format(batches)
            logger.info("Path of ckpt: {}".format(ckpt_save_path))
            saver.save(sess, ckpt_save_path)

            # 获取验证集的评测结果, 记录当前最好的效果
            p, r, f1 = valid(data_processor_valid, max_batches=10)
            if f1 > best_f1:
                best_f1 = f1
                logger.info("############# best performance now here ###############")
