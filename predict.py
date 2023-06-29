import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession('./ckpt/model.onnx')

for name_and_shape in session.get_inputs():
    print(name_and_shape)
for name_and_shape in session.get_outputs():
    print(name_and_shape)

input_data = {
    "inputs_seq:0": np.array([
        [444, 1804,  663,  519 , 3],
        [444, 1804,  514,  519 , 3],
    ], dtype=np.int32),
    "inputs_seq_len:0": np.array([5, 5], dtype=np.int32)
}
# output_names = ["projection/transitions:0", "projection/Softmax:0", "projection/cond_2/ReverseSequence_1:0"]
output_names = ["projection/dense/BiasAdd:0", "projection/Softmax:0", "projection/transitions:0"]

for key, val in input_data.items():
    print(key, val.shape)
print("=====输出")
output_data = session.run(output_names, input_data)
print(output_data)
print(len(output_data))
for name, val in zip(output_names, output_data):
    print(name, val.shape)


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


for logit in output_data[0]:
    preds_seq, crf_scores = viterbi_decode(logit, output_data[2])
    print(preds_seq)

"""
试试 numpy 进行解码
"""
