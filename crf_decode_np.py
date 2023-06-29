import numpy as np

n_states = 3  # number of possible tags
n_features = 4  # number of input features
W = np.random.randn(n_states, n_features)  # state feature weights
T = np.random.randn(n_states, n_states)  # transition feature weights

n_steps = 5  # length of the input sequence
X = np.random.randn(n_steps, n_features)  # input feature matrix
y = np.array([0, 1, 2, 1, 0])  # true label sequence


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
        print(W.shape, X[i].shape, T.shape, score[i - 1][:, None].shape)
        breakpoint()
        # TODO: 这里报错
        score[i] = W.dot(X[i]) + T + score[i - 1][:, None]
        backpointer[i] = np.argmax(score[i], axis=0)
        # Normalize the score to avoid numerical issues
        score[i] -= np.max(score[i])
    # Trace back the optimal path from the last step
    y_pred = np.zeros(n_steps, dtype=np.int32)
    y_pred[-1] = np.argmax(score[-1])
    for i in range(n_steps - 2, -1, -1):
        y_pred[i] = backpointer[i + 1][y_pred[i + 1]]
    # Return the optimal path and the maximum log-likelihood
    return y_pred, np.max(score[-1])


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




# y_pred, score = viterbi_decode(W, T, X)
# 这个解码每个序列的, 所以 X 的 shape 是 (序列长, 标签数). T 的 shape 是 (标签数, 标签数)
X = np.random.randn(n_steps, n_states)
y_pred, score = viterbi_decode(X, T)
print(y_pred)
print(score)
