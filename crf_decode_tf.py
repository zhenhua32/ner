def crf_decode(potentials, transition_params, sequence_length):
  """Decode the highest scoring sequence of tags in TensorFlow.

  This is a function for tensor.

  Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor of
              unary potentials.
    transition_params: A [num_tags, num_tags] matrix of
              binary potentials.
    sequence_length: A [batch_size] vector of true sequence lengths.

  Returns:
    decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                Contains the highest scoring tag indices.
    best_score: A [batch_size] vector, containing the score of `decode_tags`.
  """
  # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
  # and the max activation.
  def _single_seq_fn():
    squeezed_potentials = array_ops.squeeze(potentials, [1])
    decode_tags = array_ops.expand_dims(
        math_ops.argmax(squeezed_potentials, axis=1), 1)
    best_score = math_ops.reduce_max(squeezed_potentials, axis=1)
    return math_ops.cast(decode_tags, dtype=dtypes.int32), best_score

  def _multi_seq_fn():
    """Decoding of highest scoring sequence."""

    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    num_tags = tensor_shape.dimension_value(potentials.shape[2])

    # Computes forward decoding. Get last score and backpointers.
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
    initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
    inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
    # Sequence length is not allowed to be less than zero.
    sequence_length_less_one = math_ops.maximum(
        constant_op.constant(0, dtype=sequence_length.dtype),
        sequence_length - 1)
    backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
        crf_fwd_cell,
        inputs=inputs,
        sequence_length=sequence_length_less_one,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)
    backpointers = gen_array_ops.reverse_sequence(  # [B, T - 1, O]
        backpointers, sequence_length_less_one, seq_dim=1)

    # Computes backward decoding. Extract tag indices from backpointers.
    crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
    initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),  # [B]
                                  dtype=dtypes.int32)
    initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
    decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
        crf_bwd_cell,
        inputs=backpointers,
        sequence_length=sequence_length_less_one,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)
    decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
    decode_tags = array_ops.concat([initial_state, decode_tags],   # [B, T]
                                   axis=1)
    decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
        decode_tags, sequence_length, seq_dim=1)

    best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
    return decode_tags, best_score

  return utils.smart_cond(
      pred=math_ops.equal(tensor_shape.dimension_value(potentials.shape[1]) or
                          array_ops.shape(potentials)[1], 1),
      true_fn=_single_seq_fn,
      false_fn=_multi_seq_fn)
