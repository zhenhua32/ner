import tensorflow as tf


ckpt = tf.train.load_checkpoint('./ckpt/model.ckpt.batch8')
for name_and_shape in ckpt.get_variable_to_shape_map().items():
    print(name_and_shape)

print("========================")

import tensorflow as tf
reader = tf.train.NewCheckpointReader('./ckpt/model.ckpt.batch8')
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    print(variable_name, all_variables[variable_name])
    # print(reader.get_tensor(variable_name))

