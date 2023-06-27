import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession('./ckpt/model.onnx')
# outputs = session.run([output names], inputs)

for name_and_shape in session.get_inputs():
    print(name_and_shape)
for name_and_shape in session.get_outputs():
    print(name_and_shape)

input_data = {
    "inputs_seq:0": np.array([
        [444, 1804,  663,  519 , 3],
        [444, 1804,  663,  519 , 3],
    ], dtype=np.int32),
    "inputs_seq_len:0": np.array([5, 5], dtype=np.int32)
}
output_names = ["projection/transitions:0", "projection/Softmax:0"]

for key, val in input_data.items():
    print(key, val.shape)
print("=====输出")
output_data = session.run(output_names, input_data)
print(output_data)
print(len(output_data))
print(output_data[0].shape)
print(output_data[1].shape)
