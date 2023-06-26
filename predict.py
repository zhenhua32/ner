import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession('./ckpt/model.onnx')
# outputs = session.run([output names], inputs)

for name_and_shape in session.get_inputs():
    print(name_and_shape)
for name_and_shape in session.get_outputs():
    print(name_and_shape)

input_data = {
    "inputs_seq:0": np.array([[ 
        444, 1804,  663,  519 ,   3  ,  3  ,  3  ,358  , 78  ,313 , 150  ,610  ,190 , 313,
        44  ,190 , 313  ,373  ,717 ,  75 ,1881 , 313,  190 , 690 , 190 ,2388 ,   3 , 150,
    610  ,190 , 156 , 126 , 118 ,   3 , 190, 2409 ,  78 ,   3   , 5  ,  4  ,  4  ,  8, 17]], dtype=np.int32),
    "inputs_seq_len:0": np.array([43], dtype=np.int32)
}
output_names = ["projection/dense/kernel:0"]


output_data = session.run(output_names, input_data)
print(output_data)
print(len(output_data))
print(output_data[0].shape)
