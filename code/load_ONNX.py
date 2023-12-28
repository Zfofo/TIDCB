# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb

import onnxruntime
import numpy as np
import time

assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()

ort_session = onnxruntime.InferenceSession("/aicity/my_TIPCB/best_th.onnx", providers=["CUDAExecutionProvider"])
for i in ort_session.get_inputs():
    print(i.name)

# ort_inputs = {
#     "images": np.random.randn(128, 3, 384, 128).astype(np.float32),
#     "txt": np.random.randint(128, 25000, (1,64), dtype=np.int64),
#     "attention_mask": np.ones((128,64), dtype=np.int64)    
# }

ort_inputs = {
    "images": np.random.randn(128, 3, 384, 128).astype(np.float32),
    "txt": np.ones([128, 64], dtype=np.int64),
    "attention_mask": np.ones((128,64), dtype=np.int64)    
}


start = time.time()
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)
print(ort_outs[0].shape)
print("time taken:", time.time()-start)
