import os
os.environ.setdefault('BUILD_CUDA_EXT', '0')
from rkllm.api import RKLLM

MODEL_PATH = os.path.abspath(os.environ['MODEL_DIR'])
OUT = os.environ['OUT']

rkllm = RKLLM()
ret = rkllm.load_huggingface(model=MODEL_PATH)
print('load ret:', ret)

ret = rkllm.build(do_quantization=True,
                  optimization_level=3,
                  quantized_dtype='w8a8',
                  quantized_algorithm='normal',
                  target_platform='rk3588',
                  num_npu_core=3)
print('build ret:', ret)

ret = rkllm.export_rkllm(OUT)
print('export ret:', ret, '->', OUT)