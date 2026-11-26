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
                  quantization_algorithm='w4a16_g128')
print('build ret:', ret)

ret = rkllm.export_rkllm(OUT)
print('export ret:', ret, '->', OUT)