import os
import torch
import warnings

torch.set_float32_matmul_precision('medium')

os.environ["NCCL_DEBUG"] = "WARN"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
