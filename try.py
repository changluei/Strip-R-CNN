import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)

import transformers.generation as tg
from transformers.generation.utils import GenerateDecoderOnlyOutput

# 把 generation.utils 里的类补到 generation 命名空间
tg.GenerateDecoderOnlyOutput = GenerateDecoderOnlyOutput

try:
    import mamba_ssm
    print("mamba_ssm:", mamba_ssm.__file__)
    from mamba_ssm import Mamba
    print("Mamba import ok")
except Exception as e:
    print(type(e).__name__, e)