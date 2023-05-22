import torch

from loguru import logger
from transformers import AutoConfig
from transformers.models.auto import modeling_auto
from typing import Optional

from embedding_server.models.model import Model
from embedding_server.models.sentence_transformer import SentenceTransformerModel

__all__ = [
    "Model",
    "get_model",
]

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str, revision: Optional[str], sharded: bool, quantize: Optional[str]
) -> Model:
    if sharded:
        raise ValueError("sharded is not supported yet")

    if quantize is not None:
        raise ValueError("quantize is not supported yet")

    return SentenceTransformerModel(model_id, revision)
