import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer

from embedding_server.models.types import Batch, Embedding
from embedding_server.pb.embedding_pb2 import InfoResponse

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model.eval()
        test_emb = self.model.encode(["test"], convert_to_numpy=False, convert_to_tensor=True)
        self.dim = int(test_emb.shape[-1])
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.check_initialized()

    @property
    def info(self) -> InfoResponse:
        return InfoResponse(
            requires_padding=True,
            dtype=str(self.dtype),
            device_type=self.device.type,
            dim=self.dim,
        )

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, batch: B) -> List[Embedding]:
        raise NotImplementedError

    def check_initialized(self):
        uninitialized_parameters = []
        for n, p in self.model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model {self.__class__.__name__}: {uninitialized_parameters}"
            )
