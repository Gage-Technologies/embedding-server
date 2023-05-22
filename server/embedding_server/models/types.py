import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from embedding_server.pb import embedding_pb2
from embedding_server.pb.embedding_pb2 import Embedding


class Batch(ABC):
    @abstractmethod
    def to_pb(self) -> embedding_pb2.Batch:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
        cls,
        pb: embedding_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class Embedding:
    embedding: List[float]
    dim: int

    def to_pb(self) -> embedding_pb2.Embedding:
        return embedding_pb2.Embedding(
            embedding=self.embedding,
            dim=self.dim
        )


@dataclass
class Execution:
    request_id: int
    embedding: Optional[Embedding]

    def to_pb(self) -> embedding_pb2.Execution:
        return embedding_pb2.Execution(
            request_id=self.request_id,
            embedding=self.embedding.to_pb() if self.embedding is not None else None,
        )
