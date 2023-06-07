import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, List, Type, Dict

from embedding_server.models import Model
from embedding_server.models.types import (
    Batch,
    Execution,
    Embedding,
)
from embedding_server.pb import embedding_pb2

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

tracer = trace.get_tracer(__name__)


@dataclass
class SentenceTransformerBatch(Batch):
    batch_id: int
    requests: List[embedding_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Texts that will be embedded
    input_texts: List[str]

    def to_pb(self) -> embedding_pb2.Batch:
        return embedding_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=len(self),
        )

    @classmethod
    def from_pb(
        cls,
        pb: embedding_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "SentenceTransformerBatch":
        inputs = []
        requests_idx_mapping = {}

        # Parse batch
        for i, r in enumerate(pb.requests):
            requests_idx_mapping[r.id] = i
            inputs.append(r.inputs)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_texts=inputs
        )

    def __len__(self):
        return len(self.requests)


class SentenceTransformerModel(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        model = SentenceTransformer(model_id, device=str(device), cache_folder=HUGGINGFACE_HUB_CACHE).to(dtype)

        super(SentenceTransformerModel, self).__init__(
            model=model,
            tokenizer=model.tokenizer,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[SentenceTransformerBatch]:
        return SentenceTransformerBatch

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("embed")
    def embed(self, batch: SentenceTransformerBatch) -> List[Execution]:
        # use the native encode function from the generic sentence_transformers module
        embeddings = self.model.encode(
            sentences=batch.input_texts,
            batch_size=len(batch.input_texts),
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=False,
            device=str(self.device)
        )
        assert embeddings.shape[0] == len(batch.requests)
        embeddings = embeddings.cpu()
        dim = embeddings.shape[1]

        out = [
            Execution(
                request_id=req.id,
                embedding=Embedding(
                    embedding=embeddings[i].tolist(),
                    dim=dim
                )
            )
            for i, req in enumerate(batch.requests)
        ]

        return out
