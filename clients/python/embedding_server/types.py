from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional, List

from embedding_server.errors import ValidationError


class EmbedRequest(BaseModel):
    # Prompt
    inputs: str

    @validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v


# `embed` return value
class EmbedResponse(BaseModel):
    # embedding for the passed input
    embedding: List[float]
    # dimensions of the embedding
    dim: int


# `token_count` return value
class TokenCountResponse(BaseModel):
    # number of tokens in the passed input
    count: int


# `info` return value
class InfoResponse(BaseModel):
    # Model info
    model_id: str
    model_sha: Optional[str]
    model_dtype: str
    model_device_type: str
    model_pipeline_tag: Optional[str]
    model_dim: int

    # Router Parameters
    max_concurrent_requests: int
    max_input_length: int
    waiting_served_ratio: float
    max_batch_total_tokens: int
    validation_workers: int

    # Router Info
    version: str
    sha: Optional[str]
    docker_label: Optional[str]


# Inference API currently deployed model
class DeployedModel(BaseModel):
    model_id: str
    sha: str
