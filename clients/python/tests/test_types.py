import pytest

from embedding_server.types import EmbedRequest
from embedding_server.errors import ValidationError


def test_request_validation():
    EmbedRequest(inputs="test")

    with pytest.raises(ValidationError):
        EmbedRequest(inputs="")

    EmbedRequest(inputs="test")
    EmbedRequest(inputs="test")

    with pytest.raises(ValidationError):
        EmbedRequest(
            inputs="test", stream=True
        )
