import pytest

from embedding_server.types import Request
from embedding_server.errors import ValidationError


def test_request_validation():
    Request(inputs="test")

    with pytest.raises(ValidationError):
        Request(inputs="")

    Request(inputs="test")
    Request(inputs="test")

    with pytest.raises(ValidationError):
        Request(
            inputs="test", stream=True
        )
