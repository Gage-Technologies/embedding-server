import pytest

from embedding_server import Client, AsyncClient
from embedding_server.errors import NotFoundError, ValidationError


def test_embed(flan_t5_xxl_url, hf_headers):
    client = Client(flan_t5_xxl_url, hf_headers)
    response = client.embed("test")

    assert response.embedding is not None
    assert len(response.embedding) == 1024
    assert response.dim == 1024


def test_generate_not_found(fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        client.embed("test")


@pytest.mark.asyncio
async def test_generate_async(flan_t5_xxl_url, hf_headers):
    client = AsyncClient(flan_t5_xxl_url, hf_headers)
    response = await client.embed("test")

    assert response.embedding is not None
    assert len(response.embedding) == 1024
    assert response.dim == 1024


@pytest.mark.asyncio
async def test_generate_async_not_found(fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        await client.embed("test")
