import json
import requests

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Dict, Optional, List, AsyncIterator, Iterator

from embedding_server.types import (
    EmbedResponse,
    EmbedRequest, InfoResponse, TokenCountResponse,
)
from embedding_server.errors import parse_error


class Client:
    """Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from embedding_server import Client

     >>> client = Client("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> client.embed("Why is the sky blue?").embedding
     [0.1, 0.2, 0.3, ...]
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    def info(self) -> InfoResponse:
        """
        Get the model info

        Returns:
            InfoResponse: model info
        """
        resp = requests.get(
            self.base_url + "/info",
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return InfoResponse(**payload)

    def embed(
        self,
        inputs: str,
    ) -> EmbedResponse:
        """
        Given a prompt, generate the following text

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            EmbedResponse: embedding for the text
        """
        request = EmbedRequest(inputs=inputs)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return EmbedResponse(**payload)

    def token_count(
        self,
        inputs: str,
    ) -> TokenCountResponse:
        """
        Given a prompt, generate the following text

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            EmbedResponse: embedding for the text
        """
        request = EmbedRequest(inputs=inputs)

        resp = requests.post(
            self.base_url + "/token_count",
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return TokenCountResponse(**payload)


class AsyncClient:
    """Asynchronous Client to make calls to a text-generation-inference instance

     Example:

     ```python
     >>> from embedding_server import AsyncClient

     >>> client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> response = await client.embed("Why is the sky blue?")
     >>> response.embedding
     [0.1, 0.2, 0.3, ...]
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout * 60)

    async def info(self) -> InfoResponse:
        """
        Get the model info

        Returns:
            InfoResponse: model info
        """
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.get(self.base_url + "/info") as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return InfoResponse(**payload)

    async def embed(
        self,
        inputs: str,
    ) -> EmbedResponse:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            EmbedResponse: embedding for the text
        """
        request = EmbedRequest(inputs=inputs)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return EmbedResponse(**payload)

    async def token_count(
        self,
        inputs: str,
    ) -> TokenCountResponse:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            EmbedResponse: embedding for the text
        """
        request = EmbedRequest(inputs=inputs)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url + "/token_count", json=request.dict()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return TokenCountResponse(**payload)
