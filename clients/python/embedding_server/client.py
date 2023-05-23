import json
import requests

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Dict, Optional, List, AsyncIterator, Iterator

from embedding_server.types import (
    Response,
    Request,
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

    def embed(
        self,
        inputs: str,
    ) -> Response:
        """
        Given a prompt, generate the following text

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            Response: embedding for the text
        """
        request = Request(inputs=inputs)

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
        return Response(**payload[0])


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

    async def embed(
        self,
        inputs: str,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            Response: embedding for the text
        """
        request = Request(inputs=inputs)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return Response(**payload[0])
