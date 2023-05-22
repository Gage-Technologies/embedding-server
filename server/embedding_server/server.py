import asyncio
import os
import torch

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from embedding_server.cache import Cache
from embedding_server.interceptor import ExceptionInterceptor
from embedding_server.models import Model, get_model
from embedding_server.pb import embedding_pb2_grpc, embedding_pb2
from embedding_server.tracing import UDSOpenTelemetryAioServerInterceptor


class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of EmbeddingService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return embedding_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return embedding_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        if request.HasField("id"):
            self.cache.delete(request.id)
        else:
            self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embedding_pb2.ClearCacheResponse()

    async def Embed(self, request, context):
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.device
        )

        executions = self.model.embed(batch)

        return embedding_pb2.EmbedResponse(
            embeddings=[e.to_pb() for e in executions],
        )

def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    uds_path: Path,
):
    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
    ):
        unix_socket_template = "unix://{}-{}"
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model(model_id, revision, sharded, quantize)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
            EmbeddingService(model, Cache(), server_urls), server
        )
        SERVICE_NAMES = (
            embedding_pb2.DESCRIPTOR.services_by_name["EmbeddingService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(model_id, revision, sharded, quantize))
