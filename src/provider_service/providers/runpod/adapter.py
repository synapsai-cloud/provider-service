"""Async wrapper around Runpod load-balancing endpoints for the orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any

from provider_service.providers.runpod import app as runpod_app
from provider_service.providers.schema import InstanceHealth, ModelStatusSummary
from provider_service.settings import (
    InferenceBackendError,
    runpod_api_key,
    runpod_target_concurrency,
)
from provider_service.inferx_client import normalize_commands
from provider_service.url import build_url


class RunpodAdapter:
    backend_name = "runpod"

    def __init__(self) -> None:
        self._urls: dict[str, str] = {}

    @property
    def base_url(self) -> str:
        return ""

    def request_api_key(self) -> str:
        return runpod_api_key()

    def remember_url(self, model_id: str, url: str) -> None:
        self._urls[str(model_id)] = str(url).rstrip("/")

    def model_inference_url(self, model_id: str) -> str:
        cached = self._urls.get(str(model_id))
        if cached:
            return cached
        url = runpod_app.lookup_endpoint_url(model_id)
        self.remember_url(model_id, url)
        return url

    def openai_api_base(self, model_id: str) -> str:
        return build_url(self.model_inference_url(model_id), "/v1")

    def footprint_url(self, model_id: str) -> str:
        try:
            return self.model_inference_url(model_id)
        except Exception:
            return ""

    async def list_functions(self) -> list[dict[str, str]]:
        return await asyncio.to_thread(runpod_app.list_endpoint_functions)

    async def list_instances(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(runpod_app.list_endpoint_instances)

    def summarize_instances(
        self,
        pods: list[dict[str, Any]],
        *,
        min_instances: int = 0,
    ) -> ModelStatusSummary:
        return runpod_app.summarize_endpoint_instances(pods, min_instances=min_instances)

    def summarize_models(self, models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        from provider_service.providers.schema import collect_model_summaries

        return collect_model_summaries(self, runpod_app.list_endpoint_instances(), models or [])

    async def deploy_model(
        self,
        *,
        model_id: str,
        command: list[str],
        image: str,
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
        gpu_count: int = 1,
        vram_mb: int = 24000,
        hf_token: str | None = None,
        extra_env: dict[str, str] | None = None,
        request_limit: int | None = None,
        port: int | None = None,
        health_path: str | None = None,
    ) -> dict[str, Any]:
        """Start ``command`` from ``image`` on a Runpod endpoint."""
        if not command:
            raise InferenceBackendError("command is required")
        if not str(image or "").strip():
            raise InferenceBackendError("image is required")
        limit = runpod_target_concurrency(request_limit=request_limit)
        created = await asyncio.to_thread(
            runpod_app.deploy_inference_endpoint,
            model_id=model_id,
            command=list(command),
            image=image,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            min_instances=min_instances,
            max_instances=max_instances,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
            hf_token=hf_token,
            request_concurrency=limit,
            extra_env=extra_env,
            port=port,
            health_path=health_path,
        )
        self.remember_url(model_id, created["url"])
        return created

    async def deploy_custom_function(
        self,
        *,
        model_id: str,
        image: str,
        commands: str | list[str],
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
        gpu_count: int = 1,
        vram_mb: int = 24000,
        mode: str = "create",
        request_limit: int | None = None,
    ) -> dict[str, Any]:
        del mode
        limit = runpod_target_concurrency(request_limit=request_limit)
        created = await asyncio.to_thread(
            runpod_app.deploy_inference_endpoint,
            model_id=model_id,
            command=normalize_commands(commands),
            image=image,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            min_instances=min_instances,
            max_instances=max_instances,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
            request_concurrency=limit,
        )
        self.remember_url(model_id, created["url"])
        return created

    def deployment_storage_gb(self, model_id: str) -> float:
        return runpod_app.endpoint_storage_gb(runpod_app.find_endpoint(model_id))

    async def upsert_scaling_policy(
        self,
        *,
        model_id: str,
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            runpod_app.update_endpoint_autoscaler,
            model_id,
            min_instances=min_instances,
            max_instances=max_instances,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
        )

    async def delete_function(self, model_id: str) -> dict[str, Any]:
        self._urls.pop(str(model_id), None)
        return await asyncio.to_thread(runpod_app.stop_inference_endpoint, model_id)

    async def delete_pod(
        self,
        *,
        funcname: str,
        fprevision: str,
        pod_id: str,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        del funcname, fprevision, tenant, namespace
        return await asyncio.to_thread(runpod_app.stop_endpoint_worker, pod_id)

    async def get_model_logs(self, model_id: str, *, tail: int = 200) -> str:
        """Recent container stdout from this model's Runpod workers."""
        return await asyncio.to_thread(runpod_app.get_model_logs, model_id, tail=tail)

    async def check_persistent_instances(self, model_id: str) -> list[InstanceHealth]:
        del model_id
        return []
