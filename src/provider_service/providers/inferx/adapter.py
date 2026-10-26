"""Async wrapper around InferxBackend for the orchestrator and deployment manager."""

from __future__ import annotations

import asyncio
from typing import Any

from provider_service.providers.inferx import app as inferx_app
from provider_service.providers.schema import InstanceHealth, ModelStatusSummary, tail_log_text
from provider_service.settings import cuda_visible_devices_value
from provider_service.inferx_client import (
    DEFAULT_FUNCTION_IMAGE,
    DEFAULT_GPU_COUNT,
    DEFAULT_VRAM_MB,
    InferxBackend,
    InferxError,
    build_vllm_serve_commands,
    inferx_fprevision_from_pod_name,
    inferx_function_name,
    inferx_funccall_url,
    inferx_scaling_policy,
    resolve_inferx_base_url,
)


def _gpu_envs(gpu_count: int) -> list[list[str]]:
    return [["CUDA_VISIBLE_DEVICES", cuda_visible_devices_value(gpu_count)]]


class InferxAdapter:
    backend_name = "inferx"

    def __init__(self, backend: InferxBackend | None = None):
        self.backend = backend or InferxBackend()
        self._min_instances: dict[str, int] = {}

    @property
    def base_url(self) -> str:
        return self.backend.base_url or resolve_inferx_base_url()

    def request_api_key(self) -> str:
        return self.backend.api_key

    def footprint_url(self, model_id: str) -> str:
        del model_id
        return self.base_url

    def model_inference_url(self, model_id: str) -> str:
        return inferx_funccall_url(
            base_url=self.base_url,
            tenant=self.backend.tenant,
            function_name=model_id,
        )

    def openai_api_base(self, model_id: str) -> str:
        return inferx_funccall_url(
            base_url=self.base_url,
            tenant=self.backend.tenant,
            function_name=model_id,
            openai_v1=True,
        )

    async def list_functions(self) -> list[dict[str, str]]:
        return await asyncio.to_thread(self.backend.list_functions)

    async def list_instances(self) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self.backend.list_instances)
        min_map = getattr(self, "_min_instances", None) or {}
        return inferx_app.normalize_pod_instances(
            rows,
            min_instances_by_function=min_map,
        )

    def summarize_instances(
        self,
        pods: list[dict[str, Any]],
        *,
        min_instances: int = 0,
    ) -> ModelStatusSummary:
        return inferx_app.summarize_pod_instances(pods, min_instances=min_instances)

    def summarize_models(self, models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        from provider_service.providers.schema import collect_model_summaries

        rows = inferx_app.normalize_pod_instances(
            self.backend.list_instances(),
            min_instances_by_function=getattr(self, "_min_instances", None) or {},
        )
        return collect_model_summaries(self, rows, models or [])

    async def deploy_model(
        self,
        *,
        model_id: str,
        hf_model: str,
        task: str | None = None,
        revision: str = "main",
        additional_args: dict[str, Any] | None = None,
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
        gpu_count: int = DEFAULT_GPU_COUNT,
        vram_mb: int = DEFAULT_VRAM_MB,
        mode: str = "create",
        sample_query: dict[str, Any] | None = None,
        hf_token: str | None = None,
        precision: str | None = None,
        quantization: str | None = None,
        artifact_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Deploy a managed model."""
        del client_id
        pipeline = str(task or "").strip()
        return await self.deploy_vllm_function(
            model_id=model_id,
            hf_model=hf_model,
            revision=revision,
            additional_args=additional_args,
            min_instances=min_instances,
            max_instances=max_instances,
            scale_up_threshold=scale_up_threshold,
            worker_timeout=worker_timeout,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            mode=mode,
            sample_query=sample_query,
            task=pipeline,
            hf_token=hf_token,
            precision=precision,
            quantization=quantization,
            artifact_id=artifact_id,
        )

    async def deploy_vllm_function(
        self,
        *,
        model_id: str,
        hf_model: str,
        revision: str = "main",
        additional_args: dict[str, Any] | None = None,
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
        gpu_count: int = DEFAULT_GPU_COUNT,
        vram_mb: int = DEFAULT_VRAM_MB,
        mode: str = "create",
        sample_query: dict[str, Any] | None = None,
        task: str | None = None,
        hf_token: str | None = None,
        precision: str | None = None,
        quantization: str | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        del task, hf_token, precision, quantization, artifact_id
        name = inferx_function_name(model_id)
        commands = build_vllm_serve_commands(
            hf_model,
            model_id=model_id,
            revision=revision,
            gpu_count=gpu_count,
            additional_args=additional_args,
        )
        policy = inferx_scaling_policy(
            min_instances,
            max_instances,
            scale_up_threshold=scale_up_threshold,
            worker_timeout=worker_timeout,
        )
        created = await asyncio.to_thread(
            self.backend.deploy_function,
            name,
            commands,
            hf_model=hf_model,
            image=DEFAULT_FUNCTION_IMAGE,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            editor_mode="advanced",
            mode=mode,
            sample_query=sample_query,
            policy=policy,
            envs=_gpu_envs(gpu_count),
        )
        await asyncio.to_thread(self.backend.upsert_function_policy, name, policy)
        if hasattr(self, "_min_instances"):
            self._min_instances[name] = max(0, int(min_instances or 0))
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
        gpu_count: int = DEFAULT_GPU_COUNT,
        vram_mb: int = DEFAULT_VRAM_MB,
        mode: str = "create",
    ) -> dict[str, Any]:
        name = inferx_function_name(model_id)
        policy = inferx_scaling_policy(
            min_instances,
            max_instances,
            scale_up_threshold=scale_up_threshold,
            worker_timeout=worker_timeout,
        )
        created = await asyncio.to_thread(
            self.backend.deploy_function,
            name,
            commands,
            hf_model="",
            image=image,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            editor_mode="advanced",
            mode=mode,
            policy=policy,
            envs=_gpu_envs(gpu_count),
        )
        await asyncio.to_thread(self.backend.upsert_function_policy, name, policy)
        if hasattr(self, "_min_instances"):
            self._min_instances[name] = max(0, int(min_instances or 0))
        return created

    async def upsert_scaling_policy(
        self,
        *,
        model_id: str,
        min_instances: int = 0,
        max_instances: int = 1,
        scale_up_threshold: float | None = None,
        worker_timeout: float | None = None,
    ) -> dict[str, Any]:
        name = inferx_function_name(model_id)
        policy = inferx_scaling_policy(
            min_instances,
            max_instances,
            scale_up_threshold=scale_up_threshold,
            worker_timeout=worker_timeout,
        )
        if hasattr(self, "_min_instances"):
            self._min_instances[name] = max(0, int(min_instances or 0))
        return await asyncio.to_thread(self.backend.upsert_function_policy, name, policy)

    async def delete_function(self, model_id: str) -> dict[str, Any]:
        name = inferx_function_name(model_id)
        if hasattr(self, "_min_instances"):
            self._min_instances.pop(name, None)
        return await asyncio.to_thread(self.backend.delete_function, name)

    async def delete_pod(
        self,
        *,
        funcname: str,
        fprevision: str,
        pod_id: str,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        revision = str(fprevision or "").strip()
        if revision == "":
            revision = inferx_fprevision_from_pod_name(str(pod_id or ""))
        return await asyncio.to_thread(
            lambda: self.backend.delete_pod(
                funcname=funcname,
                fprevision=revision,
                pod_id=pod_id,
                tenant=tenant,
                namespace=namespace,
            )
        )

    async def get_model_logs(self, model_id: str, *, tail: int = 200) -> str:
        """Recent stdout from each InferX pod running this model."""
        return await asyncio.to_thread(self._model_logs, model_id, max(1, int(tail)))

    def _model_logs(self, model_id: str, tail: int) -> str:
        name = inferx_function_name(model_id)
        sections: list[str] = []
        for row in self.backend.list_instances():
            if str(row.get("funcname") or "") != name:
                continue
            revision = str(row.get("fprevision") or "")
            pod_id = str(row.get("id") or "")
            if revision == "" or pod_id == "":
                continue
            body = self.backend.read_pod_log(
                funcname=name,
                fprevision=revision,
                pod_id=pod_id,
                tenant=row.get("tenant") or None,
                namespace=str(row.get("namespace") or "default"),
            )
            state = str(row.get("state") or "unknown")
            sections.append(f"----- {pod_id} ({state}) -----\n{tail_log_text(body, tail)}")
        return "\n".join(sections)

    def deployment_storage_gb(self, model_id: str) -> float:
        """InferX list/save routes do not return a storage size for a function."""
        del model_id
        return 0.0

    async def check_persistent_instances(self, model_id: str) -> list[InstanceHealth]:
        del model_id
        return []


def is_inferx_error(exc: BaseException) -> bool:
    return isinstance(exc, InferxError)
