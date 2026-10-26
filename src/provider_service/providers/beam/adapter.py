"""Async wrapper around Beam Pods for the orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any

from provider_service.providers.beam import app as beam_app
from provider_service.providers.schema import ApplicationLogs, InferenceRoute, InstanceHealth, ModelStatusSummary
from provider_service.settings import (
    InferenceBackendError,
    beam_api_token,
    beam_target_concurrency,
)
from provider_service.inferx_client import normalize_commands
from provider_service.url import build_url


class _RouteBook:
    """In-flight counts for one model. A persistent replica accepts several requests."""

    def __init__(
        self,
        persistent_url: str,
        burst_url: str,
        persistent_slots: int,
        persistent_instances: int = 0,
        per_replica: int = 0,
    ) -> None:
        self.persistent_url = persistent_url
        self.burst_url = burst_url
        self.persistent_instances = max(0, int(persistent_instances))
        self.per_replica = max(1, int(per_replica or 1))
        if self.persistent_instances and per_replica:
            self.persistent_slots = self.persistent_instances * self.per_replica
        else:
            self.persistent_slots = max(0, int(persistent_slots))
        self.persistent_inflight = 0
        self.burst_inflight = 0

    def set_per_replica(self, per_replica: int) -> None:
        self.per_replica = max(1, int(per_replica))
        if self.persistent_instances:
            self.persistent_slots = self.persistent_instances * self.per_replica

    def take(self) -> tuple[str, str]:
        if self.persistent_url and self.persistent_inflight < self.persistent_slots:
            self.persistent_inflight += 1
            return "persistent", self.persistent_url
        if self.burst_url:
            self.burst_inflight += 1
            return "burst", self.burst_url
        if self.persistent_url:
            self.persistent_inflight += 1
            return "persistent", self.persistent_url
        raise InferenceBackendError("Beam has no URL for this model")

    def release(self, role: str) -> None:
        if role == "persistent":
            self.persistent_inflight = max(0, self.persistent_inflight - 1)
        elif role == "burst":
            self.burst_inflight = max(0, self.burst_inflight - 1)


class BeamAdapter:
    backend_name = "beam"

    def __init__(self) -> None:
        self._urls: dict[str, str] = {}
        self._routes: dict[str, _RouteBook] = {}

    @property
    def base_url(self) -> str:
        return ""

    def request_api_key(self) -> str:
        from provider_service.settings import beam_api_token

        return beam_api_token()

    def remember_url(self, model_id: str, url: str) -> None:
        self._urls[str(model_id)] = str(url).rstrip("/")

    def model_inference_url(self, model_id: str) -> str:
        cached = self._urls.get(str(model_id))
        if cached:
            return cached
        url = beam_app.lookup_pod_url(model_id)
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
        return await asyncio.to_thread(beam_app.list_pod_functions)

    async def list_instances(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(beam_app.list_pod_instances)

    def summarize_instances(
        self,
        pods: list[dict[str, Any]],
        *,
        min_instances: int = 0,
    ) -> ModelStatusSummary:
        return beam_app.summarize_pod_instances(pods, min_instances=min_instances)

    def summarize_models(self, models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        from provider_service.providers.schema import collect_model_summaries

        return collect_model_summaries(self, beam_app.list_pod_instances(), models or [])

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
        readiness_path: str | None = None,
    ) -> dict[str, Any]:
        if not command:
            raise InferenceBackendError("command is required")
        if not str(image or "").strip():
            raise InferenceBackendError("image is required")
        limit = beam_target_concurrency(request_limit=request_limit)
        created = await asyncio.to_thread(
            beam_app.deploy_inference_pod,
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
            tasks_per_container=limit,
            extra_env=extra_env,
            port=port,
            readiness_path=readiness_path,
        )
        self._remember_capacity(model_id, created)
        return created

    def _remember_capacity(self, model_id: str, created: dict[str, Any]) -> None:
        self.remember_url(model_id, created["url"])
        persistent = int(created.get("persistent_instances") or 0)
        per_replica = int(created.get("tasks_per_container") or beam_target_concurrency())
        self._routes[str(model_id)] = _RouteBook(
            persistent_url=str(created.get("persistent_url") or ""),
            burst_url=str(created.get("burst_url") or ""),
            persistent_instances=persistent,
            per_replica=per_replica,
            persistent_slots=persistent * per_replica,
        )

    async def _book(self, model_id: str) -> _RouteBook:
        key = str(model_id)
        book = self._routes.get(key)
        if book is not None:
            return book
        described = await asyncio.to_thread(beam_app.describe_routes, model_id)
        persistent = int(described.get("persistent_instances") or 0)
        per_replica = beam_target_concurrency()
        book = _RouteBook(
            persistent_url=str(described.get("persistent_url") or ""),
            burst_url=str(described.get("burst_url") or ""),
            persistent_instances=persistent,
            per_replica=per_replica,
            persistent_slots=persistent * per_replica,
        )
        self._routes[key] = book
        if book.persistent_url or book.burst_url:
            self.remember_url(model_id, book.persistent_url or book.burst_url)
        return book

    async def assign_request(
        self,
        model_id: str,
        *,
        task: str | None = None,
        request_limit: int | None = None,
    ) -> InferenceRoute:
        del task
        book = await self._book(model_id)
        book.set_per_replica(beam_target_concurrency(request_limit=request_limit))
        role, url = book.take()
        return InferenceRoute(url=url, role=role)

    def deployment_storage_gb(self, model_id: str) -> float:
        from provider_service.providers.beam.capacity import burst_storage_gb

        return burst_storage_gb(model_id)

    async def release_request(self, model_id: str, role: str) -> None:
        book = self._routes.get(str(model_id))
        if book is not None:
            book.release(role)

    async def check_persistent_instances(self, model_id: str, untrusted: bool = False) -> list[InstanceHealth]:
        book = await self._book(model_id)
        if book.persistent_url == "":
            return []
        state, detail = await asyncio.to_thread(
            _probe_ready,
            book.persistent_url,
            model_id=model_id,
            untrusted=untrusted,
        )
        return [
            InstanceHealth(
                instance_id=model_id,
                state=state,
                url=book.persistent_url,
                detail=detail,
            )
        ]

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
        limit = beam_target_concurrency(request_limit=request_limit)
        created = await asyncio.to_thread(
            beam_app.deploy_inference_pod,
            model_id=model_id,
            command=normalize_commands(commands),
            image=image,
            tasks_per_container=limit,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            min_instances=min_instances,
            max_instances=max_instances,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
            untrusted=True,
        )
        self._remember_capacity(model_id, created)
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
        return await asyncio.to_thread(
            beam_app.update_pod_autoscaler,
            model_id,
            min_instances=min_instances,
            max_instances=max_instances,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
        )

    async def delete_function(self, model_id: str) -> dict[str, Any]:
        self._urls.pop(str(model_id), None)
        self._routes.pop(str(model_id), None)
        return await asyncio.to_thread(beam_app.stop_inference_pod, model_id)

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
        return await asyncio.to_thread(beam_app.stop_pod_container, pod_id)

    async def get_model_logs(self, model_id: str, *, tail: int = 200) -> str:
        """Recent container output for this model's Beam pods."""
        return await asyncio.to_thread(beam_app.get_model_logs, model_id, tail=tail)

    async def read_application_logs(self, model_id: str, *, cursor: str = "") -> ApplicationLogs:
        """Stdout from running custom-image containers. Beam system lines are omitted."""
        return await asyncio.to_thread(beam_app.read_application_logs, model_id, cursor=cursor)


def _probe_ready(url: str, *, model_id: str = "", untrusted: bool = False) -> tuple[str, str]:
    """Ask the warm replica whether the model is loaded."""
    import httpx

    from provider_service.providers.beam.guest_gate import edge_headers

    ready_url = build_url(url, "/model/ready")
    headers = {}
    token = beam_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if untrusted:
        headers.update(edge_headers(model_id, token))
    try:
        response = httpx.get(ready_url, headers=headers, timeout=10.0)
    except httpx.HTTPError as exc:
        return "unhealthy", str(exc)
    if response.status_code != 200:
        return "unhealthy", f"HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return "unhealthy", "ready probe did not return JSON"
    if isinstance(body, dict) and body.get("model_ready") is True:
        return "healthy", ""
    return "unhealthy", "model is not ready"
