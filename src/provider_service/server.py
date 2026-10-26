"""HTTP front for InferX, Modal, Runpod, and Beam.

One process holds the provider adapters. Callers send a backend name and an
operation. Provider credentials are read from this process environment.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from provider_service.providers.schema import (
    ApplicationLogs,
    InferenceRoute,
    InstanceHealth,
    ModelStatusSummary,
    ProviderInstance,
)
from provider_service.settings import InferenceBackendError

app = FastAPI(title="Provider service")

_ADAPTERS: dict[str, Any] = {}
_BACKENDS = ("inferx", "modal", "runpod", "beam")
_OPS = {
    "base_url",
    "request_api_key",
    "remember_url",
    "model_inference_url",
    "openai_api_base",
    "footprint_url",
    "list_functions",
    "list_instances",
    "summarize_instances",
    "summarize_models",
    "deploy_model",
    "deploy_custom_function",
    "deploy_vllm_function",
    "deployment_storage_gb",
    "upsert_scaling_policy",
    "delete_function",
    "delete_pod",
    "get_model_logs",
    "check_persistent_instances",
    "assign_request",
    "release_request",
    "read_application_logs",
}
_TYPES = {
    "ApplicationLogs": ApplicationLogs,
    "InferenceRoute": InferenceRoute,
    "InstanceHealth": InstanceHealth,
    "ModelStatusSummary": ModelStatusSummary,
    "ProviderInstance": ProviderInstance,
}


def load_env_file() -> None:
    """Load ``.env`` without overriding variables already set in the process."""
    candidates = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        for raw in resolved.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _adapter(name: str) -> Any:
    if name not in _BACKENDS:
        raise InferenceBackendError(
            "backend must be one of: " + ", ".join(_BACKENDS) + f" (got {name!r})"
        )
    cached = _ADAPTERS.get(name)
    if cached is not None:
        return cached
    if name == "modal":
        from provider_service.providers.modal.adapter import ModalAdapter

        cached = ModalAdapter()
    elif name == "runpod":
        from provider_service.providers.runpod.adapter import RunpodAdapter

        cached = RunpodAdapter()
    elif name == "beam":
        from provider_service.providers.beam.adapter import BeamAdapter

        cached = BeamAdapter()
    else:
        from provider_service.providers.inferx.adapter import InferxAdapter

        cached = InferxAdapter()
    _ADAPTERS[name] = cached
    return cached


def _special(backend: str, op: str, args: dict[str, Any]) -> Any:
    if backend == "beam" and op == "split_capacity":
        from provider_service.providers.beam.capacity import split_capacity

        return list(split_capacity(int(args.get("min_instances") or 0), int(args.get("max_instances") or 0)))
    if backend == "beam" and op == "checkpoint_enabled":
        from provider_service.settings import beam_checkpoint_enabled

        return beam_checkpoint_enabled()
    if backend == "beam" and op == "wait_for_checkpoint":
        from provider_service.providers.beam.app import wait_for_checkpoint

        wait_for_checkpoint(
            str(args.get("model_id") or ""),
            timeout_s=args.get("timeout_s"),
            poll_s=float(args.get("poll_s") or 5.0),
        )
        return None
    if backend == "beam" and op == "edge_headers":
        from provider_service.providers.beam.guest_gate import edge_headers
        from provider_service.settings import beam_api_token

        return edge_headers(str(args.get("model_id") or ""), beam_api_token())
    raise KeyError(op)


def dispatch(backend: str, op: str, args: dict[str, Any]) -> Any:
    try:
        return _special(backend, op, args)
    except KeyError:
        pass
    if op not in _OPS:
        raise InferenceBackendError(f"Unknown provider operation {op!r}")
    target = _adapter(backend)
    if op == "base_url":
        return target.base_url
    method = getattr(target, op, None)
    if not callable(method):
        raise InferenceBackendError(f"{backend} does not implement {op}")
    return method(**args)


def encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        payload = {"__type__": type(value).__name__}
        for field in fields(value):
            payload[field.name] = encode(getattr(value, field.name))
        return payload
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value


def _authorized(authorization: str | None) -> bool:
    expected = (os.getenv("PROVIDER_SERVICE_TOKEN") or "").strip()
    if expected == "":
        return False
    presented = (authorization or "").strip()
    prefix = "Bearer "
    if presented.startswith(prefix):
        presented = presented[len(prefix):].strip()
    return presented == expected


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1")
async def invoke(body: dict[str, Any], authorization: str | None = Header(default=None)) -> JSONResponse:
    if not _authorized(authorization):
        return JSONResponse({"ok": False, "error": "Unauthorized", "type": "AuthError"}, status_code=401)
    backend = str(body.get("backend") or "").strip().lower()
    op = str(body.get("op") or "").strip()
    args = body.get("args") or {}
    if not isinstance(args, dict):
        return JSONResponse(
            {"ok": False, "error": "args must be an object", "type": "InferenceBackendError"},
            status_code=400,
        )
    try:
        result = dispatch(backend, op, args)
        if inspect.iscoroutine(result):
            result = await result
    except Exception as exc:
        from provider_service.inferx_client import InferxError

        known = isinstance(exc, (InferenceBackendError, InferxError))
        return JSONResponse(
            {"ok": False, "error": str(exc) or type(exc).__name__, "type": type(exc).__name__},
            status_code=400 if known else 500,
        )
    return JSONResponse({"ok": True, "result": encode(result)})
