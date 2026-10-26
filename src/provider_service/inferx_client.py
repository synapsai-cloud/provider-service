"""HTTP helpers for the hosted InferX backend (https://model.inferx.net).

Two auth modes on the same host:

- API key (`Authorization: Bearer ix_...`) for inference routes
  (`/endpoints/v1/models`, `/funccall/...`, `/v1/chat/completions`).
- Flask `session` cookie for dashboard routes
  (`/listfunc`, `/listpod`, `/func_save`, `/proxy/pod/...` DELETE, `/proxy/podlog/...`,
  `/apikeys`, `/generate_apikeys`, `/usage/tokens/{tenant}`).

Gateway management paths such as `/functions/...` are not public on this host.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any
from urllib.parse import quote

import httpx

from provider_service.url import build_url

DEFAULT_BASE_URL = "https://model.inferx.net"
DEFAULT_FUNCTION_IMAGE = "vllm/vllm-openai:v0.19.1"
DEFAULT_GPU_COUNT = 1
DEFAULT_VRAM_MB = 24000
# Stock vLLM needs enough KV cache for one request at max_model_len. Qwen3-class
# models advertise 256k context, which does not fit a 24 GB GPU, so InferX
# always passes an explicit cap (same default as SynapsAI max_total_tokens).
DEFAULT_MAX_MODEL_LEN = 64000
# InferX FuncPolicySpec requires this field (no serde default). Dashboard default is 1.
DEFAULT_STANDBY_PER_NODE = 1
FUNC_DATA_ID_RE = re.compile(r'data-id="([^"/]+)/([^"/]+)/([^"]+)"')
POD_GROUP_RE = re.compile(
    r'<tbody\s+class="pod-sort-group"(?P<attrs>[^>]*)>(?P<body>.*?)</tbody>',
    re.DOTALL | re.IGNORECASE,
)
POD_HREF_RE = re.compile(
    r'pod\?tenant=([^&"]+)(?:&amp;|&){2}namespace=([^&"]+)(?:&amp;|&){2}name=([^&"]+)'
)
POD_STATE_RE = re.compile(r'class="pod-state">\s*([^<]+?)\s*<')
POD_FUNCNAME_RE = re.compile(r'data-funcname="([^"]+)"')
POD_SPEC_ID_RE = re.compile(r'data-id="([^"]+)"')
POD_FPREVISION_RE = re.compile(r'data-fprevision="([^"]+)"')
POD_CREATE_TYPE_RE = re.compile(r'data-sort-type="([^"]*)"', re.IGNORECASE)


class InferxError(Exception):
    pass


class InferxAuthError(InferxError):
    pass


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _from_secret(key: str) -> str:
    del key
    return ""


def resolve_inferx_base_url(base_url: str | None = None) -> str:
    return _first_nonempty(
        base_url,
        os.getenv("INFERX_BASE_URL"),
        _from_secret("INFERX_BASE_URL"),
        DEFAULT_BASE_URL,
    ).rstrip("/")


def resolve_inferx_api_key(api_key: str | None = None) -> str:
    return _first_nonempty(api_key, os.getenv("INFERX_API_KEY"), _from_secret("INFERX_API_KEY"))


def resolve_inferx_session(session: str | None = None) -> str:
    return _first_nonempty(session, os.getenv("INFERX_SESSION"), _from_secret("INFERX_SESSION"))


def resolve_inferx_tenant(tenant: str | None = None) -> str:
    return _first_nonempty(tenant, os.getenv("INFERX_TENANT"), _from_secret("INFERX_TENANT"))


def normalize_commands(commands: str | list[str]) -> list[str]:
    """Accept a shell string or argv list and return dashboard `spec.commands`."""
    if isinstance(commands, str):
        tokens = shlex.split(commands)
    elif isinstance(commands, list):
        tokens = [str(item) for item in commands]
    else:
        raise InferxError("commands must be a string or a list of strings")
    if len(tokens) == 0:
        raise InferxError("commands must not be empty")
    if any(token.strip() == "" for token in tokens):
        raise InferxError("commands must not contain empty strings")
    return tokens


def build_func_save_payload(
    *,
    name: str,
    commands: str | list[str],
    tenant: str,
    namespace: str = "default",
    image: str = DEFAULT_FUNCTION_IMAGE,
    hf_model: str = "",
    gpu_count: int = DEFAULT_GPU_COUNT,
    vram_mb: int = DEFAULT_VRAM_MB,
    editor_mode: str = "advanced",
    mode: str = "create",
    envs: list[list[str]] | None = None,
    sample_query: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    catalog_id: int | None = None,
) -> dict[str, Any]:
    """Build the JSON body the dashboard `/func_save` route expects.

    `editor_mode="advanced"` sends `commands` as the full custom command.
    `editor_mode="basic"` lets the dashboard wrap extra args around `hf_model`.
    """
    normalized_name = str(name or "").strip()
    normalized_tenant = str(tenant or "").strip()
    normalized_namespace = str(namespace or "").strip()
    normalized_image = str(image or "").strip()
    normalized_mode = str(mode or "").strip()
    normalized_editor_mode = str(editor_mode or "").strip()

    if normalized_name == "":
        raise InferxError("name is required")
    if normalized_tenant == "":
        raise InferxError("tenant is required")
    if normalized_namespace == "":
        raise InferxError("namespace is required")
    if normalized_image == "":
        raise InferxError("image is required")
    if normalized_mode not in ("create", "update"):
        raise InferxError('mode must be "create" or "update"')
    if normalized_editor_mode not in ("basic", "advanced"):
        raise InferxError('editor_mode must be "basic" or "advanced"')
    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count <= 0:
        raise InferxError("gpu_count must be a positive integer")
    if not isinstance(vram_mb, int) or isinstance(vram_mb, bool) or vram_mb <= 0:
        raise InferxError("vram_mb must be a positive integer")

    payload: dict[str, Any] = {
        "mode": normalized_mode,
        "tenant": normalized_tenant,
        "namespace": normalized_namespace,
        "name": normalized_name,
        "hf_model": str(hf_model or "").strip(),
        "editor_mode": normalized_editor_mode,
        "spec": {
            "image": normalized_image,
            "commands": normalize_commands(commands),
            "resources": {"GPU": {"Count": gpu_count, "vRam": vram_mb}},
        },
    }
    if envs is not None:
        payload["spec"]["envs"] = envs
    if sample_query is not None:
        payload["spec"]["sample_query"] = sample_query
    if policy is not None:
        payload["spec"]["policy"] = policy if "Obj" in policy else {"Obj": policy}
    if catalog_id is not None:
        payload["catalog_id"] = catalog_id
    return payload


INFERX_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_vllm_max_model_len(additional_args: dict[str, Any] | None) -> int:
    """Pick a stock-vLLM `--max-model-len` that fits GPU KV cache.

    SynapsAI uses `max_seq_len` (per request) and `max_total_tokens` (global
    budget). InferX runs unmodified vLLM, which only understands max_model_len
    and refuses to start unless one request of that length fits in KV cache.
    """
    extra = additional_args if isinstance(additional_args, dict) else {}
    values: list[int] = []
    for key in ("max_seq_len", "max_model_len", "max_total_tokens"):
        parsed = _positive_int(extra.get(key))
        if parsed is not None:
            values.append(parsed)
    return min(values) if values else DEFAULT_MAX_MODEL_LEN


def inferx_function_name(model_id: str) -> str:
    """Map a SynapsAI model_id to an InferX function name."""
    name = str(model_id or "").strip()
    if name == "":
        raise InferxError("model_id is required")
    if INFERX_FUNCTION_NAME_RE.fullmatch(name):
        return name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    if sanitized == "" or not INFERX_FUNCTION_NAME_RE.fullmatch(sanitized):
        raise InferxError(f"model_id {model_id!r} cannot be used as an InferX function name")
    return sanitized


def inferx_funccall_url(
    *,
    base_url: str | None = None,
    tenant: str | None = None,
    function_name: str,
    namespace: str = "default",
    openai_v1: bool = False,
) -> str:
    """Public InferX inference URL: `/funccall/{tenant}/{namespace}/{function}`.

    OpenAI-compatible clients (R2R/LiteLLM) need `openai_v1=True` so the
    base is `.../r2r-qwen3-embedding-8b/v1`.
    """
    resolved_tenant = resolve_inferx_tenant(tenant)
    if resolved_tenant == "":
        raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
    resolved_namespace = str(namespace or "").strip() or "default"
    name = inferx_function_name(function_name)
    path = f"/funccall/{resolved_tenant}/{resolved_namespace}/{name}"
    if openai_v1:
        path = f"{path}/v1"
    return build_url(resolve_inferx_base_url(base_url), path)

def build_vllm_serve_commands(
    hf_model: str,
    *,
    model_id: str,
    revision: str = "main",
    gpu_count: int = DEFAULT_GPU_COUNT,
    additional_args: dict[str, Any] | None = None,
) -> list[str]:
    """Build extra argv for InferX's stock `vllm/vllm-openai` image.

    That image already runs `vllm serve`, so commands must be flags only
    (`--model ...`), not another `vllm serve MODEL` prefix.
    """
    model = str(hf_model or "").strip()
    if model == "":
        raise InferxError("hf_model is required")
    extra = additional_args if isinstance(additional_args, dict) else {}
    tensor_parallel = extra.get("tensor_parallel_size") or extra.get("tp") or gpu_count
    runner = str(extra.get("runner") or "").strip()
    is_pooling = runner.lower() == "pooling"
    # InferX's CUDA syscall shim panics on FlashInfer/allocator spikes. Pooling
    # models leave headroom for snapshot and use Triton attention (vLLM V1 no
    # longer registers TORCH_SDPA as a decoder backend).
    default_gpu_util = 0.8 if is_pooling else 0.95
    gpu_memory_utilization = extra.get("gpu_memory_utilization", default_gpu_util)
    commands = [
        "--model", model,
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        f"--tensor-parallel-size={int(tensor_parallel)}",
    ]
    if revision and revision != "main":
        commands.extend(["--revision", str(revision)])
    if runner:
        commands.extend(["--runner", runner])
    commands.extend(["--max-model-len", str(resolve_vllm_max_model_len(extra))])
    if not extra.get("use_cuda_graphs", False):
        commands.append("--enforce-eager")
    # FlashInfer autotune runs extra dummy forwards during warmup. On 24 GB
    # InferX nodes it can trip NVML/CUDA allocator assertions and segfault
    # (same default as the in-process vLLM engine: autotune off).
    if not extra.get("enable_flashinfer_autotune", False):
        commands.append("--no-enable-flashinfer-autotune")
    attention_backend = extra.get("attention_backend")
    if attention_backend is None and is_pooling:
        attention_backend = "TRITON_ATTN"
    if attention_backend:
        commands.extend(["--attention-backend", str(attention_backend)])
    if not is_pooling:
        from provider_service.tool_flags import vllm_auto_tool_choice_flags

        commands.extend(vllm_auto_tool_choice_flags(model))
    hf_overrides = extra.get("hf_overrides")
    if hf_overrides:
        if not isinstance(hf_overrides, str):
            hf_overrides = json.dumps(hf_overrides)
        commands.extend(["--hf-overrides", hf_overrides])
    return commands


def build_inferx_embeddings_sample_query(
    hf_model: str,
    *,
    dimensions: int | None = None,
) -> dict[str, Any]:
    """Playground request for a vLLM pooling / embeddings function.

    InferX always stores `sample_query`. `apiType` must be one of their
    serde variants (`unknown`, `text2text`, ...). Do not use `text2text`
    or `knowledgebase`: the dashboard then rewrites `path` to
    `v1/chat/completions`. `unknown` keeps `path` as `v1/embeddings`.
    """
    body: dict[str, Any] = {
        "model": str(hf_model or "").strip(),
        "input": ["ping"],
        "encoding_format": "float",
    }
    if dimensions is not None:
        body["dimensions"] = int(dimensions)
    return {
        "apiType": "unknown",
        "prompt": "ping",
        "path": "v1/embeddings",
        "body": body,
    }


def _scale_wait_ratio(scale_up_threshold: float | None) -> float | None:
    """Map a deployment scale-up threshold to InferX `wait_ratio`.

    The dashboard stores a percentage (`50` means 50%). Values already in
    `[0, 1]` are ratios and are passed through.
    """
    if scale_up_threshold is None:
        return None
    try:
        parsed = float(scale_up_threshold)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return 0.0
    if parsed > 1:
        parsed = parsed / 100.0
    return parsed


def inferx_scaling_policy(
    min_instances: int = 0,
    max_instances: int = 1,
    *,
    scale_up_threshold: float | None = None,
    worker_timeout: float | None = None,
) -> dict[str, Any]:
    """Build InferX `spec.policy.Obj` from a model deployment config.

    `min_concurrent_workers` / `max_concurrent_workers` become replica bounds.
    A minimum of at least 1 keeps a pod running. `scale_up_threshold` becomes
    `scaleout_policy.WaitQueueRatio.wait_ratio`. `worker_timeout` is how many
    seconds an idle replica stays up (`scalein_timeout`).
    """
    min_replica = max(0, int(min_instances))
    max_replica = max(1, int(max_instances), min_replica)
    policy: dict[str, Any] = {
        "min_replica": min_replica,
        "max_replica": max_replica,
        "standby_per_node": DEFAULT_STANDBY_PER_NODE,
    }
    wait_ratio = _scale_wait_ratio(scale_up_threshold)
    if wait_ratio is not None:
        policy["scaleout_policy"] = {"WaitQueueRatio": {"wait_ratio": wait_ratio}}
    if worker_timeout is not None:
        try:
            timeout = float(worker_timeout)
        except (TypeError, ValueError):
            timeout = 0.0
        policy["scalein_timeout"] = max(0.0, timeout)
    return policy


def inferx_replica_policy(min_instances: int = 0, max_instances: int = 1) -> dict[str, Any]:
    """Replica bounds for InferX `spec.policy.Obj` (dashboard snake_case keys)."""
    return inferx_scaling_policy(min_instances, max_instances)


def build_funcpolicy_object(
    *,
    name: str,
    tenant: str,
    namespace: str = "default",
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Standalone `funcpolicy` object. Same-name policies override `spec.policy` live."""
    normalized_name = str(name or "").strip()
    normalized_tenant = str(tenant or "").strip()
    normalized_namespace = str(namespace or "").strip() or "default"
    if normalized_name == "":
        raise InferxError("name is required")
    if normalized_tenant == "":
        raise InferxError("tenant is required")
    if not isinstance(policy, dict):
        raise InferxError("policy must be an object")
    object_spec = dict(policy)
    if "standby_per_node" not in object_spec:
        object_spec["standby_per_node"] = DEFAULT_STANDBY_PER_NODE
    return {
        "type": "funcpolicy",
        "tenant": normalized_tenant,
        "namespace": normalized_namespace,
        "name": normalized_name,
        "object": object_spec,
    }


def funcpolicy_is_missing(response: httpx.Response) -> bool:
    """True when an update failed because the funcpolicy object does not exist yet."""
    if response.status_code not in (400, 404):
        return False
    text = (response.text or "").lower().replace(" ", "")
    return "notexist" in text or "doesnotexist" in text


def build_apikey_create_payload(
    *,
    keyname: str,
    restrict_tenant: str,
    access_level: str = "inference",
    restrict_namespace: str | None = None,
    expires_in_days: int | None = None,
) -> dict[str, Any]:
    """Build the JSON body the dashboard `PUT /apikeys` route expects."""
    normalized_keyname = str(keyname or "").strip()
    normalized_tenant = str(restrict_tenant or "").strip()
    normalized_access_level = str(access_level or "").strip()
    normalized_namespace = str(restrict_namespace or "").strip()

    if normalized_keyname == "":
        raise InferxError("keyname is required")
    if normalized_tenant == "":
        raise InferxError("restrict_tenant is required")
    if normalized_access_level not in ("inference", "read", "full"):
        raise InferxError('access_level must be "inference", "read", or "full"')
    if expires_in_days is not None and (
        not isinstance(expires_in_days, int)
        or isinstance(expires_in_days, bool)
        or expires_in_days <= 0
    ):
        raise InferxError("expires_in_days must be a positive integer")

    payload: dict[str, Any] = {
        "username": "",
        "keyname": normalized_keyname,
        "access_level": normalized_access_level,
        "restrict_tenant": normalized_tenant,
    }
    if normalized_namespace != "":
        payload["restrict_namespace"] = normalized_namespace
    if expires_in_days is not None:
        payload["expires_in_days"] = expires_in_days
    return payload


def parse_function_rows(html: str) -> list[dict[str, str]]:
    """Extract tenant/namespace/name rows from the `/listfunc` HTML page."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for tenant, namespace, name in FUNC_DATA_ID_RE.findall(html or ""):
        key = (tenant, namespace, name)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"tenant": tenant, "namespace": namespace, "name": name})
    return rows


def inferx_fprevision_from_pod_name(pod_name: str) -> str:
    """Return the function revision embedded in a pod name.

    Dashboard pod names look like ``{tenant}/{namespace}/{funcname}/{fprevision}/{id}``.
    """
    parts = [part for part in str(pod_name or "").split("/") if part != ""]
    if len(parts) < 5:
        return ""
    return parts[-2]


def parse_instance_rows(html: str) -> list[dict[str, str]]:
    """Extract instance rows from the `/listpod` HTML page."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in POD_GROUP_RE.finditer(html or ""):
        body = group.group("body")
        href = POD_HREF_RE.search(body)
        if href is None:
            continue
        tenant, namespace, name = href.group(1), href.group(2), href.group(3)
        key = (tenant, namespace, name)
        if key in seen:
            continue
        seen.add(key)
        state_match = POD_STATE_RE.search(body)
        funcname_match = POD_FUNCNAME_RE.search(body)
        spec_id_match = POD_SPEC_ID_RE.search(body)
        fprevision_match = POD_FPREVISION_RE.search(body)
        row = {"tenant": tenant, "namespace": namespace, "name": name}
        if state_match is not None:
            row["state"] = state_match.group(1).strip()
        if funcname_match is not None:
            row["funcname"] = funcname_match.group(1)
        if spec_id_match is not None:
            row["id"] = spec_id_match.group(1)
        fprevision = ""
        if fprevision_match is not None:
            fprevision = fprevision_match.group(1).strip()
        if fprevision == "":
            fprevision = inferx_fprevision_from_pod_name(name)
        if fprevision != "":
            row["fprevision"] = fprevision
        create_type_match = POD_CREATE_TYPE_RE.search(group.group("attrs") or "")
        if create_type_match is not None and create_type_match.group(1).strip() != "":
            row["create_type"] = create_type_match.group(1).strip()
        rows.append(row)
    return rows


def inferx_request(
    path: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json: Any = None,
    data: Any = None,
    headers: dict[str, str] | None = None,
    api_key: str | None = None,
    session: str | None = None,
    base_url: str | None = None,
    timeout: float = 30,
    follow_redirects: bool = False,
) -> httpx.Response:
    """Send one request to InferX.

    Pass `api_key` for inference routes or `session` for dashboard routes.
    Unspecified credentials fall back to `INFERX_API_KEY` / `INFERX_SESSION`
    / `INFERX_BASE_URL`.
    """
    url = build_url(resolve_inferx_base_url(base_url), path)
    request_headers = dict(headers or {})
    cookies: dict[str, str] = {}

    resolved_api_key = resolve_inferx_api_key(api_key)
    if resolved_api_key != "":
        request_headers.setdefault("Authorization", f"Bearer {resolved_api_key}")

    resolved_session = resolve_inferx_session(session)
    if resolved_session != "":
        cookies["session"] = resolved_session

    with httpx.Client(timeout=timeout, follow_redirects=follow_redirects) as client:
        return client.request(
            method.upper(),
            url,
            params=params,
            json=json,
            data=data,
            headers=request_headers,
            cookies=cookies,
        )


class InferxBackend:
    """Reusable InferX client with the credentials resolved once."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        session: str | None = None,
        tenant: str | None = None,
        timeout: float = 30,
    ):
        self.base_url = resolve_inferx_base_url(base_url)
        self.api_key = resolve_inferx_api_key(api_key)
        self.session = resolve_inferx_session(session)
        self.tenant = resolve_inferx_tenant(tenant)
        self.timeout = timeout

    def request(self, path: str, **kwargs) -> httpx.Response:
        kwargs.setdefault("base_url", self.base_url)
        kwargs.setdefault("api_key", self.api_key or None)
        kwargs.setdefault("session", self.session or None)
        kwargs.setdefault("timeout", self.timeout)
        return inferx_request(path, **kwargs)

    def list_published_models(self) -> list[dict[str, Any]]:
        response = self.request("/endpoints/v1/models")
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        return data if isinstance(data, list) else []

    def list_functions(self, tenant: str | None = None, namespace: str = "default") -> list[dict[str, str]]:
        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        if resolved_tenant == "":
            raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /listfunc")

        response = self.request(
            "/listfunc",
            params={"tenant": resolved_tenant, "namespace": namespace},
        )
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        response.raise_for_status()
        return parse_function_rows(response.text)

    def list_instances(
        self,
        tenant: str | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, str]]:
        """List running instances from dashboard GET `/listpod`.

        Omit tenant to list every instance the session can see. Pass tenant
        and optionally namespace to filter, matching the Instances page.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /listpod")

        resolved_tenant = resolve_inferx_tenant(self.tenant if tenant is None else tenant)
        resolved_namespace = str(namespace or "").strip()
        params: dict[str, str] = {}
        if resolved_tenant != "":
            params["tenant"] = resolved_tenant
            if resolved_namespace != "":
                params["namespace"] = resolved_namespace

        response = self.request("/listpod", params=params or None)
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        response.raise_for_status()
        return parse_instance_rows(response.text)

    def list_apikeys(self) -> list[dict[str, Any]]:
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /generate_apikeys")
        response = self.request("/generate_apikeys")
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def token_usage(self, tenant: str | None = None, hours: int = 24) -> dict[str, Any]:
        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        if resolved_tenant == "":
            raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /usage/tokens")

        response = self.request(f"/usage/tokens/{resolved_tenant}", params={"hours": hours})
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"raw": payload}

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **body: Any,
    ) -> dict[str, Any]:
        if self.api_key == "":
            raise InferxAuthError("API key is required for /v1/chat/completions")
        payload = {"model": model, "messages": messages, **body}
        response = self.request("/v1/chat/completions", method="POST", json=payload)
        response.raise_for_status()
        return response.json()

    def deploy_function(
        self,
        name: str,
        commands: str | list[str],
        *,
        tenant: str | None = None,
        namespace: str = "default",
        image: str = DEFAULT_FUNCTION_IMAGE,
        hf_model: str = "",
        gpu_count: int = DEFAULT_GPU_COUNT,
        vram_mb: int = DEFAULT_VRAM_MB,
        editor_mode: str = "advanced",
        mode: str = "create",
        envs: list[list[str]] | None = None,
        sample_query: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        catalog_id: int | None = None,
    ) -> dict[str, Any]:
        """Create or update a function via dashboard POST `/func_save`.

        Requires a logged-in `session` cookie. Custom commands are sent as
        `spec.commands` with `editor_mode="advanced"` so the dashboard keeps
        them intact. Image must be on the live InferX whitelist; GPU count
        and vRam are validated against current node capacity server-side.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /func_save")

        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        payload = build_func_save_payload(
            name=name,
            commands=commands,
            tenant=resolved_tenant,
            namespace=namespace,
            image=image,
            hf_model=hf_model,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            editor_mode=editor_mode,
            mode=mode,
            envs=envs,
            sample_query=sample_query,
            policy=policy,
            catalog_id=catalog_id,
        )
        response = self.request(
            "/func_save",
            method="POST",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=max(self.timeout, 60),
        )
        return self._dashboard_json(response, action="deploy function")

    def upsert_function_policy(
        self,
        name: str,
        policy: dict[str, Any],
        *,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Create or update a same-name funcpolicy without a new function version.

        The gateway and scheduler prefer this object over `spec.policy` and
        apply it to the running function. POST updates an existing object;
        PUT creates it when the update reports that it is missing.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required to update a function policy")

        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        payload = build_funcpolicy_object(
            name=name,
            tenant=resolved_tenant,
            namespace=namespace,
            policy=policy,
        )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = self.request("/proxy/object/", method="POST", json=payload, headers=headers)
        if funcpolicy_is_missing(response):
            response = self.request("/proxy/object/", method="PUT", json=payload, headers=headers)
        return self._dashboard_json(response, action="update function policy")

    def delete_pod(
        self,
        *,
        funcname: str,
        fprevision: str,
        pod_id: str,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Terminate one pod via dashboard DELETE `/proxy/pod/{tenant}/{namespace}/{funcname}/{fprevision}/{id}/`.

        The dashboard proxies that to the gateway and treats HTTP 202 as accepted.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required to terminate a pod")

        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        if resolved_tenant == "":
            raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
        resolved_namespace = str(namespace or "").strip() or "default"
        resolved_funcname = str(funcname or "").strip()
        resolved_revision = str(fprevision or "").strip()
        resolved_pod_id = str(pod_id or "").strip()
        if resolved_funcname == "" or resolved_revision == "" or resolved_pod_id == "":
            raise InferxError("funcname, fprevision, and pod id are required to terminate a pod")

        path = "/proxy/pod/{}/{}/{}/{}/{}/".format(
            quote(resolved_tenant, safe=""),
            quote(resolved_namespace, safe=""),
            quote(resolved_funcname, safe=""),
            quote(resolved_revision, safe=""),
            quote(resolved_pod_id, safe=""),
        )
        response = self.request(path, method="DELETE")
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        if response.status_code not in (200, 202, 204):
            raise InferxError(
                f"terminate pod failed (HTTP {response.status_code}): {response.text}"
            )
        return {"ok": True, "status_code": response.status_code}

    def read_pod_log(
        self,
        *,
        funcname: str,
        fprevision: str,
        pod_id: str,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> str:
        """Read one pod's stdout via dashboard GET `/proxy/podlog/...`.

        The dashboard proxies that path to the gateway ``/podlog`` route used
        by the instance page.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required to read a pod log")

        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        if resolved_tenant == "":
            raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
        resolved_namespace = str(namespace or "").strip() or "default"
        resolved_funcname = str(funcname or "").strip()
        resolved_revision = str(fprevision or "").strip()
        resolved_pod_id = str(pod_id or "").strip()
        if resolved_funcname == "" or resolved_revision == "" or resolved_pod_id == "":
            raise InferxError("funcname, fprevision, and pod id are required to read a pod log")

        path = "/proxy/podlog/{}/{}/{}/{}/{}/".format(
            quote(resolved_tenant, safe=""),
            quote(resolved_namespace, safe=""),
            quote(resolved_funcname, safe=""),
            quote(resolved_revision, safe=""),
            quote(resolved_pod_id, safe=""),
        )
        response = self.request(path)
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        if response.status_code >= 400:
            raise InferxError(
                f"read pod log failed (HTTP {response.status_code}): {response.text[:500]}"
            )
        text = response.text or ""
        if "<br>" in text or "&emsp;" in text:
            text = text.replace("<br>", "\n").replace("&emsp;", "    ")
        return text

    def delete_function(
        self,
        name: str,
        *,
        tenant: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Delete a function via dashboard POST `/function/delete/{tenant}/{namespace}/{name}`."""
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /function/delete")

        resolved_tenant = resolve_inferx_tenant(tenant or self.tenant)
        if resolved_tenant == "":
            raise InferxError("tenant is required (pass tenant= or set INFERX_TENANT)")
        normalized_name = str(name or "").strip()
        normalized_namespace = str(namespace or "").strip() or "default"
        if normalized_name == "":
            raise InferxError("name is required")

        response = self.request(
            f"/function/delete/{resolved_tenant}/{normalized_namespace}/{normalized_name}",
            method="POST",
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )
        if response.status_code in (200, 204, 302, 404):
            return {"ok": True, "status_code": response.status_code}
        return self._dashboard_json(response, action="delete function")

    def create_inference_apikey(
        self,
        keyname: str,
        *,
        tenant: str | None = None,
        restrict_namespace: str | None = None,
        expires_in_days: int | None = None,
    ) -> dict[str, Any]:
        """Create an inference-scoped API key via dashboard PUT `/apikeys`.

        Requires a logged-in `session` cookie. `restrict_tenant` defaults to
        this client's tenant. The raw key is returned once in `apikey`.
        """
        if self.session == "":
            raise InferxAuthError("dashboard session cookie is required for /apikeys")

        payload = build_apikey_create_payload(
            keyname=keyname,
            restrict_tenant=resolve_inferx_tenant(tenant or self.tenant),
            access_level="inference",
            restrict_namespace=restrict_namespace,
            expires_in_days=expires_in_days,
        )
        response = self.request(
            "/apikeys",
            method="PUT",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        return self._dashboard_json(response, action="create inference API key")

    def _dashboard_json(self, response: httpx.Response, *, action: str) -> dict[str, Any]:
        if response.status_code in (301, 302):
            raise InferxAuthError("InferX session expired or missing; copy a fresh session cookie")
        if response.status_code >= 400:
            message = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("error"):
                    message = str(payload["error"])
            except Exception:
                pass
            raise InferxError(f"{action} failed (HTTP {response.status_code}): {message}")
        try:
            payload = response.json()
        except Exception:
            return {"status_code": response.status_code, "text": response.text}
        return payload if isinstance(payload, dict) else {"data": payload}
