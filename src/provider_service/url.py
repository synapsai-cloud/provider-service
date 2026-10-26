def upstream_inference_path(endpoint: str, *, v3: bool) -> str:
    """Map a public inference path onto the process that serves it.

    Inference v3 mounts its routes under ``/api``. InferX and custom
    containers keep the path the caller already used.
    """
    if not v3 or not endpoint:
        return endpoint
    if endpoint.startswith("/v1"):
        return f"/api{endpoint}"
    if endpoint.startswith("/videos"):
        return f"/api/v1{endpoint}"
    return endpoint


def build_url(base: str, endpoint: str) -> str:
    """
    Build a URL from a base URL and an endpoint.
    
    Args:
        base: Base URL.
        endpoint: Endpoint.
    
    Returns:
        URL.
    """
    base = base.rstrip("/")
    endpoint = endpoint.lstrip("/")
    return f"{base}/{endpoint}"