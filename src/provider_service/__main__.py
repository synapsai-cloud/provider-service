"""Run the provider service."""

from __future__ import annotations

import os

import uvicorn

from provider_service.server import load_env_file


def main() -> None:
    load_env_file()
    if (os.getenv("PROVIDER_SERVICE_TOKEN") or "").strip() == "":
        raise SystemExit("PROVIDER_SERVICE_TOKEN is required")
    host = (os.getenv("PROVIDER_SERVICE_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("PROVIDER_SERVICE_PORT") or "8091").strip())
    uvicorn.run("provider_service.server:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
