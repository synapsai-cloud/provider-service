"""Run the provider service."""

from __future__ import annotations

import uvicorn

from provider_service.cli import dispatch, listen_address, require_token


def serve() -> None:
    require_token()
    host, port = listen_address()
    uvicorn.run("provider_service.server:app", host=host, port=port, factory=False)


def main(argv: list[str] | None = None) -> None:
    dispatch(argv, serve)


if __name__ == "__main__":
    main()
