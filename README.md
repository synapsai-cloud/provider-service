# Provider service

HTTP control plane for InferX, Modal, Runpod, and Beam. It is the process that loads `beam-client` and `beta9`. Those packages are AGPL-3.0, and so is this service. See `LICENSE`.

Callers send one JSON request:

```json
{"backend": "beam", "op": "deploy_model", "args": {"model_id": "m1"}}
```

`backend` is `inferx`, `modal`, `runpod`, or `beam`. The service keeps one adapter for each backend in this process, including Beam's in-flight route book.

## Run

```bash
pip install -e .
cp .env.example .env
# set PROVIDER_SERVICE_TOKEN and the provider credentials you use
python -m provider_service
```

The listen address is `PROVIDER_SERVICE_HOST` and `PROVIDER_SERVICE_PORT` (default `127.0.0.1:8091`). Every `/v1` call must send `Authorization: Bearer <PROVIDER_SERVICE_TOKEN>`. `GET /health` does not.

Callers authenticate with the same `PROVIDER_SERVICE_TOKEN`. Provider credentials stay in this service's `.env`.

## Source

This service is [AGPL-3.0-only](LICENSE). `GET /health` returns a `source` URL so anyone reaching the process over the network can obtain the corresponding source (GNU AGPL section 13).

https://github.com/synapsai-cloud/provider-service

A modified deployment sets `PROVIDER_SERVICE_SOURCE_URL` to the network location of that modified source. `/health` then offers that URL instead.
