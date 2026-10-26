"""Keep the Beam workspace token out of a customer's process.

The pod stays authorized, so Beam still requires ``Authorization``. This process
is the entrypoint: it removes ``BETA9_TOKEN`` before the customer's command,
then accepts traffic and drops ``Authorization`` before forwarding. Callers
also present ``X-Synaps-Edge``, an HMAC of the model id.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect

EDGE_HEADER = "X-Synaps-Edge"
LISTEN_PORT = 8000
APP_PORT = 8001
_DROPPED_ENV = (
    "BETA9_TOKEN",
    "BEAM_TOKEN",
    "SYNAPS_EDGE_TOKEN",
    "SYNAPS_EDGE_SCRIPT",
    "SYNAPS_LISTEN_PORT",
)


def edge_token(model_id: str, workspace_token: str) -> str:
    """Header secret for one model. It cannot be turned back into the workspace token."""
    model = str(model_id or "").strip()
    secret = str(workspace_token or "")
    if model == "" or secret == "":
        return ""
    return hmac.new(secret.encode(), model.encode(), hashlib.sha256).hexdigest()


def edge_headers(model_id: str, workspace_token: str) -> dict[str, str]:
    token = edge_token(model_id, workspace_token)
    if token == "":
        return {}
    return {EDGE_HEADER: token}


def untrusted_entrypoint(command: list[str]) -> list[str]:
    """Shell entrypoint that drops the workspace token before the customer command.

    The customer process is started only after those variables are unset, so it
    cannot read them from ``/proc``. The gate process starts afterward and is
    the one that still holds the edge secret.
    """
    if not command:
        raise ValueError("command is required")
    script = r"""
if command -v python3 >/dev/null 2>&1; then py=python3
elif command -v python >/dev/null 2>&1; then py=python
else
  echo "This image needs python so the Beam workspace token can be removed before your process starts." >&2
  exit 1
fi
edge=$SYNAPS_EDGE_TOKEN
program=$SYNAPS_EDGE_SCRIPT
listen=${SYNAPS_LISTEN_PORT:-8000}
app=${SYNAPS_APP_PORT:-8001}
unset BETA9_TOKEN BEAM_TOKEN SYNAPS_EDGE_TOKEN SYNAPS_EDGE_SCRIPT SYNAPS_LISTEN_PORT
export PORT="$app"
"$@" &
child=$!
export SYNAPS_EDGE_TOKEN="$edge"
export SYNAPS_LISTEN_PORT="$listen"
export SYNAPS_APP_PORT="$app"
unset edge listen app
"$py" -c "$program" &
gate=$!
wait "$child"
status=$?
kill "$gate" 2>/dev/null || true
exit "$status"
"""
    return ["sh", "-c", script, "sh", *command]


def gate_program() -> str:
    """Stdlib program executed inside the container. It does not import this package."""
    return inspect.getsource(_serve) + "\n_serve()\n"


def _serve() -> None:
    import os
    import socket
    import threading
    from urllib.parse import parse_qsl, urlencode

    edge = os.environ.get("SYNAPS_EDGE_TOKEN") or ""
    listen_port = int(os.environ.get("SYNAPS_LISTEN_PORT") or "8000")
    app_port = int(os.environ.get("SYNAPS_APP_PORT") or "8001")
    drop = {b"authorization", b"proxy-authorization", b"x-synaps-edge"}

    def edge_ok(lines: list[bytes]) -> bool:
        if edge == "":
            return False
        for line in lines:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"x-synaps-edge" and value.strip().decode("latin1") == edge:
                return True
        return False

    def strip_query(target: bytes) -> bytes:
        path, question, query = target.partition(b"?")
        if question == b"":
            return target
        kept = [
            (key, value)
            for key, value in parse_qsl(query.decode("latin1"), keep_blank_values=True)
            if key != "auth_token"
        ]
        if not kept:
            return path
        return path + b"?" + urlencode(kept).encode("latin1")

    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                buf = src.recv(65536)
                if not buf:
                    break
                dst.sendall(buf)
        except Exception:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    def handle(client: socket.socket) -> None:
        upstream = None
        try:
            client.settimeout(240)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    return
                data += chunk
                if len(data) > 1024 * 1024:
                    client.sendall(b"HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\n\r\n")
                    return
            head, rest = data.split(b"\r\n\r\n", 1)
            lines = head.split(b"\r\n")
            if len(lines) < 1 or not edge_ok(lines[1:]):
                client.sendall(b"HTTP/1.1 401 Unauthorized\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                return
            start = lines[0].split(b" ")
            if len(start) >= 2:
                start[1] = strip_query(start[1])
                lines[0] = b" ".join(start)
            forwarded = [lines[0]]
            for line in lines[1:]:
                name = line.split(b":", 1)[0].strip().lower()
                if name in drop:
                    continue
                forwarded.append(line)
            upstream = socket.create_connection(("127.0.0.1", app_port), timeout=10)
            upstream.sendall(b"\r\n".join(forwarded) + b"\r\n\r\n" + rest)
            back = threading.Thread(target=pump, args=(upstream, client), daemon=True)
            back.start()
            pump(client, upstream)
            back.join(timeout=5)
        except Exception:
            try:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            except Exception:
                pass
        finally:
            if upstream is not None:
                upstream.close()
            client.close()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", listen_port))
    server.listen(128)
    while True:
        client, _addr = server.accept()
        threading.Thread(target=handle, args=(client,), daemon=True).start()

