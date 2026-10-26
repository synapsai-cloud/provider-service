"""The custom-image gate must not reveal the Beam workspace token."""

import socket
import threading

from provider_service.providers.beam.guest_gate import (
    EDGE_HEADER,
    _serve,
    edge_token,
    gate_program,
    untrusted_entrypoint,
)


def test_edge_token_is_not_the_workspace_token():
    secret = edge_token("model-a", "workspace-token")
    assert secret
    assert secret != "workspace-token"
    assert edge_token("model-a", "workspace-token") == secret
    assert edge_token("model-b", "workspace-token") != secret
    assert edge_token("", "workspace-token") == ""


def test_untrusted_entrypoint_drops_workspace_token_before_the_command():
    command = untrusted_entrypoint(["python", "server.py", "--port", "8000"])
    script = command[2]
    assert command[0] == "sh"
    assert "unset BETA9_TOKEN BEAM_TOKEN" in script
    assert script.index("unset BETA9_TOKEN") < script.index('"$@"')
    assert "workspace-token" not in script
    assert command[4:] == ["python", "server.py", "--port", "8000"]


def test_gate_program_is_standalone():
    program = gate_program()
    assert "import modules" not in program
    compile(program, "<gate>", "exec")


def test_gate_strips_workspace_token_and_requires_edge_header(monkeypatch):
    monkeypatch.setenv("SYNAPS_EDGE_TOKEN", "edge-secret")
    monkeypatch.setenv("SYNAPS_LISTEN_PORT", "0")
    monkeypatch.setenv("SYNAPS_APP_PORT", "0")

    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(1)
    app_port = upstream.getsockname()[1]
    monkeypatch.setenv("SYNAPS_APP_PORT", str(app_port))

    gate = socket.socket()
    gate.bind(("127.0.0.1", 0))
    listen_port = gate.getsockname()[1]
    gate.close()
    monkeypatch.setenv("SYNAPS_LISTEN_PORT", str(listen_port))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    def request(headers: dict[str, str]) -> bytes:
        client = socket.create_connection(("127.0.0.1", listen_port), timeout=2)
        lines = ["POST /v1/chat?auth_token=workspace-token&x=1 HTTP/1.1", "Host: gate"]
        lines.extend(f"{key}: {value}" for key, value in headers.items())
        client.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        data = b""
        client.settimeout(2)
        while b"\r\n\r\n" not in data:
            data += client.recv(4096)
        client.close()
        return data

    denied = request({"Authorization": "Bearer workspace-token"})
    assert denied.startswith(b"HTTP/1.1 401")

    seen = {}

    def accept_once():
        conn, _addr = upstream.accept()
        data = b""
        conn.settimeout(2)
        while b"\r\n\r\n" not in data:
            data += conn.recv(4096)
        seen["request"] = data
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        conn.close()

    accepter = threading.Thread(target=accept_once, daemon=True)
    accepter.start()
    allowed = request({EDGE_HEADER: "edge-secret", "Authorization": "Bearer workspace-token"})
    accepter.join(timeout=2)
    upstream.close()

    assert allowed.startswith(b"HTTP/1.1 200")
    forwarded = seen["request"].decode()
    assert "Authorization" not in forwarded
    assert "auth_token" not in forwarded
    assert "x=1" in forwarded
    assert EDGE_HEADER not in forwarded
