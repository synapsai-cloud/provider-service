"""Start, stop, and restart the provider service process."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from provider_service.server import load_env_file

PID_NAME = ".provider-service.pid"
LOG_NAME = ".provider-service.log"
_STARTUP_TIMEOUT = 60.0
_STOP_TIMEOUT = 10.0


def require_token() -> None:
    load_env_file()
    if (os.getenv("PROVIDER_SERVICE_TOKEN") or "").strip() == "":
        raise SystemExit("PROVIDER_SERVICE_TOKEN is required")


def listen_address() -> tuple[str, int]:
    host = (os.getenv("PROVIDER_SERVICE_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("PROVIDER_SERVICE_PORT") or "8091").strip())
    return host, port


def start(directory: Path | None = None) -> None:
    """Run the service in the background."""
    directory = _directory(directory)
    require_token()
    existing = _read_pid(directory)
    if existing is not None and process_alive(existing):
        raise SystemExit(f"provider service is already running (pid {existing})")
    _clear_pid(directory)
    host, port = listen_address()
    log_path = directory / LOG_NAME
    command = [sys.executable, "-m", "provider_service", "run"]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=directory,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **_detach_kwargs(),
        )
    _write_pid(directory, proc.pid)
    if not _wait_until_listening(proc, host, port):
        code = proc.poll()
        terminate_process(proc.pid)
        _clear_pid(directory)
        detail = f" ({code})" if code is not None else ""
        raise SystemExit(f"provider service failed to start{detail}; see {LOG_NAME}")
    print(f"provider service started (pid {proc.pid})")


def stop(directory: Path | None = None) -> None:
    """Stop the background service, if it is running."""
    directory = _directory(directory)
    pid = _read_pid(directory)
    if pid is None or not process_alive(pid):
        _clear_pid(directory)
        print("provider service is not running")
        return
    terminate_process(pid)
    if not _wait_until_stopped(pid):
        if os.name != "nt":
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if not _wait_until_stopped(pid, timeout=2):
            raise SystemExit(f"provider service did not stop (pid {pid})")
    _clear_pid(directory)
    print("provider service stopped")


def restart(directory: Path | None = None) -> None:
    stop(directory)
    start(directory)


def dispatch(argv: list[str] | None, serve: Callable[[], None]) -> None:
    parser = argparse.ArgumentParser(prog="provider-service")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("start", help="Start the service in the background.")
    commands.add_parser("stop", help="Stop the background service.")
    commands.add_parser("restart", help="Stop the service, then start it again.")
    commands.add_parser("run", help="Run the service in the foreground.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "start":
        start()
    elif args.command == "stop":
        stop()
    elif args.command == "restart":
        restart()
    else:
        serve()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def _directory(directory: Path | None) -> Path:
    return (directory or Path.cwd()).resolve()


def _pid_path(directory: Path) -> Path:
    return directory / PID_NAME


def _read_pid(directory: Path) -> int | None:
    path = _pid_path(directory)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        return None
    return int(text)


def _write_pid(directory: Path, pid: int) -> None:
    _pid_path(directory).write_text(f"{pid}\n", encoding="utf-8")


def _clear_pid(directory: Path) -> None:
    _pid_path(directory).unlink(missing_ok=True)


def _detach_kwargs() -> dict[str, int | bool]:
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


def _probe_host(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _wait_until_listening(proc: subprocess.Popen[bytes], host: str, port: int) -> bool:
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    url = f"http://{_probe_host(host)}:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    return False


def _wait_until_stopped(pid: int, timeout: float = _STOP_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not process_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _windows_process_alive(pid: int) -> bool:
    import ctypes

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(ok) and code.value == 259
