"""Background start, stop, and restart for the provider service."""

import subprocess

import pytest

from provider_service import __main__ as entry
from provider_service import cli


class _Proc:
    def __init__(self, pid: int = 4242, code: int | None = None) -> None:
        self.pid = pid
        self.code = code
        self.polled = False

    def poll(self) -> int | None:
        self.polled = True
        return self.code


def _silence_env(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setenv("PROVIDER_SERVICE_TOKEN", "token")


def test_start_records_the_pid_when_the_process_stays_up(tmp_path, monkeypatch, capsys):
    _silence_env(monkeypatch)
    spawned: list[tuple[list[str], dict]] = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return _Proc()

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "_wait_until_listening", lambda proc, host, port: True)
    cli.start(tmp_path)
    command, kwargs = spawned[0]
    assert command[1:] == ["-m", "provider_service", "run"]
    assert kwargs["cwd"] == tmp_path
    assert (tmp_path / cli.PID_NAME).read_text(encoding="utf-8").strip() == "4242"
    assert "started (pid 4242)" in capsys.readouterr().out


def test_start_refuses_when_the_service_is_already_running(tmp_path, monkeypatch):
    _silence_env(monkeypatch)
    (tmp_path / cli.PID_NAME).write_text("99\n", encoding="utf-8")
    monkeypatch.setattr(cli, "process_alive", lambda pid: pid == 99)
    with pytest.raises(SystemExit, match="already running"):
        cli.start(tmp_path)


def test_start_requires_a_token(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("PROVIDER_SERVICE_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="PROVIDER_SERVICE_TOKEN is required"):
        cli.start(tmp_path)


def test_start_reports_a_process_that_exits(tmp_path, monkeypatch):
    _silence_env(monkeypatch)
    stopped: list[int] = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: _Proc(code=1))
    monkeypatch.setattr(cli, "_wait_until_listening", lambda proc, host, port: False)
    monkeypatch.setattr(cli, "terminate_process", stopped.append)
    with pytest.raises(SystemExit, match="failed to start \\(1\\)"):
        cli.start(tmp_path)
    assert stopped == [4242]
    assert not (tmp_path / cli.PID_NAME).exists()


def test_stop_terminates_the_recorded_process(tmp_path, monkeypatch, capsys):
    (tmp_path / cli.PID_NAME).write_text("77\n", encoding="utf-8")
    alive = {"77": True}

    monkeypatch.setattr(cli, "process_alive", lambda pid: alive.get(str(pid), False))

    def terminate(pid: int) -> None:
        alive[str(pid)] = False

    monkeypatch.setattr(cli, "terminate_process", terminate)
    cli.stop(tmp_path)
    assert not (tmp_path / cli.PID_NAME).exists()
    assert capsys.readouterr().out.strip() == "provider service stopped"


def test_stop_clears_a_stale_pid(tmp_path, monkeypatch, capsys):
    (tmp_path / cli.PID_NAME).write_text("not-a-pid\n", encoding="utf-8")
    cli.stop(tmp_path)
    assert not (tmp_path / cli.PID_NAME).exists()
    assert "not running" in capsys.readouterr().out


def test_restart_stops_then_starts(tmp_path, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(cli, "stop", lambda directory=None: order.append(f"stop:{directory}"))
    monkeypatch.setattr(cli, "start", lambda directory=None: order.append(f"start:{directory}"))
    cli.restart(tmp_path)
    assert order == [f"stop:{tmp_path}", f"start:{tmp_path}"]


def test_commands_dispatch_to_start_stop_and_restart(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(cli, "start", lambda directory=None: seen.append("start"))
    monkeypatch.setattr(cli, "stop", lambda directory=None: seen.append("stop"))
    monkeypatch.setattr(cli, "restart", lambda directory=None: seen.append("restart"))
    monkeypatch.setattr(entry, "serve", lambda: seen.append("serve"))
    entry.main(["start"])
    entry.main(["stop"])
    entry.main(["restart"])
    entry.main([])
    entry.main(["run"])
    assert seen == ["start", "stop", "restart", "serve", "serve"]


def test_detach_uses_a_new_session_or_process_group():
    kwargs = cli._detach_kwargs()
    if subprocess.os.name == "nt":
        flags = kwargs["creationflags"]
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_NO_WINDOW
    else:
        assert kwargs["start_new_session"] is True
