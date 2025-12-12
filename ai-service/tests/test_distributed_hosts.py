import os
import subprocess

import app.distributed.hosts as hosts


def test_host_config_ssh_target() -> None:
    cfg = hosts.HostConfig(name="h", ssh_host="example.com", ssh_user="alice")
    assert cfg.ssh_target == "alice@example.com"

    cfg2 = hosts.HostConfig(name="h", ssh_host="bob@example.com")
    assert cfg2.ssh_target == "bob@example.com"

    cfg3 = hosts.HostConfig(name="h", ssh_host="example.com")
    assert cfg3.ssh_target == "example.com"


def test_host_config_http_worker_url_strips_user() -> None:
    cfg = hosts.HostConfig(name="h", ssh_host="bob@example.com", worker_port=8123)
    assert cfg.http_worker_url == "http://example.com:8123"


def test_load_remote_hosts_splits_user_at_host(tmp_path) -> None:
    config_path = tmp_path / "hosts.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hosts:",
                "  test:",
                "    ssh_host: 'alice@1.2.3.4'",
                "    ssh_port: 2222",
                "    ringrift_path: '~/Development/RingRift'",
                "    venv_activate: 'source ~/Development/RingRift/ai-service/venv/bin/activate'",
            ]
        )
        + "\n"
    )

    hosts._HOST_CONFIG_CACHE.clear()
    loaded = hosts.load_remote_hosts(str(config_path))
    assert "test" in loaded

    cfg = loaded["test"]
    assert cfg.ssh_user == "alice"
    assert cfg.ssh_host == "1.2.3.4"
    assert cfg.ssh_target == "alice@1.2.3.4"
    assert cfg.ssh_port == 2222
    assert cfg.work_directory.endswith("/ai-service")
    assert cfg.http_worker_url == "http://1.2.3.4:8765"
    hosts._HOST_CONFIG_CACHE.clear()


def test_ssh_executor_builds_port_key_and_venv(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "cmd": cmd,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(hosts.subprocess, "run", fake_run)

    cfg = hosts.HostConfig(
        name="h",
        ssh_host="example.com",
        ssh_user="alice",
        ssh_port=2222,
        ssh_key=str(tmp_path / "id_rsa"),
        work_dir="~/Development/RingRift",
        venv_activate="source ~/venv/bin/activate",
    )

    executor = hosts.SSHExecutor(cfg)
    result = executor.run("python -V", timeout=12, capture_output=True)
    assert result.returncode == 0
    assert len(calls) == 1

    cmd = calls[0]["cmd"]
    assert cmd[0] == "ssh"
    assert "-p" in cmd
    assert cmd[cmd.index("-p") + 1] == "2222"
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == os.path.expanduser(cfg.ssh_key_path)
    assert cmd[-2] == "alice@example.com"
    assert cmd[-1].startswith("cd ")
    assert " && . ~/venv/bin/activate && python -V" in cmd[-1]


def test_ssh_executor_scp_from_uses_port_and_key(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "cmd": cmd,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hosts.subprocess, "run", fake_run)

    cfg = hosts.HostConfig(
        name="h",
        ssh_host="example.com",
        ssh_user="alice",
        ssh_port=2222,
        ssh_key=str(tmp_path / "id_rsa"),
    )
    executor = hosts.SSHExecutor(cfg)
    executor.scp_from("/remote/file.txt", str(tmp_path / "file.txt"), timeout=7)

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == "scp"
    assert "-P" in cmd
    assert cmd[cmd.index("-P") + 1] == "2222"
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == os.path.expanduser(cfg.ssh_key_path)
    assert cmd[-2] == "alice@example.com:/remote/file.txt"
