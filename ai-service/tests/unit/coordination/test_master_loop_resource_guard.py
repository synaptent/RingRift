from __future__ import annotations

from scripts.master_loop import MasterLoopController


def test_master_loop_self_guard_requests_restart_on_rss(monkeypatch) -> None:
    monkeypatch.setenv("RINGRIFT_MASTER_LOOP_RSS_BUDGET_GB", "1")
    monkeypatch.setenv("RINGRIFT_MASTER_LOOP_MAX_UPTIME_HOURS", "0")
    controller = MasterLoopController(
        configs=["hex8_2p"],
        dry_run=True,
        skip_daemons=True,
        daemon_profile="lean",
    )
    controller._running = True
    monkeypatch.setattr(controller, "_get_current_rss_gb", lambda: 2.5)

    controller._master_loop_self_guard_check()

    assert not controller._running
    assert controller._shutdown_event.is_set()
    assert controller._restart_requested_reason == "rss_budget_exceeded:2.50GB>1.00GB"


def test_master_loop_status_includes_process_metrics(monkeypatch) -> None:
    controller = MasterLoopController(
        configs=["hex8_2p"],
        dry_run=True,
        skip_daemons=True,
        daemon_profile="lean",
    )
    monkeypatch.setattr(controller, "_get_current_rss_gb", lambda: 0.25)

    status = controller.get_status()

    assert status["process"]["rss_gb"] == 0.25
    assert status["process"]["rss_budget_gb"] > 0
