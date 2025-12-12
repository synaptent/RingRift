from __future__ import annotations

import scripts.run_ssh_distributed_tournament as ssh_tourney

from app.distributed.hosts import HostConfig


def test_parse_tiers_spec_range_and_list() -> None:
    assert ssh_tourney.parse_tiers_spec("D1-D3") == ["D1", "D2", "D3"]
    assert ssh_tourney.parse_tiers_spec("d1,d2,D2") == ["D1", "D2"]


def test_enumerate_matchups() -> None:
    matchups = ssh_tourney.enumerate_matchups(["D1", "D2", "D3"])
    assert matchups == [("D1", "D2"), ("D1", "D3"), ("D2", "D3")]


def test_assign_matchups_respects_worker_slots() -> None:
    matchups = [("D1", "D2"), ("D1", "D3"), ("D2", "D3")]
    slots = [
        ssh_tourney.WorkerSlot(host_name="h1", slot_index=0),
        ssh_tourney.WorkerSlot(host_name="h1", slot_index=1),
        ssh_tourney.WorkerSlot(host_name="h2", slot_index=0),
    ]
    assignments = ssh_tourney.assign_matchups_to_worker_slots(
        matchups,
        slots,
        cost_fn=lambda m: 1.0,
    )

    assigned = []
    for tasks in assignments.values():
        assigned.extend(tasks)
    assert sorted(assigned) == sorted(matchups)


def test_resolve_remote_path() -> None:
    host = HostConfig(name="h", ssh_host="example.com", work_dir="~/Development/RingRift")
    rel = ssh_tourney.resolve_remote_path(host, "results/x.json")
    assert rel.startswith(host.work_directory)
    assert rel.endswith("/results/x.json")

    assert ssh_tourney.resolve_remote_path(host, "/abs/x.json") == "/abs/x.json"
    assert ssh_tourney.resolve_remote_path(host, "~/x.json") == "~/x.json"


def test_build_remote_tournament_command_includes_checkpoint_and_label() -> None:
    cmd = ssh_tourney.build_remote_tournament_command(
        tier_a="D1",
        tier_b="D2",
        board="square8",
        games_per_matchup=10,
        output_dir="results/tournaments/ssh_shards/abcd",
        output_checkpoint="results/tournaments/ssh_shards/abcd/D1_vs_D2.checkpoint.json",
        output_report="results/tournaments/ssh_shards/abcd/D1_vs_D2.report.json",
        seed=1,
        think_time_scale=1.0,
        max_moves=300,
        wilson_confidence=0.95,
        worker_label="host1",
        nn_model_id=None,
    )
    assert "--output-checkpoint" in cmd
    assert "D1_vs_D2.checkpoint.json" in cmd
    assert "--worker-label" in cmd
    assert "host1" in cmd

