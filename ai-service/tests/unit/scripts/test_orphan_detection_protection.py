"""Tests for OrphanProcessDetectionLoop systemd-unit protection.

The orphan detector used to SIGKILL any process whose cmdline contained
``selfplay`` after 15 minutes, which struck the standalone
``ringrift-training.service`` running ``minimal_alphazero_loop.py``
because its cmdline has ``--selfplay-budget`` / ``--selfplay-randomness``.
These tests lock in the fix:

1. Default patterns are script-path-anchored (no bare ``selfplay``).
2. Protected-unit prefixes cover every real ringrift systemd unit.
3. ``_process_systemd_unit`` parses /proc/<pid>/cgroup reliably.
4. ``_is_protected_systemd_process`` blocks kills against protected units
   and lets everything else through.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from scripts.p2p.loops.job_loops import (
    OrphanProcessDetectionConfig,
    OrphanProcessDetectionLoop,
)


class TestPatternDefaults:
    def test_no_bare_word_patterns(self) -> None:
        """Bare words like 'selfplay' match any --selfplay-* argument — too
        permissive.  Every default pattern must be a real script path.
        """
        config = OrphanProcessDetectionConfig()
        for pattern in config.orphan_patterns:
            assert pattern.startswith("scripts/"), pattern
            assert pattern.endswith(".py"), pattern

    def test_no_pattern_matches_minimal_alphazero_loop_cmdline(self) -> None:
        """The production cmdline that used to trip the detector must no
        longer match any default pattern."""
        cmdline = (
            "/home/ubuntu/venv/bin/python scripts/minimal_alphazero_loop.py "
            "--model models/canonical_hex8_2p_v5_heavy.pth "
            "--work-dir data/minimal_loop_hex8_2p_v5_heavy "
            "--board-type hex8 --num-players 2 "
            "--selfplay-budget 200 --eval-budget 128 "
            "--selfplay-randomness 0.25 "
            "--supplemental-data-dir data/minimal_loop_hex8_2p/supplemental"
        )
        config = OrphanProcessDetectionConfig()
        for pattern in config.orphan_patterns:
            assert pattern not in cmdline, (
                f"pattern {pattern!r} still matches minimal_alphazero_loop cmdline"
            )


class TestProtectedUnitDefaults:
    def test_covers_training_and_p2p_services(self) -> None:
        config = OrphanProcessDetectionConfig()
        protected = set(config.protected_unit_prefixes)
        # The services this fix was written to protect
        assert "ringrift-training" in protected
        assert "ringrift-p2p" in protected
        # selfplay-worker + web services also shouldn't be targeted
        assert "ringrift-selfplay-worker" in protected
        assert "ringrift-ai" in protected
        assert "ringrift-server" in protected


class TestProcessSystemdUnit:
    def test_parses_system_slice_service(self) -> None:
        # cgroup v2 single-line format
        cgroup = "0::/system.slice/ringrift-training.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(12345)
        assert unit == "ringrift-training.service"

    def test_parses_system_slice_scope(self) -> None:
        cgroup = "0::/system.slice/session-c1.scope\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(12345)
        assert unit == "session-c1.scope"

    def test_parses_user_slice_service(self) -> None:
        cgroup = "0::/user.slice/user-1000.slice/session-5.scope/user-script.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(12345)
        assert unit == "user-script.service"

    def test_returns_none_for_nonexistent_pid(self) -> None:
        def _raise(*_args, **_kwargs):
            raise FileNotFoundError
        with patch("builtins.open", _raise):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(99999)
        assert unit is None

    def test_returns_none_for_non_systemd_process(self) -> None:
        # Process in no recognizable slice
        cgroup = "0::/\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(12345)
        assert unit is None

    def test_handles_multiline_cgroup_v1(self) -> None:
        # Legacy cgroup v1 has multiple lines per PID
        cgroup = (
            "11:memory:/system.slice/ringrift-p2p.service\n"
            "9:cpu,cpuacct:/system.slice/ringrift-p2p.service\n"
            "0::/system.slice/ringrift-p2p.service\n"
        )
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = OrphanProcessDetectionLoop._process_systemd_unit(12345)
        assert unit == "ringrift-p2p.service"


class TestIsProtectedSystemdProcess:
    def _make_loop(self) -> OrphanProcessDetectionLoop:
        return OrphanProcessDetectionLoop(
            get_tracked_pids=lambda: set(),
            config=OrphanProcessDetectionConfig(),
        )

    def test_training_service_is_protected(self) -> None:
        loop = self._make_loop()
        cgroup = "0::/system.slice/ringrift-training.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit == "ringrift-training.service"

    def test_selfplay_worker_is_protected(self) -> None:
        loop = self._make_loop()
        cgroup = "0::/system.slice/ringrift-selfplay-worker.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit == "ringrift-selfplay-worker.service"

    def test_p2p_service_is_protected(self) -> None:
        loop = self._make_loop()
        cgroup = "0::/system.slice/ringrift-p2p.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit == "ringrift-p2p.service"

    def test_unrelated_service_not_protected(self) -> None:
        """A service that happens to match a kill pattern but is not one
        of ours must still be killable — we only protect our own units.
        """
        loop = self._make_loop()
        cgroup = "0::/system.slice/some-random-app.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit is None

    def test_untracked_background_process_not_protected(self) -> None:
        """A process spawned outside any systemd unit (e.g. a detached
        python job from an ssh session) stays killable — that's the kind
        of legitimate P2P orphan this loop exists to clean up.
        """
        loop = self._make_loop()
        cgroup = "0::/\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit is None

    def test_prefix_match_handles_versioned_unit_names(self) -> None:
        """Protection should catch a future rename like
        ``ringrift-training@hex8_2p.service`` without a code change.
        """
        loop = self._make_loop()
        cgroup = "0::/system.slice/ringrift-training@hex8_2p.service\n"
        with patch("builtins.open", mock_open(read_data=cgroup)):
            unit = loop._is_protected_systemd_process(12345)
        assert unit is not None
        assert unit.startswith("ringrift-training")
