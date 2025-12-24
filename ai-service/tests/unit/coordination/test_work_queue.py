"""Comprehensive tests for WorkQueue and WorkItem.

Tests cover:
- WorkItem dataclass (creation, serialization, status checks)
- WorkQueue (CRUD operations, claiming, timeouts, retries)
- Priority ordering
- Dependency handling
- Concurrent access patterns
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.work_queue import (
    SlackWorkQueueNotifier,
    WorkItem,
    WorkQueue,
    WorkStatus,
    WorkType,
)


# =============================================================================
# WorkItem Tests
# =============================================================================

class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_work_item_creation_defaults(self):
        """Test WorkItem creation with default values."""
        item = WorkItem()
        assert item.work_type == WorkType.SELFPLAY
        assert item.priority == 50
        assert item.status == WorkStatus.PENDING
        assert item.attempts == 0
        assert item.max_attempts == 3
        assert item.timeout_seconds == 3600.0
        assert item.claimed_by == ""
        assert item.depends_on == []

    def test_work_item_creation_custom(self):
        """Test WorkItem creation with custom values."""
        item = WorkItem(
            work_id="test123",
            work_type=WorkType.TRAINING,
            priority=90,
            config={"board": "hex8", "num_players": 2},
            timeout_seconds=7200.0,
        )
        assert item.work_id == "test123"
        assert item.work_type == WorkType.TRAINING
        assert item.priority == 90
        assert item.config == {"board": "hex8", "num_players": 2}
        assert item.timeout_seconds == 7200.0

    def test_work_item_to_dict(self):
        """Test WorkItem serialization to dict."""
        item = WorkItem(
            work_id="abc",
            work_type=WorkType.GPU_CMAES,
            priority=80,
            status=WorkStatus.RUNNING,
        )
        d = item.to_dict()
        assert d["work_id"] == "abc"
        assert d["work_type"] == "gpu_cmaes"
        assert d["priority"] == 80
        assert d["status"] == "running"

    def test_work_item_from_dict(self):
        """Test WorkItem deserialization from dict."""
        d = {
            "work_id": "xyz",
            "work_type": "training",
            "priority": 100,
            "status": "completed",
            "config": {"key": "value"},
        }
        item = WorkItem.from_dict(d)
        assert item.work_id == "xyz"
        assert item.work_type == WorkType.TRAINING
        assert item.priority == 100
        assert item.status == WorkStatus.COMPLETED
        assert item.config == {"key": "value"}

    def test_work_item_from_dict_string_depends_on(self):
        """Test WorkItem handles string depends_on from DB."""
        d = {
            "work_id": "test",
            "depends_on": '["dep1", "dep2"]',
        }
        item = WorkItem.from_dict(d)
        assert item.depends_on == ["dep1", "dep2"]

    def test_work_item_is_claimable_pending(self):
        """Test is_claimable for pending work."""
        item = WorkItem(status=WorkStatus.PENDING, attempts=0)
        assert item.is_claimable() is True

    def test_work_item_is_claimable_not_pending(self):
        """Test is_claimable for non-pending work."""
        for status in [WorkStatus.RUNNING, WorkStatus.COMPLETED, WorkStatus.FAILED]:
            item = WorkItem(status=status)
            assert item.is_claimable() is False

    def test_work_item_is_claimable_max_attempts(self):
        """Test is_claimable when max attempts reached."""
        item = WorkItem(status=WorkStatus.PENDING, attempts=3, max_attempts=3)
        assert item.is_claimable() is False

    def test_work_item_has_pending_dependencies_none(self):
        """Test has_pending_dependencies with no dependencies."""
        item = WorkItem(depends_on=[])
        assert item.has_pending_dependencies(set()) is False

    def test_work_item_has_pending_dependencies_all_met(self):
        """Test has_pending_dependencies when all deps are met."""
        item = WorkItem(depends_on=["dep1", "dep2"])
        completed = {"dep1", "dep2", "other"}
        assert item.has_pending_dependencies(completed) is False

    def test_work_item_has_pending_dependencies_some_unmet(self):
        """Test has_pending_dependencies when some deps are unmet."""
        item = WorkItem(depends_on=["dep1", "dep2", "dep3"])
        completed = {"dep1"}
        assert item.has_pending_dependencies(completed) is True

    def test_work_item_is_timed_out_not_claimed(self):
        """Test is_timed_out for unclaimed work."""
        item = WorkItem(status=WorkStatus.PENDING)
        assert item.is_timed_out() is False

    def test_work_item_is_timed_out_within_limit(self):
        """Test is_timed_out within timeout limit."""
        item = WorkItem(
            status=WorkStatus.RUNNING,
            claimed_at=time.time(),
            timeout_seconds=3600.0,
        )
        assert item.is_timed_out() is False

    def test_work_item_is_timed_out_exceeded(self):
        """Test is_timed_out when timeout exceeded."""
        item = WorkItem(
            status=WorkStatus.RUNNING,
            claimed_at=time.time() - 100,
            timeout_seconds=50.0,
        )
        assert item.is_timed_out() is True


# =============================================================================
# WorkQueue Tests
# =============================================================================

class TestWorkQueue:
    """Tests for WorkQueue class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def queue(self, temp_db):
        """Create a WorkQueue with temp database."""
        return WorkQueue(db_path=temp_db, policy_manager=None)

    def test_queue_initialization(self, queue):
        """Test queue initializes correctly."""
        assert queue.items == {}
        assert queue.db_path.exists()

    def test_add_work(self, queue):
        """Test adding work to queue."""
        item = WorkItem(work_type=WorkType.TRAINING, priority=90)
        work_id = queue.add_work(item)
        assert work_id == item.work_id
        assert work_id in queue.items
        assert queue.items[work_id].status == WorkStatus.PENDING

    def test_add_training_convenience(self, queue):
        """Test add_training convenience method."""
        work_id = queue.add_training("hex8", 2, priority=100)
        assert work_id in queue.items
        item = queue.items[work_id]
        assert item.work_type == WorkType.TRAINING
        assert item.priority == 100
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2

    def test_add_gpu_cmaes_convenience(self, queue):
        """Test add_gpu_cmaes convenience method."""
        work_id = queue.add_gpu_cmaes("square8", 4)
        item = queue.items[work_id]
        assert item.work_type == WorkType.GPU_CMAES

    def test_add_cpu_cmaes_convenience(self, queue):
        """Test add_cpu_cmaes convenience method."""
        work_id = queue.add_cpu_cmaes("square19", 3)
        item = queue.items[work_id]
        assert item.work_type == WorkType.CPU_CMAES

    def test_claim_work_basic(self, queue):
        """Test claiming work from queue."""
        queue.add_work(WorkItem(work_type=WorkType.SELFPLAY))
        claimed = queue.claim_work("node-1", capabilities=["selfplay"])
        assert claimed is not None
        assert claimed.status == WorkStatus.CLAIMED
        assert claimed.claimed_by == "node-1"

    def test_claim_work_empty_queue(self, queue):
        """Test claiming from empty queue."""
        claimed = queue.claim_work("node-1")
        assert claimed is None

    def test_claim_work_priority_order(self, queue):
        """Test work is claimed in priority order."""
        queue.add_work(WorkItem(work_id="low", priority=10))
        queue.add_work(WorkItem(work_id="high", priority=90))
        queue.add_work(WorkItem(work_id="mid", priority=50))

        claimed = queue.claim_work("node-1")
        assert claimed.work_id == "high"

    def test_claim_work_capability_filter(self, queue):
        """Test claim respects capability filter."""
        queue.add_work(WorkItem(work_id="train", work_type=WorkType.TRAINING))
        queue.add_work(WorkItem(work_id="play", work_type=WorkType.SELFPLAY))

        # Node with only selfplay capability
        claimed = queue.claim_work("node-1", capabilities=["selfplay"])
        assert claimed.work_id == "play"

    def test_claim_work_no_double_claim(self, queue):
        """Test same work can't be claimed twice."""
        queue.add_work(WorkItem(work_id="only"))
        queue.claim_work("node-1")
        claimed2 = queue.claim_work("node-2")
        assert claimed2 is None

    def test_start_work(self, queue):
        """Test starting claimed work."""
        queue.add_work(WorkItem(work_id="test"))
        queue.claim_work("node-1")
        success = queue.start_work("test")
        assert success is True
        assert queue.items["test"].status == WorkStatus.RUNNING
        assert queue.items["test"].started_at > 0

    def test_start_work_not_claimed(self, queue):
        """Test starting unclaimed work fails."""
        queue.add_work(WorkItem(work_id="test"))
        success = queue.start_work("test")
        assert success is False

    def test_complete_work(self, queue):
        """Test completing work."""
        queue.add_work(WorkItem(work_id="test"))
        queue.claim_work("node-1")
        queue.start_work("test")
        success = queue.complete_work("test", result={"accuracy": 0.95})
        assert success is True
        assert queue.items["test"].status == WorkStatus.COMPLETED
        assert queue.items["test"].result == {"accuracy": 0.95}

    def test_fail_work(self, queue):
        """Test failing work."""
        queue.add_work(WorkItem(work_id="test"))
        queue.claim_work("node-1")
        success = queue.fail_work("test", error="OOM error")
        assert success is True
        item = queue.items["test"]
        assert item.status == WorkStatus.PENDING  # Retryable
        assert item.attempts == 1
        assert item.error == "OOM error"

    def test_fail_work_max_attempts(self, queue):
        """Test work fails permanently after max attempts."""
        queue.add_work(WorkItem(work_id="test", max_attempts=2))

        # First attempt
        queue.claim_work("node-1")
        queue.fail_work("test")
        assert queue.items["test"].status == WorkStatus.PENDING

        # Second attempt
        queue.claim_work("node-2")
        queue.fail_work("test")
        assert queue.items["test"].status == WorkStatus.FAILED

    def test_cancel_work(self, queue):
        """Test cancelling work."""
        queue.add_work(WorkItem(work_id="test"))
        success = queue.cancel_work("test")
        assert success is True
        assert queue.items["test"].status == WorkStatus.CANCELLED

    def test_check_timeouts(self, queue):
        """Test timeout detection."""
        item = WorkItem(
            work_id="slow",
            status=WorkStatus.RUNNING,
            claimed_at=time.time() - 100,
            timeout_seconds=50.0,
        )
        queue.items["slow"] = item
        queue._save_item(item)

        timed_out = queue.check_timeouts()
        assert "slow" in timed_out

    def test_get_queue_status(self, queue):
        """Test queue status summary."""
        queue.add_work(WorkItem(work_id="p1", status=WorkStatus.PENDING))
        queue.add_work(WorkItem(work_id="p2", status=WorkStatus.PENDING))
        queue.add_work(WorkItem(work_id="r1", status=WorkStatus.RUNNING))

        # Manually update status for running item
        queue.items["r1"].status = WorkStatus.RUNNING
        queue._save_item(queue.items["r1"])

        status = queue.get_queue_status()
        # pending/running are lists of items, not counts
        assert len(status["pending"]) == 2
        assert len(status["running"]) == 1
        assert "by_type" in status
        assert "by_status" in status

    def test_cleanup_old_items(self, queue):
        """Test cleanup of old completed items."""
        old_item = WorkItem(
            work_id="old",
            status=WorkStatus.COMPLETED,
            completed_at=time.time() - 100000,
        )
        queue.items["old"] = old_item
        queue._save_item(old_item)

        removed = queue.cleanup_old_items(max_age_seconds=1000)
        assert removed == 1
        assert "old" not in queue.items

    def test_get_pending_count(self, queue):
        """Test pending count."""
        queue.add_work(WorkItem(status=WorkStatus.PENDING))
        queue.add_work(WorkItem(status=WorkStatus.PENDING))
        assert queue.get_pending_count() == 2

    def test_get_running_count(self, queue):
        """Test running count."""
        item = WorkItem(work_id="r1", status=WorkStatus.RUNNING)
        queue.items["r1"] = item
        queue._save_item(item)
        assert queue.get_running_count() == 1

    def test_get_history(self, queue):
        """Test work history retrieval."""
        for i in range(5):
            item = WorkItem(work_id=f"item{i}", status=WorkStatus.COMPLETED)
            queue.items[item.work_id] = item
            queue._save_item(item)

        history = queue.get_history(limit=3)
        assert len(history) == 3

    def test_get_history_status_filter(self, queue):
        """Test history with status filter."""
        queue.add_work(WorkItem(work_id="c1", status=WorkStatus.COMPLETED))
        queue.add_work(WorkItem(work_id="f1", status=WorkStatus.FAILED))

        # Update statuses directly
        queue.items["c1"].status = WorkStatus.COMPLETED
        queue.items["f1"].status = WorkStatus.FAILED
        queue._save_item(queue.items["c1"])
        queue._save_item(queue.items["f1"])

        completed = queue.get_history(status_filter="completed")
        assert all(h["status"] == "completed" for h in completed)

    def test_persistence_across_restarts(self, temp_db):
        """Test queue persists data across restarts."""
        # Create queue and add work
        q1 = WorkQueue(db_path=temp_db, policy_manager=None)
        q1.add_work(WorkItem(work_id="persist", priority=75))

        # Create new queue instance
        q2 = WorkQueue(db_path=temp_db, policy_manager=None)
        assert "persist" in q2.items
        assert q2.items["persist"].priority == 75

    def test_concurrent_claims(self, queue):
        """Test concurrent claims from multiple threads."""
        # Add several work items
        for i in range(10):
            queue.add_work(WorkItem(work_id=f"work{i}"))

        claimed_ids = []
        errors = []

        def claim_worker(node_id):
            try:
                for _ in range(3):
                    item = queue.claim_work(node_id)
                    if item:
                        claimed_ids.append(item.work_id)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=claim_worker, args=(f"node-{i}",))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Each work item should only be claimed once
        assert len(claimed_ids) == len(set(claimed_ids))


# =============================================================================
# SlackWorkQueueNotifier Tests
# =============================================================================

class TestSlackWorkQueueNotifier:
    """Tests for Slack notification."""

    def test_notifier_disabled_without_webhook(self):
        """Test notifier is disabled without webhook."""
        notifier = SlackWorkQueueNotifier(webhook_url=None)
        assert notifier.enabled is False

    def test_notifier_enabled_with_webhook(self):
        """Test notifier is enabled with webhook."""
        notifier = SlackWorkQueueNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.enabled is True

    @patch("urllib.request.urlopen")
    def test_on_work_added_high_priority(self, mock_urlopen):
        """Test notification for high-priority work."""
        notifier = SlackWorkQueueNotifier(webhook_url="https://hooks.slack.com/test")
        item = WorkItem(work_id="hp", priority=95, work_type=WorkType.TRAINING)
        notifier.on_work_added(item)
        mock_urlopen.assert_called_once()

    def test_on_work_added_low_priority_no_notification(self):
        """Test no notification for low-priority work."""
        notifier = SlackWorkQueueNotifier(webhook_url="https://hooks.slack.com/test")
        item = WorkItem(work_id="lp", priority=50)
        # Should not raise or send
        with patch("urllib.request.urlopen") as mock:
            notifier.on_work_added(item)
            mock.assert_not_called()


# =============================================================================
# WorkType and WorkStatus Tests
# =============================================================================

class TestWorkTypeEnum:
    """Tests for WorkType enum."""

    def test_all_work_types_defined(self):
        """Test all expected work types exist."""
        expected = [
            "training", "gpu_cmaes", "cpu_cmaes", "tournament",
            "gauntlet", "selfplay", "data_merge", "data_sync",
            "validation", "hyperparam_sweep"
        ]
        for wt in expected:
            assert WorkType(wt) is not None

    def test_work_type_string_value(self):
        """Test WorkType has string values."""
        assert WorkType.TRAINING.value == "training"
        assert WorkType.GPU_CMAES.value == "gpu_cmaes"


class TestWorkStatusEnum:
    """Tests for WorkStatus enum."""

    def test_all_statuses_defined(self):
        """Test all expected statuses exist."""
        expected = [
            "pending", "claimed", "running", "completed",
            "failed", "cancelled", "timeout"
        ]
        for ws in expected:
            assert WorkStatus(ws) is not None
