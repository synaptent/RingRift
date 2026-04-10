"""State persistence, cluster epoch, and metrics facade helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class StatePersistenceMixin(P2PMixinBase):
    """Mixin for P2POrchestrator state persistence, cluster epoch, and metrics facade helpers."""

    MIXIN_TYPE = "state_persistence"

    def _load_state(self):
        """Load persisted state from database.

        Phase 1 Refactoring: Delegated to StateManager.
        The StateManager returns a PersistedState object which is then
        applied to the orchestrator's instance variables.
        """
        try:
            state = self.state_manager.load_state(self.node_id)

            # P2P Hardening Phase 2 (Dec 2025): Validate and clean stale state
            is_valid, issues = self.state_manager.validate_loaded_state(state)
            if issues:
                # Clean up stale entries before applying state
                jobs_removed, peers_removed = self.state_manager.clean_stale_state(state)
                if self.verbose:
                    logger.info(
                        f"[P2POrchestrator] Startup cleanup: removed "
                        f"{jobs_removed} stale jobs, {peers_removed} stale peers"
                    )

            # Apply loaded peers
            for node_id, info_dict in state.peers.items():
                try:
                    info = NodeInfo.from_dict(info_dict)
                    self.peers[node_id] = info
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load peer {node_id}: {e}")
            # C2 fix: Sync peer snapshot after loading persisted peers
            self._sync_peer_snapshot()
            self._publish_peers_snapshot()

            # Apply loaded jobs
            for job_dict in state.jobs:
                try:
                    job = ClusterJob(
                        job_id=job_dict["job_id"],
                        job_type=JobType(job_dict["job_type"]),
                        node_id=job_dict["node_id"],
                        board_type=job_dict.get("board_type", "square8"),
                        num_players=job_dict.get("num_players", 2),
                        engine_mode=job_dict.get("engine_mode", "descent-only"),
                        pid=job_dict.get("pid", 0),
                        started_at=job_dict.get("started_at", 0.0),
                        status=job_dict.get("status", "running"),
                    )
                    self.local_jobs[job.job_id] = job
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load job: {e}")

            # Feb 2026: Clean stale jobs with dead PIDs before gossip starts.
            # Jobs from previous sessions may have PIDs that no longer exist,
            # causing training_jobs/selfplay_jobs to report phantom counts.
            stale_startup_jobs = []
            for job_id, job in list(self.local_jobs.items()):
                pid = getattr(job, "pid", 0) or 0
                if pid > 0 and getattr(job, "status", "") == "running":
                    try:
                        os.kill(pid, 0)  # Check if process exists
                    except ProcessLookupError:
                        stale_startup_jobs.append(job_id)
                    except PermissionError:
                        pass  # Process exists but owned by another user
            if stale_startup_jobs:
                for job_id in stale_startup_jobs:
                    self.local_jobs.pop(job_id, None)
                logger.info(
                    f"[P2POrchestrator] Startup cleanup: removed "
                    f"{len(stale_startup_jobs)} jobs with dead PIDs"
                )

            # Apply leader state
            # C1 fix: Use leader_state_lock for role/leader_id changes
            ls = state.leader_state
            with self.leader_state_lock:
                if ls.leader_id:
                    self.leader_id = ls.leader_id
                if ls.leader_lease_id:
                    self.leader_lease_id = ls.leader_lease_id
                if ls.leader_lease_expires:
                    self.leader_lease_expires = ls.leader_lease_expires
                if ls.last_lease_renewal:
                    self.last_lease_renewal = ls.last_lease_renewal
                if ls.role:
                    with contextlib.suppress(Exception):
                        self.role = NodeRole(ls.role)

                # Feb 23, 2026: Non-coordinator nodes must not load self-leadership.
                # After P2P restart, persisted state may have leader_id=self (from when
                # the node was leader). Without clearing this, the node continues
                # announcing itself as leader via gossip, overriding force_leader.
                _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
                if not _is_coordinator and self.leader_id == self.node_id:
                    logger.info(
                        f"[P2POrchestrator] Non-coordinator: clearing self-leadership "
                        f"loaded from state (was leader_id={self.leader_id})"
                    )
                    self.leader_id = None
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0
                    self.role = NodeRole.FOLLOWER

            # Voter grant state
            if ls.voter_grant_leader_id:
                self.voter_grant_leader_id = ls.voter_grant_leader_id
            if ls.voter_grant_lease_id:
                self.voter_grant_lease_id = ls.voter_grant_lease_id
            if ls.voter_grant_expires:
                self.voter_grant_expires = ls.voter_grant_expires

            # Phase 15.1.1: Restore fenced lease token state
            # These fields may not exist in older state files, so use getattr with defaults
            persisted_epoch = getattr(ls, "lease_epoch", 0) or 0
            persisted_fence = getattr(ls, "fence_token", "") or ""
            persisted_last_seen = getattr(ls, "last_seen_epoch", 0) or 0
            # Only restore if higher than current (monotonic guarantee)
            if persisted_epoch > self._lease_epoch:
                self._lease_epoch = persisted_epoch
            if persisted_fence and not self._fence_token:
                self._fence_token = persisted_fence
            if persisted_last_seen > self._last_seen_epoch:
                self._last_seen_epoch = persisted_last_seen
            if persisted_epoch > 0:
                logger.info(
                    f"[P2POrchestrator] Restored lease fencing: epoch={self._lease_epoch}, "
                    f"last_seen={self._last_seen_epoch}"
                )

            # Feb 2026: Restore forced leader override from persisted state
            if getattr(ls, "forced_leader_override", False):
                self._forced_leader_override = True
                logger.info("[P2P] Restored forced_leader_override from persisted state")

            # Optional persisted voter configuration (convergence helper). Only
            # apply when voters are not explicitly configured via env/config.
            if (
                ls.voter_node_ids
                and not (getattr(self, "voter_node_ids", []) or [])
                and str(getattr(self, "voter_config_source", "none") or "none") == "none"
            ):
                if self.quorum_manager.maybe_adopt_voter_node_ids(ls.voter_node_ids, source="state"):
                    # Sync adopted state back to orchestrator attributes
                    self.voter_node_ids = self.quorum_manager.voter_node_ids
                    self.voter_config_source = self.quorum_manager.voter_config_source
                    self.voter_quorum_size = min(VOTER_MIN_QUORUM, len(self.voter_node_ids)) if self.voter_node_ids else 0

            # Self-heal inconsistent persisted leader state (can happen after
            # abrupt shutdowns or partial writes): never keep role=leader without
            # a matching leader_id.
            if self.role == NodeRole.LEADER and not self.leader_id:
                logger.info("Loaded role=leader but leader_id is empty; stepping down to follower")
                # C1 fix: Use leader_state_lock for role changes
                with self.leader_state_lock:
                    self.role = NodeRole.FOLLOWER
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0

            logger.info(f"Loaded state: {len(self.peers)} peers, {len(self.local_jobs)} jobs")

            # December 2025 P2P Hardening: Validate loaded state on startup
            # This detects stale jobs, stale peers, and expired leases
            is_valid, issues = self.state_manager.validate_loaded_state(state)
            if not is_valid:
                logger.warning(f"[P2P] Startup state validation found {len(issues)} issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                # Clean up stale entries
                stale_jobs_cleared = self.state_manager.clear_stale_jobs_by_age(max_age_hours=24.0)
                stale_peers_cleared = self.state_manager.clear_stale_peers(max_stale_seconds=300.0)
                if stale_jobs_cleared or stale_peers_cleared:
                    logger.info(f"[P2P] Cleared {stale_jobs_cleared} stale jobs, {stale_peers_cleared} stale peers")
            else:
                logger.info("[P2P] Startup state validation passed")

            # Dec 28, 2025 (Phase 7): Load persisted peer health state
            # Jan 28, 2026: Uses health_metrics_manager directly
            try:
                peer_health_states = self.state_manager.load_all_peer_health(max_age_seconds=3600.0)
                if peer_health_states:
                    self.health_metrics_manager.apply_loaded_peer_health(peer_health_states)
                    logger.info(f"[P2P] Loaded {len(peer_health_states)} peer health records")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[P2P] Failed to load peer health state: {e}")

            # Jan 12, 2026: Initialize job snapshot with loaded jobs
            try:
                self._job_snapshot.update(self.local_jobs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[P2P] Failed to initialize job snapshot: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        """Save current state to database.

        Phase 1 Refactoring: Delegated to StateManager.
        Creates a PersistedLeaderState from instance variables and
        passes it to the StateManager for persistence.
        """
        try:
            # Build leader state from instance variables
            role_value = self.role.value if hasattr(self.role, "value") else str(self.role)
            leader_state = PersistedLeaderState(
                leader_id=self.leader_id or "",
                leader_lease_id=self.leader_lease_id or "",
                leader_lease_expires=float(self.leader_lease_expires or 0.0),
                last_lease_renewal=float(self.last_lease_renewal or 0.0),
                role=role_value,
                voter_grant_leader_id=str(getattr(self, "voter_grant_leader_id", "") or ""),
                voter_grant_lease_id=str(getattr(self, "voter_grant_lease_id", "") or ""),
                voter_grant_expires=float(getattr(self, "voter_grant_expires", 0.0) or 0.0),
                voter_node_ids=list(getattr(self, "voter_node_ids", []) or []),
                voter_config_source=str(getattr(self, "voter_config_source", "") or ""),
                # Phase 15.1.1: Fenced lease token state
                lease_epoch=int(getattr(self, "_lease_epoch", 0) or 0),
                fence_token=str(getattr(self, "_fence_token", "") or ""),
                last_seen_epoch=int(getattr(self, "_last_seen_epoch", 0) or 0),
                # Feb 2026: Persist forced leader override across restarts
                forced_leader_override=getattr(self, "_forced_leader_override", False),
            )

            # Delegate to StateManager
            self.state_manager.save_state(
                node_id=self.node_id,
                peers=self.peers,
                jobs=self.local_jobs,
                leader_state=leader_state,
                peers_lock=self.peers_lock,
                jobs_lock=self.jobs_lock,
            )

            # Dec 28, 2025 (Phase 7): Save peer health state
            try:
                # Inline: was _collect_peer_health_states()
                peer_health_states = self.health_metrics_manager.collect_peer_health_states()
                if peer_health_states:
                    saved = self.state_manager.save_peer_health_batch(peer_health_states)
                    if saved > 0 and self.verbose:
                        logger.debug(f"[P2P] Saved {saved} peer health records")
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"[P2P] Error saving peer health state: {e}")

            # Jan 12, 2026: Sync job snapshot for lock-free /status reads
            try:
                self._job_snapshot.update(self.local_jobs)
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"[P2P] Error syncing job snapshot: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save state: {e}")

    def _save_cluster_epoch(self) -> None:
        """Save cluster epoch to database.

        Phase 1 Refactoring: Delegated to StateManager.
        Kept for backward compatibility.
        """
        self.state_manager.set_cluster_epoch(self._cluster_epoch)
        self.state_manager.save_cluster_epoch()

    def _increment_cluster_epoch(self) -> None:
        """Increment cluster epoch (called on leader change).

        Phase 1 Refactoring: Delegated to StateManager.
        Kept for backward compatibility.
        """
        self._cluster_epoch = self.state_manager.increment_cluster_epoch()

    def record_metric(
        self,
        metric_type: str,
        value: float,
        board_type: str | None = None,
        num_players: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a metric to the history table for observability.

        Phase 1 Refactoring: Delegated to MetricsManager.

        Metric types:
        - training_loss: NNUE training loss
        - elo_rating: Model Elo rating
        - gpu_utilization: GPU utilization percentage
        - selfplay_games_per_hour: Game generation rate
        - validation_rate: GPU selfplay validation rate
        - tournament_win_rate: Tournament win rate for new model
        """
        self.metrics_manager.record_metric(
            metric_type=metric_type,
            value=value,
            board_type=board_type,
            num_players=num_players,
            metadata=metadata,
        )

    def get_metrics_history(
        self,
        metric_type: str,
        board_type: str | None = None,
        num_players: int | None = None,
        hours: float = 24,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get metrics history. Feb 2026: Delegates to MetricsManager."""
        return self.metrics_manager.get_history(
            metric_type, board_type, num_players, hours, limit
        )

    def get_metrics_summary(self, hours: float = 24) -> dict[str, Any]:
        """Get metrics summary. Feb 2026: Delegates to MetricsManager."""
        return self.metrics_manager.get_summary(hours)
