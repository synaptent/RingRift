"""Relay command polling and stability-controller action helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2 follow-up).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class RelayCommandExecutionMixin(P2PMixinBase):
    """Mixin for P2POrchestrator relay command polling and stability-controller action helpers."""

    MIXIN_TYPE = "relay_command_execution"

    async def _execute_relay_commands(self, commands: list[dict[str, Any]]) -> None:
        """Execute relay commands (polling mode for NAT-blocked nodes)."""
        now = time.time()
        for cmd in commands:
            try:
                cmd_id = str(cmd.get("id") or "")
                cmd_type = str(cmd.get("type") or "")
                payload = cmd.get("payload") or {}
                if not cmd_id or not cmd_type:
                    continue

                # Check for stale commands (>5 min old indicates relay/polling issues)
                cmd_ts = cmd.get("ts") or cmd.get("timestamp") or now
                cmd_age_secs = now - float(cmd_ts)
                if cmd_age_secs > 300:
                    logger.info(f"WARNING: Relay command {cmd_id} ({cmd_type}) is {cmd_age_secs:.0f}s old - relay delivery may be delayed")

                attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0) + 1
                self.relay_command_attempts[cmd_id] = attempts

                ok = False
                err = ""
                if cmd_type == "start_job":
                    job_type = JobType(str(payload.get("job_type") or "selfplay"))
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    engine_mode = str(payload.get("engine_mode") or "mixed")
                    job_id = str(payload.get("job_id") or "")

                    if job_id:
                        with self.jobs_lock:
                            existing = self.local_jobs.get(job_id)
                        if existing and existing.status == "running":
                            ok = True
                        else:
                            job = await self._start_local_job(
                                job_type,
                                board_type=board_type,
                                num_players=num_players,
                                engine_mode=engine_mode,
                                job_id=job_id,
                            )
                            ok = job is not None
                    else:
                        job = await self._start_local_job(
                            job_type,
                            board_type=board_type,
                            num_players=num_players,
                            engine_mode=engine_mode,
                        )
                        ok = job is not None
                elif cmd_type == "cleanup":
                    fire_and_forget(
                        self._cleanup_local_disk(),
                        name=f"cleanup_local_disk:{self.node_id}",
                    )
                    ok = True
                elif cmd_type == "restart_stuck_jobs":
                    fire_and_forget(
                        self._restart_local_stuck_jobs(),
                        name=f"restart_stuck_jobs:{self.node_id}",
                    )
                    ok = True
                elif cmd_type == "reduce_selfplay":
                    target = payload.get("target_selfplay_jobs", payload.get("target", 0))
                    reason = str(payload.get("reason") or "relay")
                    try:
                        target_jobs = int(target)
                    except (ValueError):
                        target_jobs = 0
                    await self._reduce_local_selfplay_jobs(target_jobs, reason=reason)
                    ok = True
                elif cmd_type == "cleanup_files":
                    files = payload.get("files", []) or []
                    reason = str(payload.get("reason") or "relay")
                    if not isinstance(files, list) or not files:
                        ok = False
                        err = "no_files"
                    else:
                        data_dir = self.get_data_directory()
                        freed_bytes = 0
                        deleted_count = 0
                        data_root = data_dir.resolve()
                        for file_path in files:
                            full_path = data_dir / (str(file_path or "").lstrip("/"))
                            try:
                                resolved = full_path.resolve()
                                resolved.relative_to(data_root)
                            except (AttributeError):
                                continue
                            if not resolved.exists():
                                continue
                            try:
                                size = resolved.stat().st_size
                                resolved.unlink()
                                freed_bytes += size
                                deleted_count += 1
                            except (AttributeError):
                                continue
                        print(
                            f"[P2P] Relay cleanup_files: {deleted_count} files deleted, "
                            f"{freed_bytes / 1e6:.1f}MB freed (reason={reason})"
                        )
                        ok = True
                elif cmd_type == "canonical_selfplay":
                    job_id = str(payload.get("job_id") or "")
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    num_games = int(payload.get("num_games") or payload.get("games_per_node") or 500)
                    seed = int(payload.get("seed") or 0)
                    if not job_id:
                        ok = False
                        err = "missing_job_id"
                    else:
                        fire_and_forget(
                            self._run_local_canonical_selfplay(
                                job_id,
                                board_type,
                                num_players,
                                num_games,
                                seed,
                            ),
                            name=f"canonical_selfplay:{job_id}",
                        )
                        ok = True
                else:
                    ok = False
                    err = f"unknown_command_type:{cmd_type}"

                if ok:
                    self._add_pending_relay_ack(cmd_id)
                    self._add_pending_relay_result({"id": cmd_id, "ok": True})
                    self.relay_command_attempts.pop(cmd_id, None)
                else:
                    if not err:
                        err = "command_failed"
                    if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                        self._add_pending_relay_ack(cmd_id)
                        self._add_pending_relay_result({"id": cmd_id, "ok": False, "error": err})
                        self.relay_command_attempts.pop(cmd_id, None)
            except Exception as exc:
                try:
                    cmd_id = str(cmd.get("id") or "")
                    if cmd_id:
                        attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0)
                        if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                            self._add_pending_relay_ack(cmd_id)
                            self._add_pending_relay_result({"id": cmd_id, "ok": False, "error": str(exc)})
                            self.relay_command_attempts.pop(cmd_id, None)
                except (ValueError, AttributeError):
                    continue

    async def _action_increase_timeout(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Increase timeout for affected nodes."""
        if not self._adaptive_timeouts:
            return

        for node_id in nodes:
            self._adaptive_timeouts.increase_timeout(node_id)

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "increase_timeout",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Increased timeout for {len(nodes)} nodes")

    async def _action_decrease_timeout(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Decrease timeout for affected nodes."""
        if not self._adaptive_timeouts:
            return

        for node_id in nodes:
            self._adaptive_timeouts.decrease_timeout(node_id)

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "decrease_timeout",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Decreased timeout for {len(nodes)} nodes")

    async def _action_scale_pool(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Scale up connection pool size."""
        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "scale_pool_up",
                [],
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info("[Stability] Would scale connection pool (not implemented)")

    async def _action_reset_circuits(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Reset circuit breakers for affected nodes.

        January 22, 2026 - P2P Self-Healing Architecture:
        Now resets both node-level and per-transport circuit breakers.
        This enables transport fallover when one transport (e.g., Tailscale) fails.
        """
        reset_count = 0
        transport_reset_count = 0

        # Reset node-level circuit breakers
        try:
            from app.distributed.circuit_breaker import reset_circuit_breaker
            for node_id in nodes:
                try:
                    reset_circuit_breaker(node_id)
                    reset_count += 1
                except Exception as e:
                    logger.debug(f"Failed to reset node circuit for {node_id}: {e}")
        except ImportError:
            logger.debug("Circuit breaker module not available")

        # Reset per-transport circuit breakers for transport fallover
        try:
            from app.distributed.circuit_breaker import reset_transport_breakers_for_host
            for node_id in nodes:
                try:
                    # Get the host/IP for this node
                    peer = self.peers.get(node_id)
                    if peer:
                        host = getattr(peer, "ip", None) or getattr(peer, "host", None) or node_id
                        count = reset_transport_breakers_for_host(host)
                        transport_reset_count += count
                        if count > 0:
                            logger.debug(
                                f"[Stability] Reset {count} transport circuits for {node_id}"
                            )
                except Exception as e:
                    logger.debug(f"Failed to reset transport circuits for {node_id}: {e}")
        except ImportError:
            logger.debug("Transport circuit breaker module not available")

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "reset_circuit",
                nodes,
                {
                    "symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom),
                    "node_circuits_reset": reset_count,
                    "transport_circuits_reset": transport_reset_count,
                },
            )
        logger.info(
            f"[Stability] Reset circuits: {reset_count} node, {transport_reset_count} transport"
        )

    async def _action_increase_cooldown(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Increase cooldown period for recovery actions."""
        if self._stability_controller:
            old_cooldown = self._stability_controller._action_cooldown
            self._stability_controller._action_cooldown = min(old_cooldown * 1.5, 600.0)
            logger.info(
                f"[Stability] Increased action cooldown: {old_cooldown:.0f}s -> "
                f"{self._stability_controller._action_cooldown:.0f}s"
            )

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "increase_cooldown",
                [],
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )

    async def _action_reinject_peer(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Reinject dead peers back into alive state for retry."""
        reinjected = 0
        for node_id in nodes:
            if node_id in self.peers:
                peer = self.peers[node_id]
                if not peer.is_alive():
                    peer.last_seen = time.time()
                    peer.status = "alive"
                    reinjected += 1
                    logger.info(f"[Stability] Reinjected peer {node_id}")

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "reinject_peer",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Reinjected {reinjected}/{len(nodes)} peers")

    async def _action_emit_alert(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Emit alert for manual intervention."""
        symptom_str = symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)
        confidence = symptom.confidence if hasattr(symptom, "confidence") else 0.0
        root_cause = symptom.root_cause if hasattr(symptom, "root_cause") else "unknown"

        logger.warning(
            f"[Stability ALERT] {symptom_str} detected "
            f"(confidence={confidence:.2f}, cause={root_cause}, nodes={len(nodes)})"
        )

        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                DataEventType.STABILITY_ALERT,
                {
                    "symptom": symptom_str,
                    "confidence": confidence,
                    "root_cause": root_cause,
                    "affected_nodes": nodes[:10],
                    "timestamp": time.time(),
                },
                context="relay_command_execution",
                source="relay_command_execution",
            )
        except Exception:
            pass

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "emit_alert",
                nodes,
                {"symptom": symptom_str},
            )
