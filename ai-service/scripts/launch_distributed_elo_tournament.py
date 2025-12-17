#!/usr/bin/env python3
"""Launch large-scale distributed Elo calibration tournament across cluster.

This script:
1. Connects to the P2P leader node
2. Discovers all healthy nodes in the cluster
3. Deploys tournament workers to all nodes via SSH
4. Coordinates match distribution for maximum parallelism
5. Aggregates results into central Elo leaderboard
6. Supports ramdrive for faster game execution

Usage:
    # Run full calibration tournament
    python scripts/launch_distributed_elo_tournament.py --games 50

    # Status check
    python scripts/launch_distributed_elo_tournament.py --status

    # With ramdrive optimization
    python scripts/launch_distributed_elo_tournament.py --games 50 --ramdrive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# AI type configurations for calibration
# Split into lightweight (no NN) and heavyweight (requires NN) types
AI_TYPE_CONFIGS_LIGHTWEIGHT = {
    "random": {"ai_type": "random", "difficulty": 1},
    "heuristic": {"ai_type": "heuristic", "difficulty": 2},
    "minimax_d2": {"ai_type": "minimax", "difficulty": 2, "use_neural_net": False, "max_depth": 2},
    "minimax_d3": {"ai_type": "minimax", "difficulty": 3, "use_neural_net": False, "max_depth": 3},
    "minimax_d4": {"ai_type": "minimax", "difficulty": 4, "use_neural_net": False, "max_depth": 4},
    "mcts_100": {"ai_type": "mcts", "difficulty": 5, "use_neural_net": False, "mcts_iterations": 100},
    "mcts_200": {"ai_type": "mcts", "difficulty": 6, "use_neural_net": False, "mcts_iterations": 200},
    "mcts_400": {"ai_type": "mcts", "difficulty": 7, "use_neural_net": False, "mcts_iterations": 400},
}

# Heavyweight types that require neural networks - run on high-memory nodes only
AI_TYPE_CONFIGS_HEAVYWEIGHT = {
    "minimax_nnue": {"ai_type": "minimax", "difficulty": 4, "use_neural_net": True},
    "mcts_neural": {"ai_type": "mcts", "difficulty": 6, "use_neural_net": True, "mcts_iterations": 400},
    "mcts_neural_high": {"ai_type": "mcts", "difficulty": 7, "use_neural_net": True, "mcts_iterations": 800},
    "policy_only": {"ai_type": "policy_only", "difficulty": 3, "policy_temperature": 0.5},
    "gumbel_mcts": {"ai_type": "gumbel_mcts", "difficulty": 7, "gumbel_num_sampled_actions": 16, "gumbel_simulation_budget": 100},
    "descent": {"ai_type": "descent", "difficulty": 9},
}

# Combined for reference
AI_TYPE_CONFIGS = {**AI_TYPE_CONFIGS_LIGHTWEIGHT, **AI_TYPE_CONFIGS_HEAVYWEIGHT}

# Cluster configuration
LEADER_TAILSCALE_IP = "100.78.101.123"  # lambda-h100
LEADER_PORT = 8770

# Host configurations - loaded dynamically from config + Vast.ai discovery
SSH_HOSTS = {}  # Populated at runtime


def load_hosts_from_config() -> Dict[str, dict]:
    """Load hosts from distributed_hosts.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"

    hosts = {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for name, info in config.get("hosts", {}).items():
            ip = info.get("tailscale_ip") or info.get("ssh_host", "")
            if not ip or info.get("status") != "ready":
                continue

            # Skip SSH proxy hosts (ssh*.vast.ai) - need special handling
            if ip.startswith("ssh") and ".vast.ai" in ip:
                continue

            hosts[name] = {
                "ip": ip,
                "user": info.get("ssh_user", "ubuntu"),
                "path": info.get("ringrift_path", "~/ringrift/ai-service"),
                "gpu": (info.get("gpu", "CPU") or "CPU")[:25],
                "ssh_port": info.get("ssh_port", 22),
            }
    except Exception as e:
        print(f"[Tournament] Warning: Could not load config: {e}")

    return hosts


def discover_vast_instances() -> Dict[str, dict]:
    """Discover Vast.ai instances via vastai CLI."""
    hosts = {}
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return hosts

        instances = json.loads(result.stdout)
        for inst in instances:
            if inst.get("actual_status") != "running":
                continue
            if not inst.get("ssh_host"):
                continue

            inst_id = inst.get("id")
            name = f"vast-{inst_id}"

            # Use SSH proxy connection for Vast instances
            hosts[name] = {
                "ip": inst.get("ssh_host"),
                "user": "root",
                "path": "~/ringrift/ai-service",
                "gpu": (inst.get("gpu_name", "GPU") or "GPU")[:25],
                "ssh_port": inst.get("ssh_port", 22),
                "via_ssh_proxy": True,
            }

    except Exception as e:
        print(f"[Tournament] Vast discovery error: {e}")

    return hosts


def discover_tailscale_peers() -> Dict[str, dict]:
    """Discover nodes via Tailscale CLI."""
    hosts = {}
    try:
        # Get tailscale status as JSON
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return hosts

        status = json.loads(result.stdout)
        peers = status.get("Peer", {})
        self_info = status.get("Self", {})

        for peer_id, peer in peers.items():
            if not peer.get("Online"):
                continue

            hostname = peer.get("HostName", "")
            if not hostname:
                continue

            # Get the Tailscale IP
            tailscale_ips = peer.get("TailscaleIPs", [])
            if not tailscale_ips:
                continue

            ip = tailscale_ips[0]  # Use first IP (usually IPv4)

            # Determine user based on hostname pattern
            user = "ubuntu"  # default
            if "mac" in hostname.lower() or "mbp" in hostname.lower():
                user = "armand"
            elif "vast" in hostname.lower():
                user = "root"

            # Create a normalized name
            name = hostname.replace(" ", "-").lower()
            if name not in hosts:
                hosts[f"ts-{name}"] = {
                    "ip": ip,
                    "user": user,
                    "path": "~/ringrift/ai-service",
                    "gpu": "Unknown",
                    "ssh_port": 22,
                    "via_tailscale": True,
                }

    except FileNotFoundError:
        pass  # Tailscale CLI not installed
    except Exception as e:
        print(f"[Discovery] Tailscale error: {e}")

    return hosts


def discover_all_nodes() -> Dict[str, dict]:
    """Discover all available nodes from all sources."""
    all_hosts = {}

    # 1. Load from config (static configuration with Tailscale IPs)
    config_hosts = load_hosts_from_config()
    all_hosts.update(config_hosts)
    print(f"[Discovery] Loaded {len(config_hosts)} hosts from config")

    # 2. Discover Vast.ai instances via vastai CLI
    vast_hosts = discover_vast_instances()
    # Don't override config hosts with vast discovery
    for name, info in vast_hosts.items():
        if name not in all_hosts:
            all_hosts[name] = info
    print(f"[Discovery] Found {len(vast_hosts)} Vast.ai instances")

    # 3. Discover Tailscale peers
    ts_hosts = discover_tailscale_peers()
    # Don't override existing hosts - tailscale is supplementary
    for name, info in ts_hosts.items():
        # Check if we already have this host by IP
        existing_ips = {h["ip"] for h in all_hosts.values()}
        if info["ip"] not in existing_ips and name not in all_hosts:
            all_hosts[name] = info
    print(f"[Discovery] Found {len(ts_hosts)} Tailscale peers")

    return all_hosts


@dataclass
class MatchResult:
    match_id: str
    agent_a: str
    agent_b: str
    winner: str
    game_length: int
    duration_sec: float
    worker_node: str


@dataclass
class NodeStatus:
    node_id: str
    ip: str
    is_alive: bool
    has_p2p: bool
    selfplay_jobs: int
    can_run_tournament: bool
    gpu: str = "Unknown"


def check_node_health(node_id: str, config: dict) -> NodeStatus:
    """Check if a node is healthy and can run tournament matches.

    First tries direct HTTP health check (fast), then falls back to SSH.
    """
    import urllib.request
    import urllib.error

    ip = config["ip"]
    user = config["user"]
    port = config.get("ssh_port", 22)
    gpu = config.get("gpu", "Unknown")

    # Try direct HTTP health check first (faster than SSH)
    try:
        url = f"http://{ip}:8770/health"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            health = json.loads(response.read().decode('utf-8'))
            return NodeStatus(
                node_id=node_id,
                ip=ip,
                is_alive=True,
                has_p2p=health.get("healthy", False),
                selfplay_jobs=health.get("selfplay_jobs", 0),
                can_run_tournament=health.get("healthy", False),
                gpu=gpu,
            )
    except (urllib.error.URLError, urllib.error.HTTPError):
        pass
    except Exception:
        pass

    # Fallback to SSH-based check
    try:
        ssh_args = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]

        if port != 22:
            ssh_args.extend(["-p", str(port)])

        ssh_args.append(f"{user}@{ip}")
        ssh_args.append("curl -s http://localhost:8770/health 2>/dev/null")

        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=20)

        if result.returncode == 0 and result.stdout.strip():
            try:
                health = json.loads(result.stdout)
                return NodeStatus(
                    node_id=node_id,
                    ip=ip,
                    is_alive=True,
                    has_p2p=health.get("healthy", False),
                    selfplay_jobs=health.get("selfplay_jobs", 0),
                    can_run_tournament=health.get("healthy", False),
                    gpu=gpu,
                )
            except json.JSONDecodeError:
                pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    return NodeStatus(
        node_id=node_id, ip=ip, is_alive=False,
        has_p2p=False, selfplay_jobs=0, can_run_tournament=False, gpu=gpu
    )


def discover_healthy_nodes(max_workers: int = 20) -> Tuple[List[NodeStatus], Dict[str, dict]]:
    """Discover all healthy nodes in the cluster.

    Returns:
        Tuple of (healthy_nodes, all_hosts_config)
    """
    print("[Tournament] Discovering all available nodes...")

    # Get all hosts from all sources
    all_hosts = discover_all_nodes()
    print(f"[Tournament] Total hosts to check: {len(all_hosts)}")

    healthy_nodes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_node_health, node_id, config): (node_id, config)
            for node_id, config in all_hosts.items()
        }

        for future in as_completed(futures):
            node_id, config = futures[future]
            try:
                status = future.result()
                if status.can_run_tournament:
                    healthy_nodes.append(status)
                    gpu = config.get("gpu", "CPU")
                    print(f"  ✓ {status.node_id:<25} P2P OK  jobs={status.selfplay_jobs:<3}  {gpu}")
                else:
                    reason = "No P2P" if not status.has_p2p else "Unreachable"
                    # Only print failures for config hosts (not vast discovery misses)
                    if not node_id.startswith("vast-"):
                        print(f"  ✗ {status.node_id:<25} {reason}")
            except Exception as e:
                print(f"  ✗ {node_id:<25} Error: {e}")

    print(f"\n[Tournament] Found {len(healthy_nodes)} healthy nodes out of {len(all_hosts)}")
    return healthy_nodes, all_hosts


def setup_ramdrive_on_node(node_id: str, config: dict) -> bool:
    """Set up ramdrive on a node for faster game execution."""
    ip = config["ip"]
    user = config["user"]
    path = config.get("path", "~/ringrift/ai-service")
    port = config.get("ssh_port", 22)

    ramdrive_script = f'''
# Create ramdrive if not exists
if [ ! -d /tmp/ringrift_ramdrive ]; then
    mkdir -p /tmp/ringrift_ramdrive
    # Copy essential files to ramdrive
    cp -r {path}/app /tmp/ringrift_ramdrive/ 2>/dev/null || true
    cp -r {path}/data/models /tmp/ringrift_ramdrive/ 2>/dev/null || true
    echo "Ramdrive setup complete"
else
    echo "Ramdrive already exists"
fi
'''

    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]
    if port != 22:
        ssh_args.extend(["-p", str(port)])

    ssh_args.extend([f"{user}@{ip}", ramdrive_script])

    try:
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def run_match_via_p2p(
    node_id: str,
    node_ip: str,
    agent_a: str,
    agent_b: str,
    match_id: str,
    board_type: str = "square8",
    use_ramdrive: bool = False,
    max_retries: int = 2,
) -> Tuple[Optional[MatchResult], Optional[str]]:
    """Run a single match on a remote node via P2P orchestrator endpoint.

    Uses the /tournament/play_elo_match endpoint on port 8770 which supports
    all AI types including MCTS, Descent, Policy-Only, and Gumbel MCTS.

    Returns:
        Tuple of (MatchResult or None, error_message or None)
    """
    import urllib.request
    import urllib.error

    url = f"http://{node_ip}:8770/tournament/play_elo_match"

    # Get full config for each agent
    agent_a_config = AI_TYPE_CONFIGS.get(agent_a, {"ai_type": agent_a})
    agent_b_config = AI_TYPE_CONFIGS.get(agent_b, {"ai_type": agent_b})

    payload = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "agent_a_config": agent_a_config,
        "agent_b_config": agent_b_config,
        "board_type": board_type,
        "use_ramdrive": use_ramdrive,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))

                if result.get("success"):
                    return MatchResult(
                        match_id=result.get("match_id", match_id),
                        agent_a=agent_a,
                        agent_b=agent_b,
                        winner=result.get("winner", "draw"),
                        game_length=result.get("game_length", 0),
                        duration_sec=result.get("duration_sec", 0),
                        worker_node=result.get("worker_node", node_id),
                    ), None
                else:
                    last_error = f"Match failed: {result.get('error', 'unknown')}"

        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
                error_json = json.loads(error_body)
                last_error = f"HTTP {e.code}: {error_json.get('error', error_body[:100])}"
            except Exception:
                last_error = f"HTTP {e.code}: {str(e)}"
        except urllib.error.URLError as e:
            last_error = f"URL error: {e.reason}"
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {str(e)}"
        except TimeoutError:
            last_error = "Timeout (300s)"
        except Exception as e:
            last_error = f"Exception: {type(e).__name__}: {str(e)}"

        # Sleep briefly before retry
        if attempt < max_retries - 1:
            time.sleep(0.5 * (attempt + 1))

    return None, last_error


def run_match_on_node(
    node_id: str,
    config: dict,
    agent_a: str,
    agent_b: str,
    match_id: str,
    board_type: str = "square8",
    use_ramdrive: bool = False,
) -> Tuple[Optional[MatchResult], Optional[str]]:
    """Run a single match on a remote node.

    First tries the P2P orchestrator endpoint (fast, supports all AI types),
    falls back to SSH if P2P is unavailable.

    Returns:
        Tuple of (MatchResult or None, error_message or None)
    """
    ip = config["ip"]

    # Try P2P endpoint first (preferred - supports all AI types)
    result, p2p_error = run_match_via_p2p(
        node_id=node_id,
        node_ip=ip,
        agent_a=agent_a,
        agent_b=agent_b,
        match_id=match_id,
        board_type=board_type,
        use_ramdrive=use_ramdrive,
    )

    if result:
        return result, None

    # Fallback to SSH for nodes without P2P (limited AI type support)
    user = config["user"]
    port = config.get("ssh_port", 22)
    path = config.get("path", "~/ringrift/ai-service")

    # SSH fallback only supports simple AI types
    ai_type_map = {
        "random": ("RANDOM", "RandomAI", 1),
        "heuristic": ("HEURISTIC", "HeuristicAI", 5),
    }

    # Skip SSH fallback for complex AI types
    if agent_a not in ai_type_map or agent_b not in ai_type_map:
        return None, p2p_error or f"P2P failed and SSH fallback not available for {agent_a} vs {agent_b}"

    a_info = ai_type_map[agent_a]
    b_info = ai_type_map[agent_b]

    python_script = f'''
import sys, json, time
sys.path.insert(0, ".")
from app.models import AIConfig, AIType, BoardType, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI

board_type = BoardType.SQUARE8
state = create_initial_state(board_type, 2)
engine = DefaultRulesEngine()

cfg_a = AIConfig(ai_type=AIType.{a_info[0]}, board_type=board_type, difficulty={a_info[2]})
cfg_b = AIConfig(ai_type=AIType.{b_info[0]}, board_type=board_type, difficulty={b_info[2]})

ai_a = {"RandomAI(1, cfg_a)" if a_info[1] == "RandomAI" else "HeuristicAI(1, cfg_a)"}
ai_b = {"RandomAI(2, cfg_b)" if b_info[1] == "RandomAI" else "HeuristicAI(2, cfg_b)"}

start = time.time()
moves = 0
while state.game_status == GameStatus.ACTIVE and moves < 500:
    ai = ai_a if state.current_player == 1 else ai_b
    move = ai.select_move(state)
    if not move:
        break
    state = engine.apply_move(state, move)
    moves += 1

winner = "draw"
if state.winner == 1:
    winner = "agent_a"
elif state.winner == 2:
    winner = "agent_b"

print(json.dumps({{"winner": winner, "moves": moves, "dur": round(time.time() - start, 2)}}))
'''

    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-o", "BatchMode=yes",
    ]
    if port != 22:
        ssh_args.extend(["-p", str(port)])

    ssh_args.append(f"{user}@{ip}")
    cmd = f'cd {path} && source venv/bin/activate 2>/dev/null; python3 -c "{python_script}"'
    ssh_args.append(cmd)

    ssh_error = None
    try:
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and result.stdout.strip():
            try:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("{"):
                        data = json.loads(line)
                        if "winner" in data:
                            return MatchResult(
                                match_id=match_id,
                                agent_a=agent_a,
                                agent_b=agent_b,
                                winner=data.get("winner", "draw"),
                                game_length=data.get("moves", 0),
                                duration_sec=data.get("dur", 0),
                                worker_node=node_id,
                            ), None
            except json.JSONDecodeError as e:
                ssh_error = f"SSH JSON decode: {e}"
        else:
            ssh_error = f"SSH exit code {result.returncode}: {result.stderr[:100] if result.stderr else 'no output'}"
    except subprocess.TimeoutExpired:
        ssh_error = "SSH timeout (120s)"
    except Exception as e:
        ssh_error = f"SSH exception: {e}"

    return None, p2p_error or ssh_error or "Unknown failure"


def calculate_elo(results: List[MatchResult]) -> Dict[str, float]:
    """Calculate Elo ratings from match results."""
    K_FACTOR = 32
    INITIAL_RATING = 1500.0

    agents = set()
    for r in results:
        agents.add(r.agent_a)
        agents.add(r.agent_b)

    ratings = {agent: INITIAL_RATING for agent in agents}

    for r in sorted(results, key=lambda x: x.match_id):
        ra = ratings[r.agent_a]
        rb = ratings[r.agent_b]

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        eb = 1.0 - ea

        if r.winner == "agent_a":
            sa, sb = 1.0, 0.0
        elif r.winner == "agent_b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ratings[r.agent_a] = ra + K_FACTOR * (sa - ea)
        ratings[r.agent_b] = rb + K_FACTOR * (sb - eb)

    return ratings


def run_distributed_tournament(
    agents: List[str],
    games_per_pairing: int = 4,
    board_type: str = "square8",
    use_ramdrive: bool = False,
    max_parallel_per_node: int = 2,
    min_ram_gb: int = 0,
    retry_failed: bool = True,
    max_node_failures: int = 10,
) -> Tuple[List[MatchResult], Dict[str, float]]:
    """Run distributed tournament across all healthy nodes.

    Args:
        agents: List of agent names to include in tournament
        games_per_pairing: Number of games per agent pairing
        board_type: Board type (square8, square19, hexagonal)
        use_ramdrive: Use ramdrive for faster execution
        max_parallel_per_node: Max parallel matches per node
        min_ram_gb: Minimum RAM in GB required for node (for heavyweight agents)
        retry_failed: Retry failed matches on different nodes
        max_node_failures: Skip node after this many failures
    """

    # Discover nodes
    nodes, all_hosts = discover_healthy_nodes()
    if not nodes:
        print("[Tournament] ERROR: No healthy nodes found!")
        return [], {}

    # Filter nodes by minimum RAM if specified
    if min_ram_gb > 0:
        filtered_nodes = []
        for node in nodes:
            config = all_hosts.get(node.node_id, {})
            # Estimate RAM from GPU name or config
            gpu = config.get("gpu", "")
            # High-RAM GPUs: A40 (1TB), H100 (645GB), GH200 (96GB)
            high_ram_gpus = ["A40", "H100", "GH200", "5090", "4080"]
            if any(g in gpu for g in high_ram_gpus):
                filtered_nodes.append(node)
            elif "Lambda" in node.node_id or "lambda" in node.node_id:
                filtered_nodes.append(node)  # Lambda nodes have 80GB+ RAM

        if filtered_nodes:
            print(f"[Tournament] Filtered to {len(filtered_nodes)} high-RAM nodes (>={min_ram_gb}GB)")
            nodes = filtered_nodes
        else:
            print(f"[Tournament] WARNING: No nodes meet {min_ram_gb}GB RAM requirement, using all nodes")

    # Setup ramdrive if requested
    if use_ramdrive:
        print("\n[Tournament] Setting up ramdrive on nodes...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(setup_ramdrive_on_node, n.node_id, all_hosts[n.node_id]): n.node_id
                for n in nodes if n.node_id in all_hosts
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    if future.result():
                        print(f"  ✓ {node_id}: Ramdrive ready")
                except Exception:
                    pass

    # Generate matchups
    matchups = []
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            for _ in range(games_per_pairing):
                matchups.append((a, b))

    import random
    random.shuffle(matchups)

    print(f"\n[Tournament] Starting tournament:")
    print(f"  Agents: {len(agents)}")
    print(f"  Matchups: {len(matchups)}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Est. matches/node: {len(matchups) // len(nodes)}")

    # Distribute matches across nodes
    results: List[MatchResult] = []
    tournament_id = f"elo_dist_{uuid.uuid4().hex[:8]}"

    # Create worker tasks
    total = len(matchups)
    completed = 0
    failed = 0
    start_time = time.time()

    # Track errors for debugging
    from collections import Counter
    error_counts: Counter = Counter()
    node_failures: Counter = Counter()
    disabled_nodes: Set[str] = set()

    # Round-robin node assignment with parallel execution
    def get_available_nodes():
        """Get nodes that haven't exceeded failure threshold."""
        return [n for n in nodes if n.node_id not in disabled_nodes]

    def process_match(args):
        idx, (agent_a, agent_b), node = args
        match_id = f"{tournament_id}_{idx}"

        # Check if node is still available
        if node.node_id in disabled_nodes:
            return None, f"Node {node.node_id} disabled due to failures"

        config = all_hosts.get(node.node_id, {})
        if not config:
            return None, "No config for node"
        return run_match_on_node(node.node_id, config, agent_a, agent_b, match_id, board_type, use_ramdrive)

    # Distribute work across nodes
    max_workers = len(nodes) * max_parallel_per_node
    node_cycle = 0
    work_items = []

    for idx, matchup in enumerate(matchups):
        node = nodes[node_cycle % len(nodes)]
        node_cycle += 1
        work_items.append((idx, matchup, node))

    print(f"\n[Tournament] Running {len(matchups)} matches with {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_match, item): item for item in work_items}

        for future in as_completed(futures):
            item = futures[future]
            idx, (agent_a, agent_b), node = item

            try:
                result, error = future.result()
                if result:
                    results.append(result)
                    completed += 1
                else:
                    failed += 1
                    # Categorize error
                    error_key = error[:50] if error else "Unknown"
                    error_counts[error_key] += 1
                    node_failures[node.node_id] += 1

                    # Disable node if too many failures
                    if node_failures[node.node_id] >= max_node_failures:
                        if node.node_id not in disabled_nodes:
                            disabled_nodes.add(node.node_id)
                            print(f"  [!] Node {node.node_id} disabled after {max_node_failures} failures")
            except Exception as e:
                failed += 1
                error_counts[f"Exception: {type(e).__name__}"] += 1
                node_failures[node.node_id] += 1

                # Disable node if too many failures
                if node_failures[node.node_id] >= max_node_failures:
                    if node.node_id not in disabled_nodes:
                        disabled_nodes.add(node.node_id)
                        print(f"  [!] Node {node.node_id} disabled after {max_node_failures} failures")

            # Progress update every 10 matches
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed if elapsed > 0 else 0
                eta = (total - completed - failed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total} ({failed} failed) - {rate:.1f} matches/sec - ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n[Tournament] Initial pass completed!")
    print(f"  Total: {completed} matches in {elapsed:.1f}s")
    print(f"  Failed: {failed}")
    print(f"  Rate: {completed/elapsed:.2f} matches/sec")

    # Retry failed matches on different nodes if enabled
    if retry_failed and failed > 0:
        available_nodes = get_available_nodes()
        if available_nodes and failed < total:
            print(f"\n[Tournament] Retrying {failed} failed matches on {len(available_nodes)} available nodes...")

            # Collect failed matches
            completed_indices = set()
            for r in results:
                # Extract index from match_id
                try:
                    idx = int(r.match_id.split("_")[-1])
                    completed_indices.add(idx)
                except (ValueError, IndexError):
                    pass

            # Create retry work items
            retry_items = []
            for idx, matchup in enumerate(matchups):
                if idx not in completed_indices:
                    node = available_nodes[len(retry_items) % len(available_nodes)]
                    retry_items.append((idx, matchup, node))

            retry_completed = 0
            retry_failed = 0

            with ThreadPoolExecutor(max_workers=min(len(available_nodes) * 2, 10)) as executor:
                futures = {executor.submit(process_match, item): item for item in retry_items}
                for future in as_completed(futures):
                    try:
                        result, error = future.result()
                        if result:
                            results.append(result)
                            retry_completed += 1
                        else:
                            retry_failed += 1
                    except Exception:
                        retry_failed += 1

            completed += retry_completed
            failed = retry_failed  # Update failed count
            print(f"  Retry: {retry_completed} succeeded, {retry_failed} still failed")

    # Print error breakdown if there were failures
    if error_counts:
        print(f"\n[Tournament] Error breakdown:")
        for error, count in error_counts.most_common(10):
            print(f"  {count:>4}x {error}")

    # Print node failure breakdown
    if node_failures:
        print(f"\n[Tournament] Failures by node:")
        for node_id, count in node_failures.most_common(10):
            print(f"  {count:>4}x {node_id}")

    # Calculate Elo
    ratings = calculate_elo(results)

    return results, ratings


def print_leaderboard(ratings: Dict[str, float], results: List[MatchResult]):
    """Print Elo leaderboard."""
    print("\n" + "=" * 70)
    print("ELO CALIBRATION RESULTS")
    print("=" * 70)

    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6} {'Agent':<25} {'Elo':<8} {'Description':<30}")
    print("-" * 70)

    for i, (agent, rating) in enumerate(sorted_ratings, 1):
        desc = AI_TYPE_CONFIGS.get(agent, {}).get("ai_type", "unknown")
        wins = sum(1 for r in results if (r.agent_a == agent and r.winner == "agent_a") or (r.agent_b == agent and r.winner == "agent_b"))
        losses = sum(1 for r in results if (r.agent_a == agent and r.winner == "agent_b") or (r.agent_b == agent and r.winner == "agent_a"))
        draws = sum(1 for r in results if (r.agent_a == agent or r.agent_b == agent) and r.winner == "draw")
        record = f"{wins}W/{losses}L/{draws}D"
        print(f"{i:<6} {agent:<25} {rating:>7.1f} {record:<15} {desc}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Launch distributed Elo calibration tournament")
    parser.add_argument("--games", type=int, default=4, help="Games per pairing (default: 4)")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--ramdrive", action="store_true", help="Use ramdrive for faster execution")
    parser.add_argument("--status", action="store_true", help="Check cluster status only")
    parser.add_argument("--max-parallel", type=int, default=2, help="Max parallel matches per node")
    parser.add_argument("--agents", type=str, help="Comma-separated list of agents (default: lightweight)")
    parser.add_argument("--heavyweight", action="store_true", help="Include heavyweight NN-based agents")
    parser.add_argument("--min-ram", type=int, default=0, help="Minimum RAM in GB for nodes (filters to high-capacity nodes)")
    parser.add_argument("--no-retry", action="store_true", help="Don't retry failed matches")
    parser.add_argument("--max-node-failures", type=int, default=10, help="Disable node after N failures")

    args = parser.parse_args()

    if args.status:
        nodes, all_hosts = discover_healthy_nodes()
        print(f"\nCluster ready with {len(nodes)} healthy nodes out of {len(all_hosts)} total")
        return

    # Select agents - use lightweight by default to avoid OOM on nodes
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",")]
    elif args.heavyweight:
        agents = list(AI_TYPE_CONFIGS.keys())
    else:
        agents = list(AI_TYPE_CONFIGS_LIGHTWEIGHT.keys())

    print(f"\n[Tournament] AI Type Calibration Tournament")
    print(f"[Tournament] Agents: {agents}")
    print(f"[Tournament] Games per pairing: {args.games}")
    print(f"[Tournament] Ramdrive: {'enabled' if args.ramdrive else 'disabled'}")
    print(f"[Tournament] Retry failed: {not args.no_retry}")
    print(f"[Tournament] Min RAM filter: {args.min_ram}GB" if args.min_ram > 0 else "[Tournament] Min RAM filter: disabled")

    results, ratings = run_distributed_tournament(
        agents=agents,
        games_per_pairing=args.games,
        board_type=args.board,
        use_ramdrive=args.ramdrive,
        max_parallel_per_node=args.max_parallel,
        min_ram_gb=args.min_ram,
        retry_failed=not args.no_retry,
        max_node_failures=args.max_node_failures,
    )

    if ratings:
        print_leaderboard(ratings, results)

        # Save results
        output_dir = Path("results/elo_calibration")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"calibration_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump({
                "ratings": ratings,
                "results": [{"match_id": r.match_id, "agent_a": r.agent_a, "agent_b": r.agent_b,
                            "winner": r.winner, "game_length": r.game_length,
                            "duration_sec": r.duration_sec, "worker_node": r.worker_node}
                           for r in results],
                "timestamp": timestamp,
            }, f, indent=2)

        print(f"\n[Tournament] Results saved to {results_file}")
    else:
        print("\n[Tournament] No results collected")


if __name__ == "__main__":
    main()
