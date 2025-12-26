#!/usr/bin/env python3
"""
Comprehensive cluster node benchmarking.

Tests network speed, disk I/O, GPU performance, and latency across all providers.
"""

import subprocess
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import yaml


class NodeBenchmark:
    """Benchmark a single cluster node."""

    def __init__(self, node_id: str, ssh_host: str, ssh_port: int,
                 ssh_user: str, ssh_key: str, provider: str, gpu: str):
        self.node_id = node_id
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.provider = provider
        self.gpu = gpu
        self.ssh_base = f"ssh -p {ssh_port} -i {ssh_key} -o ConnectTimeout=10 -o StrictHostKeyChecking=no {ssh_user}@{ssh_host}"

    def check_reachable(self) -> bool:
        """Check if node is reachable via SSH."""
        try:
            result = subprocess.run(
                f"{self.ssh_base} 'echo ok'",
                shell=True,
                capture_output=True,
                timeout=15
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def ping_latency(self) -> Optional[float]:
        """Measure ping latency in ms."""
        try:
            result = subprocess.run(
                f"ping -c 3 -W 2 {self.ssh_host}",
                shell=True,
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                return None

            # Parse average latency from ping output
            output = result.stdout.decode()
            for line in output.split('\n'):
                if 'avg' in line or 'min/avg/max' in line:
                    # Extract avg value (format: min/avg/max/stddev = 1.2/3.4/5.6/7.8)
                    parts = line.split('=')[-1].strip().split('/')
                    if len(parts) >= 2:
                        return float(parts[1])
            return None
        except Exception as e:
            print(f"Ping failed for {self.node_id}: {e}", file=sys.stderr)
            return None

    def network_speed(self) -> Optional[float]:
        """Test network download speed in MB/s."""
        try:
            # Create 100MB test file, download via SSH, measure throughput
            cmd = (
                f"{self.ssh_base} 'dd if=/dev/zero bs=1M count=100 2>/dev/null' | "
                "pv -f -q -s 100M 2>&1 > /dev/null"
            )

            start = time.time()
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
            elapsed = time.time() - start

            if result.returncode == 0 and elapsed > 0:
                # 100 MB in 'elapsed' seconds
                return 100.0 / elapsed

            # Fallback: simple time-based estimate
            cmd_fallback = f"{self.ssh_base} 'dd if=/dev/zero bs=1M count=100 2>/dev/null | cat > /dev/null'"
            start = time.time()
            subprocess.run(cmd_fallback, shell=True, timeout=30)
            elapsed = time.time() - start

            if elapsed > 0:
                return 100.0 / elapsed
            return None

        except Exception as e:
            print(f"Network test failed for {self.node_id}: {e}", file=sys.stderr)
            return None

    def disk_io_speed(self) -> Optional[Tuple[float, float]]:
        """Test disk write/read speed in MB/s. Returns (write_speed, read_speed)."""
        try:
            # Write test
            write_cmd = (
                f"{self.ssh_base} "
                "'cd /tmp && dd if=/dev/zero of=testfile bs=1M count=512 oflag=direct 2>&1 | grep MB/s'"
            )
            result = subprocess.run(write_cmd, shell=True, capture_output=True, timeout=30)
            write_speed = None
            if result.returncode == 0:
                output = result.stdout.decode()
                # Parse "512 MB, 2.3 s, 220 MB/s" format
                if 'MB/s' in output:
                    try:
                        write_speed = float(output.split('MB/s')[0].split()[-1])
                    except:
                        pass

            # Read test
            read_cmd = (
                f"{self.ssh_base} "
                "'cd /tmp && dd if=testfile of=/dev/null bs=1M iflag=direct 2>&1 | grep MB/s'"
            )
            result = subprocess.run(read_cmd, shell=True, capture_output=True, timeout=30)
            read_speed = None
            if result.returncode == 0:
                output = result.stdout.decode()
                if 'MB/s' in output:
                    try:
                        read_speed = float(output.split('MB/s')[0].split()[-1])
                    except:
                        pass

            # Cleanup
            subprocess.run(f"{self.ssh_base} 'rm -f /tmp/testfile'", shell=True, timeout=10)

            return (write_speed, read_speed)

        except Exception as e:
            print(f"Disk I/O test failed for {self.node_id}: {e}", file=sys.stderr)
            return (None, None)

    def gpu_benchmark(self) -> Optional[float]:
        """Run PyTorch GPU benchmark, return TFLOPS estimate."""
        try:
            # Quick matmul benchmark
            benchmark_code = """
import torch
import time

if not torch.cuda.is_available():
    print("NO_GPU")
    exit(1)

device = torch.device('cuda')
torch.cuda.synchronize()

# Warm up
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark 100 iterations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    c = torch.matmul(a, b)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)
ops = 2 * 4096 * 4096 * 4096 * 100  # FLOPs for 100 matmuls
tflops = (ops / 1e12) / (elapsed_ms / 1000.0)
print(f'{tflops:.2f}')
"""

            # Run benchmark - escape quotes properly
            escaped_code = benchmark_code.replace('"', '\\"').replace('$', '\\$')
            cmd = f'{self.ssh_base} "python3 -c \\"{escaped_code}\\""'
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)

            if result.returncode != 0 or b'NO_GPU' in result.stdout:
                return None

            output = result.stdout.decode().strip()
            try:
                return float(output)
            except:
                return None

        except Exception as e:
            print(f"GPU benchmark failed for {self.node_id}: {e}", file=sys.stderr)
            return None

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks and return results."""
        print(f"[{self.node_id}] Checking reachability...", file=sys.stderr)

        if not self.check_reachable():
            return {
                'node_id': self.node_id,
                'provider': self.provider,
                'gpu': self.gpu,
                'status': 'UNREACHABLE',
                'latency_ms': None,
                'network_mbps': None,
                'disk_write_mbps': None,
                'disk_read_mbps': None,
                'gpu_tflops': None
            }

        print(f"[{self.node_id}] Running benchmarks...", file=sys.stderr)

        # Run benchmarks
        latency = self.ping_latency()
        network = self.network_speed()
        disk_write, disk_read = self.disk_io_speed()
        gpu_tflops = self.gpu_benchmark()

        return {
            'node_id': self.node_id,
            'provider': self.provider,
            'gpu': self.gpu,
            'status': 'OK',
            'latency_ms': latency,
            'network_mbps': network,
            'disk_write_mbps': disk_write,
            'disk_read_mbps': disk_read,
            'gpu_tflops': gpu_tflops
        }


def load_cluster_config(config_path: Path) -> List[Dict]:
    """Load cluster configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    nodes = []
    for node_id, node_info in config.get('hosts', {}).items():
        # Skip non-GPU nodes
        if not node_info.get('gpu') or 'cpu' in node_info.get('role', '').lower():
            continue

        # Determine provider
        provider = 'unknown'
        if 'vast' in node_id:
            provider = 'vast.ai'
        elif 'runpod' in node_id:
            provider = 'runpod'
        elif 'nebius' in node_id:
            provider = 'nebius'
        elif 'vultr' in node_id:
            provider = 'vultr'

        nodes.append({
            'node_id': node_id,
            'ssh_host': node_info.get('ssh_host'),
            'ssh_port': node_info.get('ssh_port', 22),
            'ssh_user': node_info.get('ssh_user', 'root'),
            'ssh_key': node_info.get('ssh_key', '~/.ssh/id_ed25519'),
            'provider': provider,
            'gpu': node_info.get('gpu', 'unknown')
        })

    return nodes


def print_results_table(results: List[Dict]):
    """Print benchmark results as a formatted table."""
    # Group by provider
    by_provider = {}
    for r in results:
        provider = r['provider']
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(r)

    print("\n" + "="*120)
    print("CLUSTER BENCHMARK RESULTS")
    print("="*120)
    print(f"{'Provider':<12} | {'Node ID':<20} | {'GPU':<15} | {'Network':<10} | {'Disk I/O':<12} | {'GPU':<10} | {'Latency':<10} | Status")
    print(f"{'':12} | {'':20} | {'':15} | {'(MB/s)':<10} | {'(MB/s)':<12} | {'(TFLOPS)':<10} | {'(ms)':<10} |")
    print("-"*120)

    for provider in sorted(by_provider.keys()):
        provider_results = by_provider[provider]
        for r in sorted(provider_results, key=lambda x: x['node_id']):
            net = f"{r['network_mbps']:.1f}" if r['network_mbps'] else "N/A"
            disk = f"{r['disk_write_mbps']:.0f}/{r['disk_read_mbps']:.0f}" if r['disk_write_mbps'] and r['disk_read_mbps'] else "N/A"
            gpu = f"{r['gpu_tflops']:.1f}" if r['gpu_tflops'] else "N/A"
            lat = f"{r['latency_ms']:.1f}" if r['latency_ms'] else "N/A"

            print(f"{provider:<12} | {r['node_id']:<20} | {r['gpu']:<15} | {net:<10} | {disk:<12} | {gpu:<10} | {lat:<10} | {r['status']}")
        print("-"*120)

    # Summary statistics
    print("\nSUMMARY BY PROVIDER:")
    print("-"*120)
    print(f"{'Provider':<12} | {'Nodes':<6} | {'Avg Network':<12} | {'Avg Disk W/R':<16} | {'Avg GPU':<12} | {'Avg Latency'}")
    print(f"{'':12} | {'':6} | {'(MB/s)':<12} | {'(MB/s)':<16} | {'(TFLOPS)':<12} | {'(ms)'}")
    print("-"*120)

    for provider in sorted(by_provider.keys()):
        provider_results = [r for r in by_provider[provider] if r['status'] == 'OK']
        if not provider_results:
            continue

        count = len(provider_results)
        avg_net = sum(r['network_mbps'] for r in provider_results if r['network_mbps']) / max(1, sum(1 for r in provider_results if r['network_mbps']))
        avg_disk_w = sum(r['disk_write_mbps'] for r in provider_results if r['disk_write_mbps']) / max(1, sum(1 for r in provider_results if r['disk_write_mbps']))
        avg_disk_r = sum(r['disk_read_mbps'] for r in provider_results if r['disk_read_mbps']) / max(1, sum(1 for r in provider_results if r['disk_read_mbps']))
        avg_gpu = sum(r['gpu_tflops'] for r in provider_results if r['gpu_tflops']) / max(1, sum(1 for r in provider_results if r['gpu_tflops']))
        avg_lat = sum(r['latency_ms'] for r in provider_results if r['latency_ms']) / max(1, sum(1 for r in provider_results if r['latency_ms']))

        print(f"{provider:<12} | {count:<6} | {avg_net:<12.1f} | {avg_disk_w:.0f}/{avg_disk_r:.0f}{'':6} | {avg_gpu:<12.1f} | {avg_lat:.1f}")

    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark cluster nodes')
    parser.add_argument('--config', type=Path,
                       default=Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml',
                       help='Path to distributed_hosts.yaml')
    parser.add_argument('--providers', nargs='+',
                       help='Filter by provider (runpod, vast.ai, nebius, vultr)')
    parser.add_argument('--nodes', nargs='+',
                       help='Filter by specific node IDs')
    parser.add_argument('--output', type=Path,
                       help='Save results to JSON file')
    args = parser.parse_args()

    # Load config
    nodes = load_cluster_config(args.config)

    # Filter
    if args.providers:
        nodes = [n for n in nodes if n['provider'] in args.providers]
    if args.nodes:
        nodes = [n for n in nodes if n['node_id'] in args.nodes]

    print(f"Testing {len(nodes)} nodes...\n", file=sys.stderr)

    # Run benchmarks
    results = []
    for node_info in nodes:
        benchmark = NodeBenchmark(**node_info)
        result = benchmark.run_all_benchmarks()
        results.append(result)
        print(f"[{node_info['node_id']}] Complete", file=sys.stderr)

    # Print results
    print_results_table(results)

    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
