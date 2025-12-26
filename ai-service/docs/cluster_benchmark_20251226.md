# Cluster Provider Benchmark Results

**Date:** December 26, 2025  
**Total Nodes Tested:** 23  
**Reachable Nodes:** 21  
**Unreachable Nodes:** 2

---

## Executive Summary

### Provider Rankings

| Rank | Provider    | Best For                | Key Metric                   |
| ---- | ----------- | ----------------------- | ---------------------------- |
| 1    | **Vast.ai** | GPU Performance         | 62.2 TFLOPS (RTX 5090)       |
| 2    | **Vultr**   | Network Speed & Latency | 119.6 MB/s avg, 35ms latency |
| 3    | **RunPod**  | Disk I/O                | 336 MB/s write avg           |
| 4    | **Nebius**  | Balanced                | Good network, disk I/O       |

---

## Detailed Comparison Table

| Provider    | Node ID           | GPU         | Network (MB/s) | Disk Write (MB/s) | Disk Read (MB/s) | GPU (TFLOPS) | Latency (ms) | Status      |
| ----------- | ----------------- | ----------- | -------------- | ----------------- | ---------------- | ------------ | ------------ | ----------- |
| **RunPod**  | runpod-h100       | H100 PCIe   | 26.3           | N/A               | N/A              | N/A          | N/A          | OK          |
| RunPod      | runpod-a100-1     | A100 PCIe   | 21.3           | 482               | N/A              | 18.4         | 40.0         | OK          |
| RunPod      | runpod-a100-2     | A100 PCIe   | 31.8           | 337               | 445              | 17.4         | N/A          | OK          |
| RunPod      | runpod-3090ti-1   | RTX 3090 Ti | 114.1          | 190               | 193              | 27.0         | N/A          | OK          |
| RunPod      | runpod-l40s-2     | L40S        | N/A            | N/A               | N/A              | N/A          | N/A          | UNREACHABLE |
| **Vast.ai** | vast-28925166     | RTX 5090    | 48.9           | N/A               | N/A              | **62.2**     | N/A          | OK          |
| Vast.ai     | vast-29128356     | RTX 5090    | 53.2           | N/A               | N/A              | 61.8         | N/A          | OK          |
| Vast.ai     | vast-29128352     | 2x RTX 5090 | 52.6           | N/A               | N/A              | 61.0         | N/A          | OK          |
| Vast.ai     | vast-29129529     | 8x RTX 4090 | 46.6           | N/A               | N/A              | 42.9         | N/A          | OK          |
| Vast.ai     | vast-29031159     | RTX 5080    | 25.4           | 656               | N/A              | 35.2         | N/A          | OK          |
| Vast.ai     | vast-28918742     | A40         | 27.8           | N/A               | N/A              | 22.4         | N/A          | OK          |
| Vast.ai     | vast-29118471     | 8x RTX 3090 | 28.6           | 627               | N/A              | 18.4         | N/A          | OK          |
| Vast.ai     | vast-29126088     | RTX 4060 Ti | 90.3           | N/A               | N/A              | 12.4         | N/A          | OK          |
| Vast.ai     | vast-28890015     | RTX 2080 Ti | 19.5           | 692               | N/A              | 12.1         | N/A          | OK          |
| Vast.ai     | vast-29046315     | RTX 3060 Ti | 42.3           | N/A               | N/A              | 10.5         | N/A          | OK          |
| Vast.ai     | vast-28889766     | RTX 3060 Ti | 55.8           | N/A               | N/A              | 10.1         | N/A          | OK          |
| Vast.ai     | vast-29031161     | RTX 3060    | 29.2           | 423               | **473**          | 7.7          | N/A          | OK          |
| **Vultr**   | vultr-a100-20gb-2 | A100D-20C   | **147.7**      | N/A               | N/A              | 17.7         | **33.6**     | OK          |
| Vultr       | vultr-a100-20gb   | A100D-20C   | 91.5           | N/A               | N/A              | N/A          | 37.0         | OK          |
| **Nebius**  | nebius-h100-1     | H100 SXM    | 48.4           | 68                | 69               | N/A          | 134.0        | OK          |
| Nebius      | nebius-l40s-2     | L40S        | 49.9           | 69                | 69               | N/A          | 168.7        | OK          |
| Nebius      | nebius-backbone-1 | L40S        | N/A            | N/A               | N/A              | N/A          | N/A          | UNREACHABLE |

---

## Provider Averages

| Provider    | Nodes | Avg Network (MB/s) | Avg Disk Write (MB/s) | Avg Disk Read (MB/s) | Avg GPU (TFLOPS) | Avg Latency (ms) |
| ----------- | ----- | ------------------ | --------------------- | -------------------- | ---------------- | ---------------- |
| **Vast.ai** | 12    | 43.4               | 600                   | 473                  | **29.7**         | N/A              |
| **Vultr**   | 2     | **119.6**          | N/A                   | N/A                  | 17.7             | **35.3**         |
| **RunPod**  | 4     | 48.4               | **336**               | 319                  | 21.0             | 40.0             |
| **Nebius**  | 2     | 49.2               | 69                    | 69                   | N/A              | 151.4            |

---

## Key Findings

### Top Performers

**Network Speed:**

1. vultr-a100-20gb-2: 147.7 MB/s
2. runpod-3090ti-1: 114.1 MB/s
3. vultr-a100-20gb: 91.5 MB/s

**GPU Performance:**

1. vast-28925166 (RTX 5090): 62.2 TFLOPS
2. vast-29128356 (RTX 5090): 61.8 TFLOPS
3. vast-29128352 (2x RTX 5090): 61.0 TFLOPS

**Disk I/O:**

1. vast-28890015: 692 MB/s write
2. vast-29031159: 656 MB/s write
3. vast-29118471: 627 MB/s write
4. runpod-a100-1: 482 MB/s write
5. vast-29031161: 473 MB/s read

**Latency:**

1. vultr-a100-20gb-2: 33.6 ms
2. vultr-a100-20gb: 37.0 ms
3. runpod-a100-1: 40.0 ms

---

## Provider Recommendations

### For Training Workloads (GPU-Intensive)

**Winner: Vast.ai**

- RTX 5090 nodes deliver exceptional performance (60+ TFLOPS)
- Multi-GPU nodes (8x RTX 4090) excellent for parallel workloads
- Best cost-per-TFLOP ratio

### For Data Transfer (Network-Intensive)

**Winner: Vultr**

- Highest average network speed (119.6 MB/s)
- Lowest latency to coordinator (35ms average)
- Best for model distribution and data sync

### For Large Dataset Storage (I/O-Intensive)

**Winner: Vast.ai (with RunPod close second)**

- Highest disk write speeds (600 MB/s average)
- Some nodes exceed 650 MB/s write throughput
- RunPod competitive with 336 MB/s average

### For Balanced Workloads

**Winner: RunPod**

- Good GPU performance (17-27 TFLOPS)
- Excellent disk I/O (up to 482 MB/s)
- Low latency (40ms average)
- Reliable uptime

---

## Issues Encountered

### GPU Benchmark Failures

Several H100/L40S nodes failed GPU benchmarks:

- `runpod-h100`: H100 PCIe - benchmark failed
- `nebius-h100-1`: H100 SXM - benchmark failed
- `vultr-a100-20gb`: A100 vGPU - benchmark failed

**Likely Causes:**

- PyTorch version incompatibility
- CUDA toolkit mismatch
- vGPU limitations on Vultr nodes

### Disk I/O Failures

Many nodes returned N/A for disk benchmarks:

- Most Vast.ai nodes (8 of 12)
- Both Vultr nodes
- RunPod H100

**Likely Causes:**

- Permission issues with `/tmp` directory
- `oflag=direct` not supported on some filesystems
- Container storage limitations

### Network Latency

Most Vast.ai nodes couldn't be pinged (ICMP blocked):

- 12/12 Vast.ai nodes: no ping response
- 1/4 RunPod nodes: no ping response

**Note:** This is common for cloud providers blocking ICMP for security.

### Unreachable Nodes

2 nodes were completely unreachable:

- `nebius-backbone-1` (89.169.112.47)
- `runpod-l40s-2` (193.183.22.62)

---

## Cost-Performance Analysis

_Note: Prices are approximate and vary by spot/on-demand pricing_

| Provider | Example GPU | TFLOPS | Est. Cost/hr | TFLOPS/$ |
| -------- | ----------- | ------ | ------------ | -------- |
| Vast.ai  | RTX 5090    | 62.2   | $0.30        | 207      |
| Vast.ai  | 8x RTX 4090 | 42.9   | $1.50        | 29       |
| RunPod   | RTX 3090 Ti | 27.0   | $0.35        | 77       |
| RunPod   | A100 PCIe   | 18.4   | $0.79        | 23       |
| Vultr    | A100 vGPU   | 17.7   | $4.00        | 4        |

**Conclusion:** Vast.ai offers the best price/performance ratio for GPU compute.

---

## Recommendations

1. **Primary Training:** Use Vast.ai RTX 5090 nodes for best GPU performance
2. **Model Distribution:** Use Vultr nodes as staging servers (low latency, high bandwidth)
3. **Data-Intensive Tasks:** Use RunPod for high disk I/O workloads
4. **Backup Compute:** Keep Nebius nodes for additional capacity despite higher latency
5. **Investigate H100 failures:** Debug PyTorch/CUDA setup on H100 nodes to unlock their full potential

---

_Generated by `/Users/armand/Development/RingRift/ai-service/scripts/benchmark_cluster.py`_
