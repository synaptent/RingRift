# SSH Access Guide for RingRift Cluster

## Quick Reference

### Lambda Nodes (via Tailscale)
```bash
# All Lambda nodes accessible via Tailscale IPs
ssh -i ~/.ssh/id_cluster ubuntu@<tailscale_ip>
```

| Node | Tailscale IP | GPU |
|------|-------------|-----|
| lambda-h100 | 100.78.101.123 | H100 PCIe |
| lambda-2xh100 | 100.97.104.89 | 2x H100 |
| lambda-a10 | 100.91.25.13 | A10 |
| lambda-gh200-a | 100.123.183.70 | GH200 480GB |
| lambda-gh200-b | 100.104.34.73 | GH200 480GB |
| lambda-gh200-c | 100.88.35.19 | GH200 480GB |
| lambda-gh200-d | 100.75.84.47 | GH200 480GB |
| lambda-gh200-e | 100.88.176.74 | GH200 480GB |
| lambda-gh200-f | 100.104.165.116 | GH200 480GB |
| lambda-gh200-g | 100.104.126.58 | GH200 480GB |
| lambda-gh200-h | 100.65.88.62 | GH200 480GB |
| lambda-gh200-i | 100.99.27.56 | GH200 480GB |

### Vast.ai Nodes (via Vast SSH Gateway)
```bash
# Use Vast.ai SSH jump hosts
ssh -p <port> root@ssh<N>.vast.ai
```

| Node | SSH Command | GPU |
|------|------------|-----|
| vast-262969f8 | `ssh -p 14398 root@ssh7.vast.ai` | RTX 5090 |
| vast-112df53d | `ssh -p 10014 root@ssh9.vast.ai` | RTX 2080 Ti |
| vast-3060ti | `ssh -p 19766 root@ssh3.vast.ai` | RTX 3060 Ti |
| vast-5070-4x | `ssh -p 10042 root@ssh2.vast.ai` | 4x RTX 5070 |

### Vast.ai Nodes (via Lambda Jump Host)
Some Vast nodes reachable via lambda-h100 as jump host:
```bash
ssh -i ~/.ssh/id_cluster ubuntu@100.78.101.123
# Then from lambda-h100:
ssh root@<tailscale_ip>
```

| Node | Tailscale IP | GPU |
|------|-------------|-----|
| vast-3070 | 100.74.154.36 | RTX 3070 |
| vast-2060s | 100.75.98.13 | RTX 2060 SUPER |

## Tunnel Infrastructure

### Primary Hub: lambda-h100 (209.20.157.81)
All tunnels accessible via `curl http://127.0.0.1:<port>/health`

| Port | Node | GPU |
|------|------|-----|
| 8770 | lambda-h100 (local) | H100 |
| 8771 | vast-5070-4x | 4x RTX 5070 |
| 8772 | lambda-gh200-h | GH200 |
| 8773 | lambda-gh200-a | GH200 |
| 8774 | lambda-gh200-c | GH200 |
| 8775 | lambda-gh200-g | GH200 |
| 8776 | lambda-gh200-i | GH200 |
| 8777 | lambda-a10 | A10 |
| 8778 | vast-3070 | RTX 3070 |
| 8779 | vast-2060s | RTX 2060S |
| 8780 | vast-262969f8 | RTX 5090 |
| 8781 | vast-112df53d | RTX 2080 Ti |
| 8782 | vast-3060ti | RTX 3060 Ti |

### Backup Hub: lambda-gh200-e (192.222.57.162)

| Port | Node |
|------|------|
| 8872 | lambda-gh200-h |
| 8873 | lambda-gh200-a |
| 8874 | lambda-gh200-c |
| 8875 | lambda-gh200-g |
| 8876 | lambda-gh200-i |
| 8877 | vast-3070 |
| 8878 | vast-2060s |

## Troubleshooting

### Tunnel Down
1. Check if autossh is running: `pgrep -f autossh`
2. Restart tunnel: `sudo systemctl restart autossh-tunnel` (Lambda) or re-run nohup command (Vast)
3. Check SSH key in target's authorized_keys

### Node Unreachable
1. Try via tunnel: `curl http://127.0.0.1:<port>/health` from lambda-h100
2. Check Tailscale status: `tailscale status`
3. Try via backup hub

### Run Health Check
```bash
/home/ubuntu/ringrift/scripts/tunnel_health_monitor.sh
```
