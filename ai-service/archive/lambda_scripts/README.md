# Archived Lambda Scripts

These scripts were used for Lambda Labs GPU cloud management.

**Archived:** December 26, 2025
**Reason:** Lambda Labs account terminated

## Archived Files

- `lambda_gpu_setup.sh` - Initial GPU instance setup for Lambda nodes
- `setup_new_lambda_instances.sh` - Batch setup for new Lambda instances
- `lambda_cli.py` - CLI wrapper for Lambda Labs API
- `lambda_watchdog.py` - Health monitoring for Lambda nodes

## Migration Notes

Lambda nodes have been replaced by:
- Nebius (H100, L40S)
- RunPod (H100, A100, L40S, 3090 Ti)
- Vast.ai (various consumer GPUs)
- Vultr (A100 vGPU)

For similar functionality on new providers, see:
- `scripts/vast_keepalive.py` - Vast.ai node management
- `scripts/vast_autoscaler.py` - Vast.ai auto-scaling
- P2P orchestrator handles cross-provider coordination
