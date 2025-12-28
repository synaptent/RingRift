# Deprecated P2P Modules

This directory contains archived P2P modules that have been deprecated and consolidated.

## December 28, 2025: GossipMetricsMixin → GossipProtocolMixin

**`_deprecated_gossip_metrics.py`** (226 LOC)

The `GossipMetricsMixin` class has been merged into `GossipProtocolMixin` in `scripts/p2p/gossip_protocol.py`.

**Reason for deprecation:**

- GossipMetricsMixin was always used together with GossipProtocolMixin
- Merging reduces mixin count and simplifies inheritance chain
- GossipProtocolMixin now inherits from P2PMixinBase for consistent patterns

**Migration:**

- `from scripts.p2p.gossip_metrics import GossipMetricsMixin`
  → `from scripts.p2p.gossip_protocol import GossipProtocolMixin`
- Or use backward-compat alias: `from scripts.p2p import GossipMetricsMixin` (deprecated)

**Methods moved:**

- `_record_gossip_metrics(event, peer_id, latency_ms)`
- `_reset_gossip_metrics_hourly()`
- `_record_gossip_compression(original_size, compressed_size)`
- `_get_gossip_metrics_summary()`
- `_get_gossip_health_status()`
- `calculate_compression_ratio()` (standalone function)

**Removal date:** Q2 2026
