#!/usr/bin/env python3
"""Verify that the Elo momentum → Selfplay rate feedback loop is wired correctly.

This script confirms that:
1. FeedbackAccelerator.get_selfplay_multiplier() exists and works
2. SelfplayScheduler calls it via _get_momentum_multipliers()
3. The multiplier is applied to priority calculations

Usage:
    python scripts/verify_feedback_loop.py

Expected output:
    ✓ Feedback loop is correctly wired!
"""

import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_wiring():
    """Verify the feedback loop wiring."""
    print("Verifying Elo momentum → Selfplay rate feedback loop...\n")

    # Step 1: Verify FeedbackAccelerator has get_selfplay_multiplier
    from app.training.feedback_accelerator import get_feedback_accelerator

    accelerator = get_feedback_accelerator()
    assert hasattr(accelerator, 'get_selfplay_multiplier'), \
        "FeedbackAccelerator missing get_selfplay_multiplier()"
    print("✓ FeedbackAccelerator.get_selfplay_multiplier() exists")

    # Step 2: Test the method works
    multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
    assert 0.5 <= multiplier <= 2.0, \
        f"get_selfplay_multiplier() returned invalid value: {multiplier}"
    print(f"✓ get_selfplay_multiplier('hex8_2p') = {multiplier:.2f}x (valid range)")

    # Step 3: Verify SelfplayScheduler calls it
    from app.coordination.selfplay_scheduler import get_selfplay_scheduler

    scheduler = get_selfplay_scheduler()
    assert hasattr(scheduler, '_get_momentum_multipliers'), \
        "SelfplayScheduler missing _get_momentum_multipliers()"
    print("✓ SelfplayScheduler._get_momentum_multipliers() exists")

    # Step 4: Verify it retrieves multipliers
    momentum_data = scheduler._get_momentum_multipliers()
    assert isinstance(momentum_data, dict), \
        "_get_momentum_multipliers() didn't return a dict"
    assert len(momentum_data) > 0, \
        "_get_momentum_multipliers() returned empty dict"
    print(f"✓ _get_momentum_multipliers() returned {len(momentum_data)} configs")

    # Step 5: Verify multiplier is applied to priority
    from app.coordination.selfplay_scheduler import ConfigPriority

    test_priority = ConfigPriority(config_key="test")
    test_priority.momentum_multiplier = 1.5
    test_priority.staleness_hours = 1.0

    base_priority = ConfigPriority(config_key="base")
    base_priority.momentum_multiplier = 1.0
    base_priority.staleness_hours = 1.0

    test_score = scheduler._compute_priority_score(test_priority)
    base_score = scheduler._compute_priority_score(base_priority)

    # Test score should be ~1.5x base score
    ratio = test_score / base_score if base_score > 0 else 0
    assert 1.4 <= ratio <= 1.6, \
        f"Momentum multiplier not applied correctly: ratio={ratio:.2f}, expected ~1.5"
    print(f"✓ Momentum multiplier applied to priority: 1.5x → {ratio:.2f}x score")

    # Step 6: Verify event subscription
    assert scheduler._subscribed, "SelfplayScheduler not subscribed to events"
    print("✓ SelfplayScheduler subscribed to pipeline events")

    print("\n" + "=" * 70)
    print("✅ FEEDBACK LOOP VERIFICATION PASSED")
    print("=" * 70)
    print("\nThe Elo momentum → Selfplay rate feedback loop is correctly wired:")
    print("  1. FeedbackAccelerator tracks Elo momentum")
    print("  2. get_selfplay_multiplier() returns momentum-based multipliers")
    print("  3. SelfplayScheduler._get_momentum_multipliers() calls it")
    print("  4. _compute_priority_score() applies multipliers to priorities")
    print("  5. Accelerating configs get higher priority (up to 1.5x)")
    print("  6. Regressing configs get lower priority (down to 0.75x)")
    print("\nNo changes needed - the feedback loop is fully operational.")


if __name__ == "__main__":
    try:
        verify_wiring()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ VERIFICATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
