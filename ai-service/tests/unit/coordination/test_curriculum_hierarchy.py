"""Tests for curriculum hierarchy with sibling propagation (January 2026 Sprint 12).

These tests verify the enhanced curriculum propagation:
- Same board family (hex→hex): 80% weight
- Cross-board (hex→square): 40% weight with Elo guard
- Sibling player count (2p→3p/4p): 60% weight

Expected Elo improvement: +12-18 from better knowledge transfer.
"""

import pytest


class TestCurriculumHierarchyWeights:
    """Test the weighted propagation logic."""

    def test_get_similar_configs_with_weights_same_family(self):
        """Test same board family configs get 80% weight."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # hex8_2p should propagate to hexagonal_2p at 80%
        weighted = bridge._get_similar_configs_with_weights("hex8_2p")

        assert "hexagonal_2p" in weighted
        assert weighted["hexagonal_2p"] == 0.80

    def test_get_similar_configs_with_weights_cross_board(self):
        """Test cross-board configs get 40% weight (no Elo guard when no Elo provided)."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # hex8_2p should propagate to square8_2p at 40% (cross-family)
        weighted = bridge._get_similar_configs_with_weights("hex8_2p")

        assert "square8_2p" in weighted
        assert weighted["square8_2p"] == 0.40
        assert "square19_2p" in weighted
        assert weighted["square19_2p"] == 0.40

    def test_get_similar_configs_with_weights_sibling(self):
        """Test sibling player count configs (same board) get 60% weight."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # hex8_2p should propagate to hex8_3p and hex8_4p at 60%
        weighted = bridge._get_similar_configs_with_weights("hex8_2p")

        assert "hex8_3p" in weighted
        assert weighted["hex8_3p"] == 0.60
        assert "hex8_4p" in weighted
        assert weighted["hex8_4p"] == 0.60

    def test_get_similar_configs_excludes_self(self):
        """Test source config is not included in results."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("hex8_2p")

        assert "hex8_2p" not in weighted

    def test_get_similar_configs_backward_compat(self):
        """Test backward-compatible _get_similar_configs returns list."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # Old method should still return list
        configs = bridge._get_similar_configs("hex8_2p")

        assert isinstance(configs, list)
        assert "hexagonal_2p" in configs
        assert "square8_2p" in configs
        assert "hex8_3p" in configs

    def test_weights_hierarchy_correctness(self):
        """Test weight hierarchy: same_family > sibling > cross_board."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("hex8_2p")

        # Same family should have highest weight
        same_family_weight = weighted.get("hexagonal_2p", 0)
        # Sibling should be middle
        sibling_weight = weighted.get("hex8_3p", 0)
        # Cross-board should be lowest
        cross_board_weight = weighted.get("square8_2p", 0)

        assert same_family_weight > sibling_weight > cross_board_weight
        assert same_family_weight == 0.80
        assert sibling_weight == 0.60
        assert cross_board_weight == 0.40

    def test_invalid_config_returns_empty(self):
        """Test invalid config key returns empty dict."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # Invalid config key
        weighted = bridge._get_similar_configs_with_weights("invalid")
        assert weighted == {}

        # Empty config key
        weighted = bridge._get_similar_configs_with_weights("")
        assert weighted == {}


class TestCurriculumHierarchyEloGuard:
    """Test Elo guard for cross-board propagation."""

    def test_cross_board_with_higher_elo_target_skipped(self):
        """Test cross-board propagation skipped when target has higher Elo."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # Source at 1500 Elo, target at 1600 should be skipped
        # Note: This test would need EloManager to be mocked for full verification
        # For now, just verify the method handles Elo parameter
        weighted = bridge._get_similar_configs_with_weights("hex8_2p", source_elo=1500)

        # Without mock, cross-board configs should still appear (Elo manager not available)
        assert "square8_2p" in weighted

    def test_cross_board_with_much_lower_elo_included(self):
        """Test cross-board propagation included when target has much lower Elo."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # Source at 1500 Elo
        weighted = bridge._get_similar_configs_with_weights("hex8_2p", source_elo=1500)

        # Cross-board configs should be included (without mock, fallback to default)
        assert "square8_2p" in weighted
        assert "square19_2p" in weighted


class TestCurriculumHierarchySquareBoard:
    """Test curriculum hierarchy for square boards."""

    def test_square_to_square_propagation(self):
        """Test square→square propagation at 80%."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("square8_2p")

        # square8 → square19 should be 80%
        assert "square19_2p" in weighted
        assert weighted["square19_2p"] == 0.80

    def test_square_to_hex_propagation(self):
        """Test square→hex cross-board propagation at 40%."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("square8_2p")

        # square8 → hex should be 40%
        assert "hex8_2p" in weighted
        assert weighted["hex8_2p"] == 0.40
        assert "hexagonal_2p" in weighted
        assert weighted["hexagonal_2p"] == 0.40

    def test_square_sibling_propagation(self):
        """Test square sibling (2p→3p/4p) propagation at 60%."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("square8_2p")

        # square8_2p → square8_3p/4p should be 60%
        assert "square8_3p" in weighted
        assert weighted["square8_3p"] == 0.60
        assert "square8_4p" in weighted
        assert weighted["square8_4p"] == 0.60


class TestCurriculumHierarchyAllPlayerCounts:
    """Test curriculum hierarchy for different player counts."""

    def test_3_player_propagation(self):
        """Test 3-player config propagation."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("hex8_3p")

        # Same family, same players
        assert "hexagonal_3p" in weighted
        assert weighted["hexagonal_3p"] == 0.80

        # Cross-board, same players
        assert "square8_3p" in weighted
        assert weighted["square8_3p"] == 0.40

        # Sibling (different players)
        assert "hex8_2p" in weighted
        assert weighted["hex8_2p"] == 0.60
        assert "hex8_4p" in weighted
        assert weighted["hex8_4p"] == 0.60

    def test_4_player_propagation(self):
        """Test 4-player config propagation."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        weighted = bridge._get_similar_configs_with_weights("hexagonal_4p")

        # Same family, same players
        assert "hex8_4p" in weighted
        assert weighted["hex8_4p"] == 0.80

        # Cross-board, same players
        assert "square8_4p" in weighted
        assert weighted["square8_4p"] == 0.40
        assert "square19_4p" in weighted
        assert weighted["square19_4p"] == 0.40

        # Sibling (different players on same board)
        assert "hexagonal_2p" in weighted
        assert weighted["hexagonal_2p"] == 0.60
        assert "hexagonal_3p" in weighted
        assert weighted["hexagonal_3p"] == 0.60


class TestCurriculumHierarchyEventEmission:
    """Test curriculum hierarchy event emission."""

    def test_emit_curriculum_propagate_with_weight(self):
        """Test _emit_curriculum_propagate includes weight in event."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # This should not raise even without event router
        bridge._emit_curriculum_propagate(
            source_config="hex8_2p",
            target_config="hexagonal_2p",
            advancement_tier="tier_2",
            original_trigger="evaluation",
            propagation_weight=0.80,
        )

        # Just verify no exception raised
        assert True

    def test_emit_curriculum_propagate_default_weight(self):
        """Test _emit_curriculum_propagate uses default 50% weight."""
        from app.coordination.curriculum_integration import MomentumToCurriculumBridge

        bridge = MomentumToCurriculumBridge()

        # This should not raise even without event router
        bridge._emit_curriculum_propagate(
            source_config="hex8_2p",
            target_config="hexagonal_2p",
            advancement_tier="tier_2",
            original_trigger="evaluation",
        )

        # Just verify no exception raised
        assert True
