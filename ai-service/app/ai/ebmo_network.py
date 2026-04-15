"""Compatibility shim for deprecated EBMO modules.

Historically, EBMO lived at `app.ai.ebmo_network`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

Some training utilities and tests (including `tests/test_ebmo_ai.py`) still
import from the legacy location. Keep this module as a thin re-export layer so
those imports remain stable.

Canonical engine/rules code does NOT depend on EBMO.
"""

from __future__ import annotations

# Re-export the deprecated implementation.
# The underlying module already emits a DeprecationWarning.
from app.ai._deprecated_ebmo_network import (
    ActionEncoder,
    ActionFeatureExtractor,
    EBMOConfig,
    EBMONetwork,
    EnergyHead,
    StateEncoder,
    combined_ebmo_loss,
    contrastive_energy_loss,
    extract_state_features,
    hard_negative_contrastive_loss,
    load_ebmo_model,
    margin_ranking_loss,
    outcome_weighted_energy_loss,
    save_ebmo_model,
)

__all__ = [
    "ActionEncoder",
    "ActionFeatureExtractor",
    "EBMOConfig",
    "EBMONetwork",
    "EnergyHead",
    "StateEncoder",
    "combined_ebmo_loss",
    "contrastive_energy_loss",
    "extract_state_features",
    "hard_negative_contrastive_loss",
    "load_ebmo_model",
    "margin_ranking_loss",
    "outcome_weighted_energy_loss",
    "save_ebmo_model",
]
