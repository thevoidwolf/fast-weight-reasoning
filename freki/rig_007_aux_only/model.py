"""rig_007 — FREKI K_chain=3 for Arm 1 (aux-only middle-ground design).

Model itself is IDENTICAL to rig_003_freki with K_chain=3 (no cleanup
bridge, no architectural change). All the aux machinery lives in run.py
so the model file stays a faithful FREKI variant.
"""
from __future__ import annotations

# Delegate to rig_003_freki's model (single M, K_chain-step chained read).
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rig_003_freki.model import (
    FrekiConfig as _FrekiConfig,
    build_model as _build_model,
    FrekiStack,
    FrekiBlock,
    FrekiMixer,
    RMSNorm,
)

# Re-export under rig_007 names for clarity in the run scripts / logs.
Rig7Config = _FrekiConfig
build_model = _build_model
