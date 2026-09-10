"""rig_010 — Cleanup+aux (Arm 2 model) + identity-homotopy curriculum (Arm 3 data).

Model is identical to rig_008 (cleanup bridge + softmax snap + settable T).
Training uses rig_009's id-or-real sampler with ρ schedule.
Aux comes from the model's own cleanup logits (non-bypassable).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rig_008_cleanup_aux.model import CleanupConfig, build_model, CleanupStack
