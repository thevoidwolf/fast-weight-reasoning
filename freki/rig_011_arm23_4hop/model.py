"""rig_011 — same CleanupStack model as rig_008 / rig_010, K_chain=4 for 4-hop."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rig_008_cleanup_aux.model import CleanupConfig, build_model, CleanupStack
