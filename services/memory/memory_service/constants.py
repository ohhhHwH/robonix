"""Shared time constants for photographic-memory shot gating (design §5).

One constant used by both the scene hook dedup (``service.py``) and the
VLM observer (``vlm_observer.py``): a re-observation of the same object at
the same position is treated as the same shot while it falls within
``SHOT_MIN_INTERVAL_S`` of the node's latest shot, and as a new shot worth
appending once that window has elapsed.
"""

from __future__ import annotations

import os

# Seconds between distinct shots of the same object+position.  Overridable
# per deployment (design §5.3) via the env var named in the design.
SHOT_MIN_INTERVAL_S = float(os.environ.get("VLM_OBSERVER_SHOT_GAP_S", "600"))

# Same threshold in nanoseconds — the unit ``MemoryNode.timestamp`` uses.
SHOT_MIN_INTERVAL_NS = int(SHOT_MIN_INTERVAL_S * 1_000_000_000)
