"""Image Store — local image persistence under data/images/{node_id}/.

Each MemoryNode's observation frames are stored as JPEG files in a
per-node subdirectory (named ``frame_0001.jpg``).  ``list()`` accepts
both ``.jpg`` and ``.png`` for backward compatibility.  Paths returned
are relative to the service package root (``services/memory/``).

Per-node limit: ``MAX_IMAGES_PER_NODE`` (default 10).  When the limit
is reached, the oldest image is evicted before saving a new one.

Orphan GC: ``gc_orphans()`` scans ``data/images/`` and removes
subdirectories whose ``node_id`` is no longer in the graph.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Set

log = logging.getLogger("scribe_mem")

_DEFAULT_IMAGE_ROOT = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "images"
)

# Max images per node before oldest eviction.
_MAX_IMAGES_PER_NODE = int(os.environ.get("MAX_IMAGES_PER_NODE", "10"))


class ImageStore:
    """Local file-system store for observation images.

    Layout:
        {image_root}/
          {node_id}/
            frame_0001.png
            frame_0002.png
            ...
    """

    def __init__(self, image_root: str = ""):
        self._root = Path(image_root or _DEFAULT_IMAGE_ROOT)

    @property
    def root(self) -> str:
        return str(self._root)

    # ── Write ──────────────────────────────────────────────────────────

    def save(self, node_id: int, image_bytes: bytes) -> str:
        """Save one image frame for a node.  Returns the relative path.

        When the per-node image limit (``MAX_IMAGES_PER_NODE``) is
        reached, the oldest frame is evicted first.
        """
        node_dir = self._node_dir(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)

        # Enforce per-node limit — evict oldest if at capacity
        existing = _list_image_files(node_dir)
        if len(existing) >= _MAX_IMAGES_PER_NODE:
            oldest = existing[0]  # sorted by name → frame_0001 is oldest
            oldest.unlink(missing_ok=True)
            log.debug("image_store: evicted %s (node %d at limit %d)",
                      oldest.name, node_id, _MAX_IMAGES_PER_NODE)

        # Find next available sequence number (may reuse evicted slot)
        seq = 1
        used = {_extract_seq(f.name) for f in _list_image_files(node_dir)}
        while seq in used:
            seq += 1

        filename = f"frame_{seq:04d}.jpg"
        filepath = node_dir / filename

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        rel = str(filepath.relative_to(self._root.parent.parent))
        log.info("image_store: wrote %s (%.1f KB)", rel, len(image_bytes) / 1024)
        return rel

    def save_batch(self, node_id: int, images: List[bytes]) -> List[str]:
        """Save multiple frames for one node."""
        paths: List[str] = []
        for img in images:
            paths.append(self.save(node_id, img))
        return paths

    # ── Read ───────────────────────────────────────────────────────────

    def list(self, node_id: int) -> List[str]:
        """Return relative paths of all images for a node, sorted by name."""
        node_dir = self._node_dir(node_id)
        files = _list_image_files(node_dir)
        return [str(f.relative_to(self._root.parent.parent)) for f in files]

    def get_node_dir(self, node_id: int) -> str:
        """Return the absolute path to a node's image directory."""
        return str(self._node_dir(node_id))

    def count(self, node_id: int) -> int:
        """Number of images stored for a node."""
        return len(self.list(node_id))

    def remove(self, node_id: int) -> bool:
        """Delete all images for a node.  Returns True if anything was deleted."""
        node_dir = self._node_dir(node_id)
        if not node_dir.exists():
            return False
        import shutil
        shutil.rmtree(str(node_dir), ignore_errors=True)
        log.debug("image_store: removed images for node %d", node_id)
        return True

    # ── GC ─────────────────────────────────────────────────────────────

    def gc_orphans(self, valid_node_ids: Optional[Set[int]] = None) -> int:
        """Remove image directories for nodes that no longer exist.

        If *valid_node_ids* is None, all existing directories are checked
        against the graph store via the caller.

        Returns the count of removed directories.
        """
        if not self._root.exists():
            return 0

        removed = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            try:
                nid = int(child.name)
            except ValueError:
                continue

            if valid_node_ids is not None and nid in valid_node_ids:
                continue  # still valid

            # If no valid_node_ids provided, only clean empty dirs
            if valid_node_ids is None:
                if not any(child.iterdir()):
                    child.rmdir()
                    removed += 1
            else:
                import shutil
                shutil.rmtree(str(child), ignore_errors=True)
                log.info("image_store: gc orphan dir for node %d", nid)
                removed += 1

        if removed:
            log.info("image_store: gc removed %d orphan directories", removed)
        return removed

    # ── Internal ──────────────────────────────────────────────────────

    def _node_dir(self, node_id: int) -> Path:
        return self._root / str(node_id)


# ── module-level helpers ──────────────────────────────────────────────

def _list_image_files(node_dir: Path) -> List[Path]:
    """Return sorted list of image files in *node_dir*."""
    if not node_dir.exists():
        return []
    files = sorted(node_dir.glob("frame_*.jpg")) + sorted(node_dir.glob("frame_*.png"))
    return files


def _extract_seq(filename: str) -> int:
    """Extract the sequence number from 'frame_0005.jpg' → 5."""
    import re
    m = re.search(r"frame_(\d+)", filename)
    return int(m.group(1)) if m else 0
