# SPDX-License-Identifier: MulanPSL-2.0
"""VLM-based asynchronous object recognition for robot patrol imagery.

VLMObserver runs as a background thread that consumes waypoint images
(via a queue), calls a vision-language model to identify objects, and
creates ``object_observation`` MemoryNodes linked to their parent
``path_segment`` nodes.

Usage::

    obs = VLMObserver(
        vlm_api_key="sk-...",
        vlm_base_url="https://api.deepseek.com",
        image_store=image_store,
        memgraph_url="http://127.0.0.1:37798",
    )
    obs.start()

    # Feed waypoint images from PathRecorder
    for wp, img_b64 in waypoints_and_images:
        obs.enqueue(path_node_id, wp, img_b64)

    # Wait for all to complete
    obs.join_empty(timeout=60)
    obs.stop()

Dedup rules:
  - Per-segment: (path_node_id, label) → at most one object_observation.
  - Cross-segment: same label at similar world position → link additional
    parent path_segment via add_edge (multi-parent support).
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("scribe_mem")

# Max VLM API retries on transient errors.
_VLM_MAX_RETRIES = int(os.environ.get("VLM_OBSERVER_MAX_RETRIES", "2"))

# VLM request timeout (seconds).
_VLM_TIMEOUT_S = float(os.environ.get("VLM_OBSERVER_TIMEOUT_S", "45.0"))

# Default VLM model.
_VLM_DEFAULT_MODEL = os.environ.get("MEM_VLM_MODEL", "qwen3.6-flash")

# Spatial proximity threshold for cross-segment dedup (metres).
_CROSS_SEGMENT_DEDUP_RADIUS_M = float(
    os.environ.get("VLM_OBSERVER_DEDUP_RADIUS_M", "2.0")
)

# Default memgraph Scene Hook endpoint.
_MEMGRAPH_HOOK_URL = os.environ.get(
    "MEMGRAPH_HOOK_URL", "http://127.0.0.1:37798",
)

# Sentinel for stopping the worker thread.
_QUEUE_STOP = object()


# ── VLM prompt ─────────────────────────────────────────────────────────

_VLM_SYSTEM_PROMPT = (
    "You are an object recognition system for robot patrol. "
    "Identify ALL objects visible in this image. For each object, output:\n"
    "- label: object category name (e.g., chair, monitor, fire_extinguisher)\n"
    "- description: brief visual description in English\n"
    "- approximate_position: relative to camera (left/center/right, near/mid/far)\n"
    "- confidence: 0.0-1.0\n"
    "Return as a JSON array. If no objects, return []."
)


def _call_vlm(
    image_b64: str,
    api_key: str,
    base_url: str,
    model: str = _VLM_DEFAULT_MODEL,
    max_tokens: int = 512,
) -> Optional[List[Dict[str, Any]]]:
    """Send an image to a VLM and parse the object list response.

    Returns a list of dicts with keys ``label``, ``description``,
    ``approximate_position``, ``confidence``, or ``None`` on failure.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _VLM_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                }},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    for attempt in range(1, _VLM_MAX_RETRIES + 1):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=_VLM_TIMEOUT_S),
            ) as client:
                r = client.post(url, json=payload, headers=headers)
            if r.status_code >= 500:
                log.warning("vlm_observer: VLM returned %d (attempt %d/%d)",
                            r.status_code, attempt, _VLM_MAX_RETRIES)
                if attempt < _VLM_MAX_RETRIES:
                    time.sleep(2.0 * attempt)
                continue
            if r.status_code == 429:
                log.warning("vlm_observer: VLM rate-limited (attempt %d/%d)",
                            attempt, _VLM_MAX_RETRIES)
                if attempt < _VLM_MAX_RETRIES:
                    time.sleep(4.0 * attempt)
                continue
            if r.status_code >= 400:
                log.warning("vlm_observer: VLM returned %d: %s",
                            r.status_code, r.text[:200])
                return None

            data = r.json()
            elapsed = time.monotonic() - t0
            content = data["choices"][0]["message"]["content"].strip()
            log.info("vlm_observer: VLM responded in %.2fs (%d chars): %s",
                     elapsed, len(content), content[:200])

            # Parse JSON
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()

            try:
                result = _json.loads(content)
            except _json.JSONDecodeError:
                # Try to find a JSON array in the text
                import re
                m = re.search(r'\[.*\]', content, re.DOTALL)
                if m:
                    try:
                        result = _json.loads(m.group())
                    except _json.JSONDecodeError:
                        log.warning("vlm_observer: could not parse VLM response: %r",
                                    content[:200])
                        return None
                else:
                    log.warning("vlm_observer: could not parse VLM response: %r",
                                content[:200])
                    return None

            if isinstance(result, list):
                # Validate and normalize each object
                objects: List[Dict[str, Any]] = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "")).strip().lower()
                    if not label:
                        continue
                    objects.append({
                        "label": label,
                        "description": str(item.get("description", "")),
                        "approximate_position": str(
                            item.get("approximate_position", "center-mid")
                        ),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
                return objects

            log.warning("vlm_observer: VLM returned non-array: %r", content[:200])
            return None

        except Exception as e:
            log.warning("vlm_observer: VLM call failed (attempt %d/%d): %s: %s",
                        attempt, _VLM_MAX_RETRIES, type(e).__name__, e)
            if attempt < _VLM_MAX_RETRIES:
                time.sleep(2.0 * attempt)

    return None


# ── Object ID generation ────────────────────────────────────────────────

def _make_obj_id(label: str, image_b64: str) -> str:
    """Generate a stable obj_id from label + image content hash."""
    h = hashlib.sha256(image_b64.encode()).hexdigest()[:8]
    return f"vlm.{label}.{h}"


# ═════════════════════════════════════════════════════════════════════════
# VLMObserver
# ═════════════════════════════════════════════════════════════════════════

class VLMObserver(threading.Thread):
    """Background thread that runs VLM object recognition on queued images.

    Each queued item is a *(path_node_id, waypoint_dict, image_b64)*
    tuple.  The observer calls the VLM, parses the object list, and
    creates / updates ``object_observation`` nodes linked to the parent
    ``path_segment``.
    """

    def __init__(
        self,
        vlm_api_key: str,
        vlm_base_url: str,
        image_store,  # : ImageStore
        *,
        memgraph_url: str = _MEMGRAPH_HOOK_URL,
        vlm_model: str = "",
    ) -> None:
        super().__init__(daemon=True, name="vlm-observer")
        self._api_key = vlm_api_key
        self._base_url = vlm_base_url
        self._model = vlm_model or _VLM_DEFAULT_MODEL
        self._images = image_store
        self._memgraph_url = memgraph_url

        # Queue: (path_node_id, waypoint_dict, image_b64_str) or _QUEUE_STOP
        self._queue: queue.Queue = queue.Queue()

        # Dedup state (guarded by self._lock since accessed from worker only)
        #   _seg_dedup: (path_node_id, label) → node_id   (per-segment)
        #   _global_dedup: label → [(node_id, x, y, path_ids), ...]  (cross-segment)
        self._seg_dedup: Dict[Tuple[int, str], int] = {}
        self._global_dedup: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

        # Stats
        self._processed: int = 0
        self._objects_found: int = 0
        self._nodes_created: int = 0
        self._nodes_updated: int = 0

    # ── public API ──────────────────────────────────────────────────────

    def enqueue(
        self, path_node_id: int, waypoint: Dict[str, Any], image_b64: str,
    ) -> None:
        """Queue a waypoint image for VLM recognition.

        Parameters:
            path_node_id: The ``path_segment`` node this waypoint belongs to.
            waypoint: Dict with keys ``x``, ``y``, ``z``, ``yaw``, ``ts``.
            image_b64: Base64-encoded JPEG image data.
        """
        self._queue.put((path_node_id, waypoint, image_b64))

    def stop(self, timeout: Optional[float] = 10.0) -> None:
        """Signal the worker to stop and wait for it to finish.

        Remaining queued items are drained (not processed).
        """
        self._queue.put(_QUEUE_STOP)
        self.join(timeout=timeout)

    def join_empty(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue is empty (all queued items processed).

        Returns ``True`` if the queue drained, ``False`` on timeout.
        """
        deadline = (time.monotonic() + timeout) if timeout else None
        while not self._queue.empty():
            if deadline and time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        # Also wait a beat for in-flight VLM call to finish
        if deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
        return True

    # ── properties ──────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        """Return processing statistics."""
        return {
            "processed": self._processed,
            "objects_found": self._objects_found,
            "nodes_created": self._nodes_created,
            "nodes_updated": self._nodes_updated,
        }

    # ── Thread run ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Main worker loop — process items from the queue."""
        log.info("vlm_observer: worker started (model=%s)", self._model)

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _QUEUE_STOP:
                break

            path_node_id, waypoint, image_b64 = item
            try:
                self._process(path_node_id, waypoint, image_b64)
            except Exception:
                log.warning("vlm_observer: unhandled error processing waypoint",
                            exc_info=True)
            finally:
                self._queue.task_done()

        # Drain remaining items without processing
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        log.info("vlm_observer: worker stopped (processed=%d, objects=%d, "
                 "created=%d, updated=%d)",
                 self._processed, self._objects_found,
                 self._nodes_created, self._nodes_updated)

    # ── processing ──────────────────────────────────────────────────────

    def _process(
        self, path_node_id: int, waypoint: Dict[str, Any], image_b64: str,
    ) -> None:
        """Process one waypoint: VLM call → dedup → create/update node."""
        # 1. Call VLM
        objects = _call_vlm(
            image_b64, self._api_key, self._base_url, self._model,
        )
        self._processed += 1

        if objects is None:
            log.info("vlm_observer: VLM returned no parseable result for "
                     "path_node=%d", path_node_id)
            return

        if not objects:
            log.debug("vlm_observer: VLM found no objects at waypoint "
                      "(path_node=%d)", path_node_id)
            return

        wx, wy = waypoint.get("x", 0.0), waypoint.get("y", 0.0)

        # 2. For each recognized object → create or update
        for obj in objects:
            label = obj["label"]
            self._objects_found += 1

            with self._lock:
                seg_key = (path_node_id, label)

                # ── Per-segment dedup: already have this label? ──
                existing_nid = self._seg_dedup.get(seg_key)
                if existing_nid is not None:
                    # Update existing node: append this image_ref
                    self._append_image_ref(existing_nid, path_node_id, waypoint, image_b64)
                    self._nodes_updated += 1
                    continue

                # ── Cross-segment dedup: same label at similar position? ──
                linked_nid = self._find_cross_segment_match(
                    label, wx, wy, path_node_id,
                )
                if linked_nid is not None:
                    self._seg_dedup[seg_key] = linked_nid
                    self._nodes_updated += 1
                    continue

                # ── Create new object_observation node ──
                node_id = self._create_node(
                    path_node_id, waypoint, obj, image_b64,
                )
                if node_id is not None and node_id > 0:
                    self._seg_dedup[seg_key] = node_id
                    self._global_dedup.setdefault(label, []).append({
                        "node_id": node_id,
                        "x": wx,
                        "y": wy,
                        "path_ids": [path_node_id],
                    })
                    self._nodes_created += 1

    # ── cross-segment dedup ──────────────────────────────────────────────

    def _find_cross_segment_match(
        self, label: str, wx: float, wy: float, path_node_id: int,
    ) -> Optional[int]:
        """Check if *label* was observed at a nearby position in another segment.

        Returns the existing node_id if within dedup radius, else None.
        If matched, adds *path_node_id* as an additional parent via add_edge.
        """
        entries = self._global_dedup.get(label)
        if not entries:
            return None

        r2 = _CROSS_SEGMENT_DEDUP_RADIUS_M ** 2
        for entry in entries:
            ex, ey = entry["x"], entry["y"]
            if (wx - ex) ** 2 + (wy - ey) ** 2 <= r2:
                nid = entry["node_id"]
                if path_node_id not in entry["path_ids"]:
                    entry["path_ids"].append(path_node_id)
                    self._link_parent(nid, path_node_id)
                return nid
        return None

    # ── node creation / update ───────────────────────────────────────────

    def _create_node(
        self,
        path_node_id: int,
        waypoint: Dict[str, Any],
        obj: Dict[str, Any],
        image_b64: str,
    ) -> Optional[int]:
        """POST a new object_observation node to memgraph.

        The node is linked to *path_node_id* via parent_node_id.
        """
        label = obj["label"]
        conf = obj["confidence"]
        pos = obj.get("approximate_position", "center-mid")
        desc = obj.get("description", "")

        # Build summary
        summary_parts = [f"VLM observed {label} ({conf:.2f}) at {pos}"]
        if desc:
            summary_parts.append(f"— {desc}")
        summary = " ".join(summary_parts)

        obj_id = _make_obj_id(label, image_b64)
        now_ns = time.time_ns()

        payload: Dict[str, Any] = {
            "session_id": "vlm-observer",
            "plan_id": "vlm-observer",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "vlm_observer",
                "msg": summary,
            },
            "spatial": {
                "origin": "world",
                "objects": [{
                    "obj_id": obj_id,
                    "label": label,
                    "x": waypoint.get("x", 0.0),
                    "y": waypoint.get("y", 0.0),
                    "z": waypoint.get("z", 0.0),
                }],
            },
            "image_base64": image_b64,
            "parent_node_id": path_node_id,
            "kv": {
                "node_type": "object_observation",
                "vlm_confidence": conf,
                "vlm_position": pos,
                "vlm_description": desc,
                "waypoint_ts": waypoint.get("ts", 0),
            },
        }

        return self._post(payload)

    def _append_image_ref(
        self, node_id: int, path_node_id: int, waypoint: Dict[str, Any],
        image_b64: str,
    ) -> None:
        """Add another image reference to an existing object_observation.

        Sends a lightweight POST with parent_node_id and the new image
        so the handler appends the image to the existing node.
        """
        now_ns = time.time_ns()
        payload: Dict[str, Any] = {
            "session_id": "vlm-observer",
            "plan_id": "vlm-observer",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "vlm_observer",
                "msg": "VLM re-observed object from another waypoint",
            },
            "spatial": {
                "origin": "world",
                "objects": [],
            },
            "image_base64": image_b64,
            "parent_node_id": node_id,
            "kv": {
                "node_type": "object_observation",
                "append_image_ref": True,
                "waypoint_ts": waypoint.get("ts", 0),
            },
        }
        self._post(payload)

    def _link_parent(self, node_id: int, new_parent_id: int) -> None:
        """Add an additional parent to an existing object_observation node.

        Uses the Scene Hook with a special ``link_parent`` kv flag.
        The pipeline adds an edge from *new_parent_id* to *node_id*
        without creating a new node.
        """
        now_ns = time.time_ns()
        payload: Dict[str, Any] = {
            "session_id": "vlm-observer",
            "plan_id": "vlm-observer",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "vlm_observer",
                "msg": "link additional parent for cross-segment match",
            },
            "spatial": {"origin": "world", "objects": []},
            "parent_node_id": new_parent_id,  # the new path_segment parent
            "kv": {
                "node_type": "object_observation",
                "link_to_existing": node_id,  # the existing node to link to
            },
        }
        self._post(payload)

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _post(self, payload: Dict[str, Any]) -> Optional[int]:
        """POST *payload* to memgraph.  Returns node_id or None."""
        import httpx

        try:
            body = _json.dumps(payload).encode()
            with httpx.Client(
                timeout=httpx.Timeout(connect=5.0, read=15.0),
            ) as client:
                r = client.post(
                    self._memgraph_url.rstrip("/"),
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
            if r.status_code >= 400:
                log.warning("vlm_observer: memgraph returned %d: %s",
                            r.status_code, r.text[:200])
                return None
            resp = r.json()
            nid = resp.get("node_id")
            return int(nid) if nid is not None else None
        except Exception as e:
            log.warning("vlm_observer: POST failed: %s: %s",
                        type(e).__name__, e)
            return None


# ═════════════════════════════════════════════════════════════════════════
# VLMFeatureExtractor — Tier 3 triggered VLM enrichment of memory nodes
# ═════════════════════════════════════════════════════════════════════════

_VLM_EXTRACTOR_PROMPT = (
    "You are analyzing images from a robot's memory to answer a specific question.\n\n"
    "Question: {user_query}\n\n"
    "Look at the image(s) carefully and describe:\n"
    "1. What objects are visible? (list with descriptions)\n"
    "2. What are the spatial relationships between objects?\n"
    "3. What state/condition are the objects in?\n"
    "4. Is there anything unusual or notable?\n\n"
    "Focus especially on information that would help answer: \"{user_query}\"\n\n"
    "Return your analysis as a concise JSON:\n"
    '{{\n'
    '  "objects_found": ["list of object labels"],\n'
    '  "key_observations": "detailed description",\n'
    '  "can_answer_query": true/false,\n'
    '  "answer_to_query": "if can_answer, otherwise empty"\n'
    '}}'
)


class VLMFeatureExtractor:
    """Tier-3 VLM enrichment: extract visual features from node images.

    Triggered by the retrieve pipeline when Tier 1+2 cannot answer a query.
    Reads images from node ``image_refs``, calls VLM to extract visual
    features, and updates the node's summary + tags + text index so
    subsequent retrievals can find the information.
    """

    def __init__(
        self,
        vlm_api_key: str,
        vlm_base_url: str,
        *,
        vlm_model: str = "",
        graph_store=None,
        text_index=None,
    ) -> None:
        self._api_key = vlm_api_key
        self._base_url = vlm_base_url
        self._model = vlm_model or _VLM_DEFAULT_MODEL
        self._graph = graph_store
        self._text_index = text_index

    async def extract_and_update(
        self,
        node_id: int,
        query: str,
        image_paths: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract visual features from node images and update the node.

        Returns the parsed VLM response dict, or None on failure.
        """
        if not self._graph:
            log.warning("vlm_extractor: no graph_store — cannot update node")
            return None

        node = self._graph.get_node(node_id)
        if node is None:
            log.warning("vlm_extractor: node %d not found", node_id)
            return None

        # Read images from disk
        images_b64: List[str] = []
        for path in image_paths[:3]:  # max 3 images
            img_b64 = self._read_image_b64(path)
            if img_b64:
                images_b64.append(img_b64)

        if not images_b64:
            log.info("vlm_extractor: no readable images for node %d", node_id)
            return None

        # Call VLM — use first image + prompt
        result = self._call_extractor(query, images_b64[0])
        if result is None:
            return None

        # Update node
        self._apply_enrichment(node, result)

        # Update text index
        if self._text_index is not None:
            try:
                self._text_index.update(node_id, node)
                log.debug("vlm_extractor: text_index updated for node %d", node_id)
            except Exception as e:
                log.debug("vlm_extractor: text_index update failed: %s", e)

        log.info(
            "vlm_extractor: node %d enriched — objects=%s, can_answer=%s",
            node_id, result.get("objects_found", []),
            result.get("can_answer_query", False),
        )
        return result

    # ── internal ───────────────────────────────────────────────────────

    @staticmethod
    def _read_image_b64(path: str) -> Optional[str]:
        """Read an image file from disk and return as base64 string."""
        import base64 as _b64
        from pathlib import Path

        try:
            p = Path(path)
            if not p.is_absolute():
                # Resolve relative to data/images
                data_dir = os.environ.get(
                    "AGENT_MEMORY_DIR",
                    str(Path(__file__).resolve().parent.parent / "memory"),
                )
                p = Path(data_dir) / ".." / "data" / "images" / path
                p = p.resolve()
            if not p.exists():
                log.debug("vlm_extractor: image not found: %s", path)
                return None
            data = p.read_bytes()
            return _b64.b64encode(data).decode("ascii")
        except OSError as e:
            log.debug("vlm_extractor: read failed for %s: %s", path, e)
            return None

    def _call_extractor(
        self, query: str, image_b64: str,
    ) -> Optional[Dict[str, Any]]:
        """Call VLM with the extraction prompt for one image.

        Returns parsed JSON dict or None.
        """
        import httpx

        prompt = _VLM_EXTRACTOR_PROMPT.format(user_query=query)
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    }},
                ],
            }],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        try:
            with httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=_VLM_TIMEOUT_S),
            ) as client:
                r = client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                log.warning("vlm_extractor: VLM returned %d: %s",
                            r.status_code, r.text[:200])
                return None
            content = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning("vlm_extractor: VLM call failed: %s", e)
            return None

        # Parse JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()

        try:
            result = _json.loads(content)
            if isinstance(result, dict):
                return result
        except _json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                try:
                    result = _json.loads(m.group())
                    if isinstance(result, dict):
                        return result
                except _json.JSONDecodeError:
                    pass

        log.warning("vlm_extractor: could not parse VLM response: %r",
                    content[:200])
        return None

    @staticmethod
    def _apply_enrichment(node, features: Dict[str, Any]) -> None:
        """Update a MemoryNode in-place with extracted VLM features.

        Modifies ``node.summary`` and ``node.tags.objects_present``.
        Caller must persist via ``GraphStore.update_node()``.
        """
        key_obs = features.get("key_observations", "")
        objects = features.get("objects_found", [])

        if key_obs:
            # Append VLM observations to summary
            enrichment = f" | VLM observations: {key_obs}"
            if enrichment not in (node.summary or ""):
                node.summary = (node.summary or "") + enrichment

        if objects and node.tags:
            existing = set(node.tags.objects_present or [])
            for obj in objects:
                obj_lower = str(obj).strip().lower()
                if obj_lower and obj_lower not in existing:
                    existing.add(obj_lower)
                    node.tags.objects_present.append(obj_lower)

        # Increase weight — enriched nodes are more valuable
        node.weight = min(1.0, node.weight + 0.1)
        node.version += 1
