# SPDX-License-Identifier: MulanPSL-2.0
"""Noise-filter regressions for metric Scene perception."""

import numpy as np

from scene_service.ingest.perception_concept_graphs import (
    _IGNORED_CLASSES,
    _is_floor_noise,
)
from scene_service.ingest.perception_vlm import _canon_class


def test_picture_frame_alias_is_ignored():
    """Spaced detector labels canonicalize to the filtered noise class."""
    assert _canon_class("picture frame") in _IGNORED_CLASSES
    assert _canon_class("picture_frame") in _IGNORED_CLASSES


def test_furniture_cloud_with_floor_mean_is_rejected():
    """One high leaked point cannot rescue a floor-dominated table mask."""
    points = np.array([
        [0.0, 0.0, -0.10],
        [0.1, 0.0, -0.09],
        [0.0, 0.1, -0.08],
        [0.1, 0.1, 0.40],
    ])
    assert _is_floor_noise("table", points) is True


def test_elevated_furniture_cloud_is_kept():
    """A plausible table-height cloud survives the floor-noise gate."""
    points = np.array([
        [0.0, 0.0, 0.65],
        [0.1, 0.0, 0.68],
        [0.0, 0.1, 0.72],
        [0.1, 0.1, 0.75],
    ])
    assert _is_floor_noise("table", points) is False
