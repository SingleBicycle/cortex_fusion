"""Feature schemas for cortical surface graph inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


FEATURE_SCHEMAS: Dict[str, Dict[str, object]] = {
    "baseline8": {
        "feature_names": [
            "pial_x",
            "pial_y",
            "pial_z",
            "white_x",
            "white_y",
            "white_z",
            "thickness",
            "curvature",
        ],
        "in_dim": 8,
        "geo_dim": 6,
        "morph_dim": 2,
        "default_recon_weights": [1.0] * 8,
    },
    "main5": {
        "feature_names": [
            "mid_x",
            "mid_y",
            "mid_z",
            "thickness",
            "curvature",
        ],
        "in_dim": 5,
        "geo_dim": 3,
        "morph_dim": 2,
        "default_recon_weights": [1.0] * 5,
    },
    "ablation2": {
        "feature_names": [
            "thickness",
            "curvature",
        ],
        "in_dim": 2,
        "geo_dim": 0,
        "morph_dim": 2,
        "default_recon_weights": [1.0, 1.0],
    },
}


def list_input_modes() -> List[str]:
    return sorted(FEATURE_SCHEMAS.keys())


def get_feature_schema(input_mode: str) -> Dict[str, object]:
    key = str(input_mode).lower()
    if key not in FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported input_mode={input_mode!r}. Choices: {list_input_modes()}")
    return deepcopy(FEATURE_SCHEMAS[key])


__all__ = [
    "FEATURE_SCHEMAS",
    "get_feature_schema",
    "list_input_modes",
]
