"""Fixed-template mesh hierarchy utilities."""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple

import torch


HIERARCHY_FILE_RE = re.compile(
    r"^(?P<hemi>lh|rh)_(?P<fine>fsaverage[0-9]+)_to_(?P<coarse>fsaverage[0-9]+)_parent\.pt$"
)


def pool_features_by_parent(
    x: torch.Tensor,
    parent_index: torch.Tensor,
    num_parents: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if x.dim() != 2:
        raise ValueError(f"Expected x to have shape [N, C], got {tuple(x.shape)}")
    parent_index = parent_index.long().view(-1)
    if x.shape[0] != int(parent_index.shape[0]):
        raise ValueError(
            f"Parent map length must match vertex count, got x={x.shape[0]} parent={parent_index.shape[0]}"
        )
    if num_parents <= 0:
        raise ValueError(f"num_parents must be positive, got {num_parents}")

    device = x.device
    num_parents = int(num_parents)
    out = x.new_zeros((num_parents, x.shape[1]))

    if valid_mask is None:
        weights = torch.ones(x.shape[0], device=device, dtype=x.dtype)
        weighted_x = x
    else:
        weights = valid_mask.to(device=device, dtype=x.dtype).view(-1)
        weighted_x = x * weights.unsqueeze(-1)

    out.index_add_(0, parent_index.to(device=device), weighted_x)
    counts = x.new_zeros((num_parents,))
    counts.index_add_(0, parent_index.to(device=device), weights)
    out = out / counts.clamp_min(1.0).unsqueeze(-1)

    pooled_valid = None if valid_mask is None else counts > 0
    return out, pooled_valid


def unpool_features_by_parent(x: torch.Tensor, parent_index: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected x to have shape [N, C], got {tuple(x.shape)}")
    parent_index = parent_index.long().view(-1).to(device=x.device)
    return x[parent_index]


def load_hierarchy_definitions(
    hierarchy_dir: str | None,
    edge_cache_dir: str | None,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not hierarchy_dir:
        return {}
    if not edge_cache_dir:
        raise ValueError("edge_cache_dir is required when hierarchy_dir is provided.")

    hierarchy_dir = os.path.abspath(hierarchy_dir)
    edge_cache_dir = os.path.abspath(edge_cache_dir)
    if not os.path.isdir(hierarchy_dir):
        raise FileNotFoundError(f"Hierarchy directory not found: {hierarchy_dir}")

    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for fname in sorted(os.listdir(hierarchy_dir)):
        match = HIERARCHY_FILE_RE.match(fname)
        if match is None:
            continue
        payload = torch.load(os.path.join(hierarchy_dir, fname), map_location="cpu")
        if isinstance(payload, torch.Tensor):
            parent_index = payload.long().contiguous()
            coarse_nodes = int(parent_index.max().item()) + 1 if parent_index.numel() > 0 else 0
            coarse_edge_index = None
        elif isinstance(payload, dict):
            parent_index = payload["parent_index"].long().contiguous()
            coarse_nodes = int(payload.get("coarse_nodes", int(parent_index.max().item()) + 1 if parent_index.numel() > 0 else 0))
            coarse_edge_index = payload.get("coarse_edge_index", None)
            if coarse_edge_index is not None:
                coarse_edge_index = coarse_edge_index.long().contiguous()
        else:
            raise TypeError(f"Unsupported hierarchy payload type in {fname}: {type(payload)}")

        hemi = str(match.group("hemi"))
        fine_res = str(match.group("fine"))
        coarse_res = str(match.group("coarse"))
        if coarse_edge_index is None:
            edge_path = os.path.join(edge_cache_dir, f"{coarse_res}_edge_index.pt")
            if not os.path.exists(edge_path):
                raise FileNotFoundError(f"Missing coarse edge cache for hierarchy {fname}: {edge_path}")
            coarse_edge_index = torch.load(edge_path, map_location="cpu").long().contiguous()
        out[(hemi, fine_res)] = {
            "fine_res": fine_res,
            "coarse_res": coarse_res,
            "parent_index": parent_index,
            "coarse_edge_index": coarse_edge_index,
            "num_coarse_nodes": coarse_nodes,
        }
    return out


__all__ = [
    "pool_features_by_parent",
    "unpool_features_by_parent",
    "load_hierarchy_definitions",
]
