"""Cache fixed template parent maps between mesh resolutions."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import torch

from src.data.io_fs import read_surface


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return centered / norms


def _nearest_parent_index(
    fine_xyz: np.ndarray,
    coarse_xyz: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    fine_unit = _normalize_xyz(fine_xyz)
    coarse_unit = _normalize_xyz(coarse_xyz)

    parent = np.empty((fine_unit.shape[0],), dtype=np.int64)
    coarse_sq = np.sum(coarse_unit * coarse_unit, axis=1, dtype=np.float32)
    for start in range(0, fine_unit.shape[0], chunk_size):
        end = min(start + chunk_size, fine_unit.shape[0])
        chunk = fine_unit[start:end]
        chunk_sq = np.sum(chunk * chunk, axis=1, dtype=np.float32, keepdims=True)
        dist = chunk_sq + coarse_sq[None, :] - (2.0 * np.matmul(chunk, coarse_unit.T))
        parent[start:end] = np.argmin(dist, axis=1)
    return parent


def _iter_pairs(pairs: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for item in pairs:
        fine, coarse = item.split(":")
        fine = fine.strip()
        coarse = coarse.strip()
        if not fine or not coarse:
            raise ValueError(f"Invalid resolution pair: {item!r}")
        yield fine, coarse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache template parent maps for mesh hierarchy pooling.")
    parser.add_argument("--faces_root", type=str, required=True, help="Root containing faces/<res>/surf/<hemi>.pial")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=["fsaverage6:fsaverage5", "fsaverage5:fsaverage4"],
        help="Fine-to-coarse resolution pairs",
    )
    parser.add_argument("--chunk_size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    faces_root = os.path.abspath(args.faces_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for fine_res, coarse_res in _iter_pairs(args.pairs):
        for hemi in ("lh", "rh"):
            fine_path = os.path.join(faces_root, fine_res, "surf", f"{hemi}.pial")
            coarse_path = os.path.join(faces_root, coarse_res, "surf", f"{hemi}.pial")
            if not os.path.exists(fine_path):
                raise FileNotFoundError(f"Missing fine template surface: {fine_path}")
            if not os.path.exists(coarse_path):
                raise FileNotFoundError(f"Missing coarse template surface: {coarse_path}")

            fine_xyz, _ = read_surface(fine_path)
            coarse_xyz, _ = read_surface(coarse_path)
            parent_index = _nearest_parent_index(
                fine_xyz=fine_xyz,
                coarse_xyz=coarse_xyz,
                chunk_size=int(args.chunk_size),
            )
            counts = np.bincount(parent_index, minlength=int(coarse_xyz.shape[0]))
            payload = {
                "hemi": hemi,
                "fine_res": fine_res,
                "coarse_res": coarse_res,
                "fine_nodes": int(fine_xyz.shape[0]),
                "coarse_nodes": int(coarse_xyz.shape[0]),
                "parent_index": torch.from_numpy(parent_index).long(),
                "min_children": int(counts.min()) if counts.size else 0,
                "max_children": int(counts.max()) if counts.size else 0,
                "mean_children": float(counts.mean()) if counts.size else 0.0,
            }
            out_path = os.path.join(out_dir, f"{hemi}_{fine_res}_to_{coarse_res}_parent.pt")
            torch.save(payload, out_path)
            print(
                f"saved={out_path} fine_nodes={payload['fine_nodes']} coarse_nodes={payload['coarse_nodes']} "
                f"children[min/mean/max]={payload['min_children']}/{payload['mean_children']:.2f}/{payload['max_children']}"
            )


if __name__ == "__main__":
    main()
