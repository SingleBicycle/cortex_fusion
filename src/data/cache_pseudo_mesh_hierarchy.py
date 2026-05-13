"""Build a coarse mesh hierarchy from a single fine template surface."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.data.io_fs import read_surface


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return centered / norms


def _farthest_point_sample(xyz: np.ndarray, num_samples: int) -> np.ndarray:
    if num_samples <= 0 or num_samples > int(xyz.shape[0]):
        raise ValueError(f"num_samples must be in [1, {xyz.shape[0]}], got {num_samples}")
    xyz = _normalize_xyz(xyz)
    seed = int(np.argmax(np.linalg.norm(xyz - xyz.mean(axis=0, keepdims=True), axis=1)))
    selected = np.empty((num_samples,), dtype=np.int64)
    selected[0] = seed

    min_dist = np.sum((xyz - xyz[seed : seed + 1]) ** 2, axis=1)
    for idx in range(1, num_samples):
        next_idx = int(np.argmax(min_dist))
        selected[idx] = next_idx
        dist = np.sum((xyz - xyz[next_idx : next_idx + 1]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return selected


def _nearest_parent_index(fine_xyz: np.ndarray, coarse_xyz: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    fine_unit = _normalize_xyz(fine_xyz)
    coarse_unit = _normalize_xyz(coarse_xyz)
    coarse_sq = np.sum(coarse_unit * coarse_unit, axis=1, dtype=np.float32)
    parent = np.empty((fine_unit.shape[0],), dtype=np.int64)
    for start in range(0, fine_unit.shape[0], chunk_size):
        end = min(start + chunk_size, fine_unit.shape[0])
        chunk = fine_unit[start:end]
        chunk_sq = np.sum(chunk * chunk, axis=1, dtype=np.float32, keepdims=True)
        dist = chunk_sq + coarse_sq[None, :] - 2.0 * np.matmul(chunk, coarse_unit.T)
        parent[start:end] = np.argmin(dist, axis=1)
    return parent


def _collapse_edge_index(fine_edge_index: torch.Tensor, parent_index: np.ndarray, num_parents: int) -> torch.Tensor:
    edges = fine_edge_index.detach().cpu().numpy().T
    coarse_edges = parent_index[edges]
    coarse_edges = coarse_edges[coarse_edges[:, 0] != coarse_edges[:, 1]]
    if coarse_edges.size == 0:
        raise RuntimeError("Collapsed hierarchy produced no inter-parent edges.")

    coarse_edges = np.sort(coarse_edges, axis=1)
    coarse_edges = np.unique(coarse_edges, axis=0)

    used = np.unique(coarse_edges.reshape(-1))
    if used.size != int(num_parents):
        missing = sorted(set(range(int(num_parents))) - set(int(x) for x in used))
        raise RuntimeError(f"Collapsed coarse graph has isolated parents: {missing[:10]}")

    rev = coarse_edges[:, ::-1]
    edge_index = torch.from_numpy(np.concatenate([coarse_edges, rev], axis=0).T).long().contiguous()
    return edge_index


def _load_topology_paths(manifest_csv: str, res: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_csv)
    out: Dict[str, str] = {}
    for hemi in ("lh", "rh"):
        sub = df[(df["hemi"].astype(str) == hemi) & (df["res"].astype(str) == res)]
        if sub.empty:
            raise RuntimeError(f"No manifest row found for hemi={hemi} res={res}")
        row = sub.iloc[0]
        out[hemi] = str(row.get("topology_path") or row.get("pial_path"))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a pseudo coarse mesh hierarchy from a fine template.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--fine_res", type=str, default="fsaverage4")
    parser.add_argument("--coarse_res", type=str, default="fsaverage3")
    parser.add_argument("--coarse_nodes", type=int, default=642)
    parser.add_argument("--fine_edge_cache_dir", type=str, required=True)
    parser.add_argument("--out_hierarchy_dir", type=str, required=True)
    parser.add_argument("--out_edge_cache_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = os.path.abspath(args.manifest)
    fine_edge_cache_dir = os.path.abspath(args.fine_edge_cache_dir)
    out_hierarchy_dir = os.path.abspath(args.out_hierarchy_dir)
    out_edge_cache_dir = os.path.abspath(args.out_edge_cache_dir)
    os.makedirs(out_hierarchy_dir, exist_ok=True)
    os.makedirs(out_edge_cache_dir, exist_ok=True)

    fine_edge_path = os.path.join(fine_edge_cache_dir, f"{args.fine_res}_edge_index.pt")
    if not os.path.exists(fine_edge_path):
        raise FileNotFoundError(f"Missing fine edge cache: {fine_edge_path}")
    fine_edge_index = torch.load(fine_edge_path, map_location="cpu").long().contiguous()
    shutil.copy2(fine_edge_path, os.path.join(out_edge_cache_dir, f"{args.fine_res}_edge_index.pt"))

    topology_paths = _load_topology_paths(manifest_csv=manifest, res=args.fine_res)
    coarse_edge_index_saved = False
    shared_seed_vertices = None

    for hemi in ("lh", "rh"):
        verts, faces = read_surface(topology_paths[hemi])
        if faces.size == 0:
            raise RuntimeError(f"Topology source has no faces: {topology_paths[hemi]}")

        if shared_seed_vertices is None:
            shared_seed_vertices = _farthest_point_sample(verts, num_samples=int(args.coarse_nodes))
        seed_vertices = np.asarray(shared_seed_vertices, dtype=np.int64)
        coarse_xyz = verts[seed_vertices]
        parent_index = _nearest_parent_index(
            fine_xyz=verts,
            coarse_xyz=coarse_xyz,
            chunk_size=int(args.chunk_size),
        )
        parent_index[seed_vertices] = np.arange(seed_vertices.shape[0], dtype=np.int64)
        counts = np.bincount(parent_index, minlength=int(args.coarse_nodes))
        if counts.min() <= 0:
            raise RuntimeError(
                f"Pseudo hierarchy produced empty parents for hemi={hemi}; min_children={int(counts.min())}"
            )

        coarse_edge_index = _collapse_edge_index(
            fine_edge_index=fine_edge_index,
            parent_index=parent_index,
            num_parents=int(args.coarse_nodes),
        )
        payload = {
            "hemi": hemi,
            "fine_res": str(args.fine_res),
            "coarse_res": str(args.coarse_res),
            "fine_nodes": int(verts.shape[0]),
            "coarse_nodes": int(args.coarse_nodes),
            "seed_vertices": torch.from_numpy(seed_vertices).long(),
            "parent_index": torch.from_numpy(parent_index).long(),
            "coarse_edge_index": coarse_edge_index,
            "min_children": int(counts.min()),
            "max_children": int(counts.max()),
            "mean_children": float(counts.mean()),
            "builder": "fps_edge_collapse",
            "topology_path": topology_paths[hemi],
        }
        out_parent_path = os.path.join(out_hierarchy_dir, f"{hemi}_{args.fine_res}_to_{args.coarse_res}_parent.pt")
        torch.save(payload, out_parent_path)
        print(
            f"saved={out_parent_path} fine_nodes={payload['fine_nodes']} coarse_nodes={payload['coarse_nodes']} "
            f"children[min/mean/max]={payload['min_children']}/{payload['mean_children']:.2f}/{payload['max_children']}"
        )

        if not coarse_edge_index_saved:
            coarse_edge_path = os.path.join(out_edge_cache_dir, f"{args.coarse_res}_edge_index.pt")
            torch.save(coarse_edge_index, coarse_edge_path)
            print(f"saved coarse_edge_index={coarse_edge_path} shape={tuple(coarse_edge_index.shape)}")
            coarse_edge_index_saved = True


if __name__ == "__main__":
    main()
