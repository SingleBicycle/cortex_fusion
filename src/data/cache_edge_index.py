"""Cache template edge_index per resolution from manifest faces."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.data.io_fs import read_surface


def faces_to_undirected_edge_index(faces: np.ndarray) -> torch.Tensor:
    """Convert triangular faces (F,3) to undirected unique edge_index (2,E)."""
    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {tri.shape}")

    e01 = tri[:, [0, 1]]
    e12 = tri[:, [1, 2]]
    e20 = tri[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0)

    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    rev = edges[:, ::-1]
    edges_ud = np.concatenate([edges, rev], axis=0)
    edge_index = torch.from_numpy(edges_ud.T).long().contiguous()
    return edge_index


def cache_edge_index(manifest_csv: str, res: str, out_dir: str = "cache/templates") -> str:
    df = pd.read_csv(manifest_csv)
    if df.empty:
        raise RuntimeError("manifest is empty")

    sub = df[df["res"] == res]
    if sub.empty:
        raise RuntimeError(f"no rows with res={res} in manifest: {manifest_csv}")

    row = sub.iloc[0]
    topology_path = str(row["topology_path"]) if "topology_path" in row and str(row["topology_path"]) else ""
    source_path = topology_path or str(row["pial_path"])
    _verts, faces = read_surface(source_path)
    if faces.size == 0:
        raise RuntimeError(f"topology source has no faces: {source_path}")
    edge_index = faces_to_undirected_edge_index(faces)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{res}_edge_index.pt")
    torch.save(edge_index, out_path)

    print(f"res={res}")
    print(f"source_topology={source_path}")
    print(f"edge_index_shape={tuple(edge_index.shape)}")
    print(f"output={out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache template edge_index by resolution")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--res", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="cache/templates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_edge_index(manifest_csv=args.manifest, res=args.res, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
