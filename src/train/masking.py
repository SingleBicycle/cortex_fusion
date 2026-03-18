"""Masking utilities for surface-graph SSL."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Optional, Sequence

import torch


def _target_mask_count(n_verts: int, ratio: float) -> int:
    if n_verts <= 0:
        raise ValueError(f"n_verts must be positive, got {n_verts}")
    if ratio <= 0.0:
        return 1
    if ratio >= 1.0:
        return n_verts
    return max(1, min(n_verts, int(math.ceil(float(ratio) * float(n_verts)))))


def build_neighbor_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    edge_index_cpu = edge_index.detach().cpu()
    if edge_index_cpu.dim() != 2 or edge_index_cpu.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index_cpu.shape)}")

    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return [sorted(adj) for adj in neighbors]


def random_vertex_mask(
    n_verts: int,
    ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    target = _target_mask_count(n_verts=n_verts, ratio=ratio)
    perm = torch.randperm(n_verts, generator=generator)
    mask = torch.zeros(n_verts, dtype=torch.bool)
    mask[perm[:target]] = True
    return mask.to(device=device)


def patch_vertex_mask(
    n_verts: int,
    ratio: float,
    neighbors: Sequence[Sequence[int]],
    device: torch.device,
    patch_hops: int = 2,
    patch_num_seeds: int = 16,
) -> torch.Tensor:
    target = _target_mask_count(n_verts=n_verts, ratio=ratio)
    patch_hops = max(0, int(patch_hops))
    patch_num_seeds = max(1, min(int(patch_num_seeds), n_verts))

    perm = torch.randperm(n_verts).tolist()
    seed_candidates = [int(idx) for idx in perm[:patch_num_seeds]]
    seed_queue = deque(seed_candidates)
    visited = set()
    masked = set()
    queue = deque((seed, 0) for seed in seed_candidates)

    while len(masked) < target:
        if not queue:
            if not seed_queue:
                remaining = [idx for idx in perm if idx not in masked]
                if not remaining:
                    break
                seed_queue.extend(remaining)
            next_seed = seed_queue.popleft()
            if next_seed in visited:
                continue
            queue.append((next_seed, 0))

        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        masked.add(node)
        if len(masked) >= target or depth >= patch_hops:
            continue

        nbrs = list(neighbors[node])
        random.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in visited:
                queue.append((int(nbr), depth + 1))

    if len(masked) < target:
        remaining = [idx for idx in perm if idx not in masked]
        masked.update(int(idx) for idx in remaining[: target - len(masked)])

    mask = torch.zeros(n_verts, dtype=torch.bool)
    if masked:
        mask[list(masked)] = True
    return mask.to(device=device)


def hybrid_vertex_mask(
    n_verts: int,
    ratio: float,
    neighbors: Sequence[Sequence[int]],
    device: torch.device,
    patch_hops: int = 2,
    patch_num_seeds: int = 16,
    patch_fraction: float = 0.7,
) -> torch.Tensor:
    target = _target_mask_count(n_verts=n_verts, ratio=ratio)
    patch_count = max(1, min(target, int(round(target * patch_fraction))))

    patch_mask = patch_vertex_mask(
        n_verts=n_verts,
        ratio=float(patch_count) / float(n_verts),
        neighbors=neighbors,
        device=device,
        patch_hops=patch_hops,
        patch_num_seeds=patch_num_seeds,
    )

    remaining = target - int(patch_mask.sum().item())
    if remaining <= 0:
        return patch_mask

    available = (~patch_mask).nonzero(as_tuple=False).view(-1)
    if available.numel() == 0:
        return patch_mask

    perm = torch.randperm(int(available.numel()), device=available.device)
    random_idx = available[perm[:remaining]]
    patch_mask[random_idx] = True
    return patch_mask


def sample_vertex_mask(
    strategy: str,
    n_verts: int,
    ratio: float,
    device: torch.device,
    edge_index: Optional[torch.Tensor] = None,
    neighbors: Optional[Sequence[Sequence[int]]] = None,
    patch_hops: int = 2,
    patch_num_seeds: int = 16,
) -> torch.Tensor:
    strategy = str(strategy).lower()
    if strategy == "random":
        return random_vertex_mask(n_verts=n_verts, ratio=ratio, device=device)

    if neighbors is None:
        if edge_index is None:
            raise ValueError("edge_index or neighbors is required for patch-based masking")
        neighbors = build_neighbor_list(edge_index=edge_index, num_nodes=n_verts)

    if strategy == "patch":
        return patch_vertex_mask(
            n_verts=n_verts,
            ratio=ratio,
            neighbors=neighbors,
            device=device,
            patch_hops=patch_hops,
            patch_num_seeds=patch_num_seeds,
        )
    if strategy == "hybrid":
        return hybrid_vertex_mask(
            n_verts=n_verts,
            ratio=ratio,
            neighbors=neighbors,
            device=device,
            patch_hops=patch_hops,
            patch_num_seeds=patch_num_seeds,
        )
    raise ValueError(f"Unsupported mask strategy: {strategy}")


__all__ = [
    "build_neighbor_list",
    "sample_vertex_mask",
]
