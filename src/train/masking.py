"""Masking utilities for surface-graph SSL."""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
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


def analyze_mask_components(mask: torch.Tensor, neighbors: Sequence[Sequence[int]]) -> Dict[str, float]:
    mask_vec = mask.detach().cpu().bool().view(-1).tolist()
    masked_vertices = int(sum(mask_vec))
    if masked_vertices <= 0:
        return {
            "num_connected_components": 0,
            "largest_component_size": 0,
            "mean_component_size": 0.0,
            "largest_component_ratio": 0.0,
        }

    visited = set()
    component_sizes: List[int] = []

    for start in range(len(mask_vec)):
        if not mask_vec[start] or start in visited:
            continue

        size = 0
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            size += 1
            for nbr in neighbors[node]:
                nbr = int(nbr)
                if nbr in visited or not mask_vec[nbr]:
                    continue
                visited.add(nbr)
                queue.append(nbr)
        component_sizes.append(size)

    largest = max(component_sizes) if component_sizes else 0
    return {
        "num_connected_components": int(len(component_sizes)),
        "largest_component_size": int(largest),
        "mean_component_size": float(masked_vertices) / float(max(len(component_sizes), 1)),
        "largest_component_ratio": 0.0 if masked_vertices <= 0 else float(largest) / float(masked_vertices),
    }


def _allocate_component_budgets(
    target: int,
    num_components: int,
    size_jitter: float = 0.0,
) -> List[int]:
    num_components = max(1, min(int(num_components), int(target)))
    base = [target // num_components] * num_components
    for idx in range(target % num_components):
        base[idx] += 1

    size_jitter = max(0.0, float(size_jitter))
    if size_jitter <= 0.0 or num_components <= 1:
        return base

    jittered = [max(1, value) for value in base]
    movable = max(0, target - num_components)
    max_shift = int(round(movable * min(size_jitter, 1.0) * 0.5))
    if max_shift <= 0:
        return jittered

    for _ in range(max_shift):
        donors = [idx for idx, value in enumerate(jittered) if value > 1]
        if not donors:
            break
        donor = random.choice(donors)
        receiver = random.randrange(num_components)
        if donor == receiver:
            continue
        jittered[donor] -= 1
        jittered[receiver] += 1
    return jittered


def _extract_positions(positions: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if positions is None:
        return None
    pos = positions.detach()
    if pos.dim() != 2 or pos.shape[1] < 3:
        return None
    return pos[:, :3].cpu().numpy()


def _sample_seed_from_pool(
    perm: Sequence[int],
    masked: set[int],
    neighbors: Sequence[Sequence[int]],
    seed_min_hops: int,
) -> Optional[int]:
    if not perm:
        return None
    if seed_min_hops <= 0 or not masked:
        for node in perm:
            node = int(node)
            if node not in masked:
                return node
        return None

    blocked = set(int(node) for node in masked)
    frontier = deque((int(node), 0) for node in masked)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= seed_min_hops:
            continue
        for nbr in neighbors[node]:
            nbr = int(nbr)
            if nbr in blocked:
                continue
            blocked.add(nbr)
            frontier.append((nbr, depth + 1))

    for node in perm:
        node = int(node)
        if node not in masked and node not in blocked:
            return node
    for node in perm:
        node = int(node)
        if node not in masked:
            return node
    return None


def _push_frontier_item(
    frontier: list,
    node: int,
    priority: float,
    tie_breaker: float,
) -> None:
    heapq.heappush(frontier, (float(priority), float(tie_breaker), int(node)))


def _grow_connected_component(
    seed: int,
    budget: int,
    masked: set[int],
    neighbors: Sequence[Sequence[int]],
    positions: Optional[torch.Tensor] = None,
    grow_mode: str = "spatial",
) -> List[int]:
    if budget <= 0:
        return []

    grow_mode = str(grow_mode).lower()
    seed = int(seed)
    positions_cpu = _extract_positions(positions)
    seed_pos = None if positions_cpu is None else positions_cpu[seed]

    component: List[int] = []
    queued = {seed}
    frontier = []
    _push_frontier_item(frontier, node=seed, priority=0.0, tie_breaker=random.random())

    while frontier and len(component) < budget:
        _, _, node = heapq.heappop(frontier)
        if node in masked:
            continue

        masked.add(node)
        component.append(node)
        if len(component) >= budget:
            break

        nbrs = list(neighbors[node])
        random.shuffle(nbrs)
        for nbr in nbrs:
            nbr = int(nbr)
            if nbr in masked or nbr in queued:
                continue
            queued.add(nbr)
            if grow_mode == "spatial" and positions_cpu is not None and seed_pos is not None:
                diff = positions_cpu[nbr] - seed_pos
                priority = float(np.dot(diff, diff))
            else:
                priority = float(len(component))
            _push_frontier_item(frontier, node=nbr, priority=priority, tie_breaker=random.random())
    return component


def _expand_existing_regions(
    target: int,
    masked: set[int],
    neighbors: Sequence[Sequence[int]],
    positions: Optional[torch.Tensor] = None,
) -> None:
    if len(masked) >= target or not masked:
        return

    positions_cpu = _extract_positions(positions)
    frontier = []
    queued = set(masked)

    for node in list(masked):
        for nbr in neighbors[int(node)]:
            nbr = int(nbr)
            if nbr in queued:
                continue
            queued.add(nbr)
            if positions_cpu is not None:
                diff = positions_cpu[nbr] - positions_cpu[int(node)]
                priority = float(np.dot(diff, diff))
            else:
                priority = 0.0
            _push_frontier_item(frontier, node=nbr, priority=priority, tie_breaker=random.random())

    while frontier and len(masked) < target:
        _, _, node = heapq.heappop(frontier)
        if node in masked:
            continue
        masked.add(node)
        for nbr in neighbors[node]:
            nbr = int(nbr)
            if nbr in queued:
                continue
            queued.add(nbr)
            if positions_cpu is not None:
                diff = positions_cpu[nbr] - positions_cpu[node]
                priority = float(np.dot(diff, diff))
            else:
                priority = 0.0
            _push_frontier_item(frontier, node=nbr, priority=priority, tie_breaker=random.random())


def random_vertex_mask(
    n_verts: int,
    ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    blocked: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    blocked_set = set(int(idx) for idx in (blocked or []))
    available = [idx for idx in range(n_verts) if idx not in blocked_set]
    if not available:
        return torch.zeros(n_verts, dtype=torch.bool, device=device)

    target = min(_target_mask_count(n_verts=n_verts, ratio=ratio), len(available))
    perm = torch.randperm(len(available), generator=generator)
    mask = torch.zeros(n_verts, dtype=torch.bool)
    chosen = [available[int(idx)] for idx in perm[:target].tolist()]
    if chosen:
        mask[chosen] = True
    return mask.to(device=device)


def patch_vertex_mask(
    n_verts: int,
    ratio: float,
    neighbors: Sequence[Sequence[int]],
    device: torch.device,
    patch_hops: int = 2,
    patch_num_seeds: int = 16,
    blocked: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    blocked_set = set(int(idx) for idx in (blocked or []))
    available_count = max(0, n_verts - len(blocked_set))
    if available_count <= 0:
        return torch.zeros(n_verts, dtype=torch.bool, device=device)

    target = min(_target_mask_count(n_verts=n_verts, ratio=ratio), available_count)
    patch_hops = max(0, int(patch_hops))
    patch_num_seeds = max(1, min(int(patch_num_seeds), available_count))

    perm = [int(idx) for idx in torch.randperm(n_verts).tolist() if int(idx) not in blocked_set]
    seed_candidates = [int(idx) for idx in perm[:patch_num_seeds]]
    seed_queue = deque(seed_candidates)
    visited = set(blocked_set)
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


def region_vertex_mask(
    n_verts: int,
    ratio: float,
    neighbors: Sequence[Sequence[int]],
    device: torch.device,
    region_num_components: int = 4,
    positions: Optional[torch.Tensor] = None,
    region_grow_mode: str = "spatial",
    region_seed_min_hops: int = 4,
    region_size_jitter: float = 0.0,
    blocked: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    blocked_set = set(int(idx) for idx in (blocked or []))
    available_count = max(0, n_verts - len(blocked_set))
    if available_count <= 0:
        return torch.zeros(n_verts, dtype=torch.bool, device=device)

    target = min(_target_mask_count(n_verts=n_verts, ratio=ratio), available_count)
    region_num_components = max(1, min(int(region_num_components), target, available_count))
    region_seed_min_hops = max(0, int(region_seed_min_hops))
    budgets = _allocate_component_budgets(
        target=target,
        num_components=region_num_components,
        size_jitter=region_size_jitter,
    )

    perm = [int(idx) for idx in torch.randperm(n_verts).tolist() if int(idx) not in blocked_set]
    masked = set(blocked_set)
    for budget in budgets:
        seed = _sample_seed_from_pool(
            perm=perm,
            masked=masked,
            neighbors=neighbors,
            seed_min_hops=region_seed_min_hops,
        )
        if seed is None:
            break
        _grow_connected_component(
            seed=seed,
            budget=budget,
            masked=masked,
            neighbors=neighbors,
            positions=positions,
            grow_mode=region_grow_mode,
        )
        if len(masked) >= target:
            break

    if len(masked) < target:
        _expand_existing_regions(
            target=target,
            masked=masked,
            neighbors=neighbors,
            positions=positions,
        )
    if len(masked) < target:
        for node in perm:
            node = int(node)
            if node in masked:
                continue
            masked.add(node)
            if len(masked) >= target:
                break

    mask = torch.zeros(n_verts, dtype=torch.bool)
    visible_masked = [idx for idx in masked if idx not in blocked_set]
    if visible_masked:
        mask[visible_masked] = True
    return mask.to(device=device)


def multiscale_vertex_mask(
    n_verts: int,
    ratio: float,
    neighbors: Sequence[Sequence[int]],
    device: torch.device,
    patch_hops: int = 2,
    patch_num_seeds: int = 16,
    region_num_components: int = 4,
    positions: Optional[torch.Tensor] = None,
    region_grow_mode: str = "spatial",
    region_seed_min_hops: int = 4,
    region_size_jitter: float = 0.0,
    region_fraction: float = 0.6,
    patch_fraction: float = 0.25,
) -> torch.Tensor:
    target = _target_mask_count(n_verts=n_verts, ratio=ratio)
    region_fraction = max(0.0, min(1.0, float(region_fraction)))
    patch_fraction = max(0.0, min(1.0 - region_fraction, float(patch_fraction)))

    region_target = int(round(target * region_fraction))
    patch_target = int(round(target * patch_fraction))
    if region_fraction > 0.0 and region_target <= 0:
        region_target = 1
    if patch_fraction > 0.0 and patch_target <= 0 and target - region_target > 0:
        patch_target = 1
    if region_target + patch_target > target:
        overflow = region_target + patch_target - target
        patch_target = max(0, patch_target - overflow)

    masked: set[int] = set()

    if region_target > 0:
        region_mask = region_vertex_mask(
            n_verts=n_verts,
            ratio=float(region_target) / float(n_verts),
            neighbors=neighbors,
            device=device,
            region_num_components=region_num_components,
            positions=positions,
            region_grow_mode=region_grow_mode,
            region_seed_min_hops=region_seed_min_hops,
            region_size_jitter=region_size_jitter,
            blocked=masked,
        )
        masked.update(int(idx) for idx in region_mask.nonzero(as_tuple=False).view(-1).tolist())

    remaining = target - len(masked)
    patch_target = min(patch_target, remaining)
    if patch_target > 0:
        patch_mask = patch_vertex_mask(
            n_verts=n_verts,
            ratio=float(patch_target) / float(n_verts),
            neighbors=neighbors,
            device=device,
            patch_hops=patch_hops,
            patch_num_seeds=patch_num_seeds,
            blocked=masked,
        )
        masked.update(int(idx) for idx in patch_mask.nonzero(as_tuple=False).view(-1).tolist())

    remaining = target - len(masked)
    if remaining > 0:
        random_mask = random_vertex_mask(
            n_verts=n_verts,
            ratio=float(remaining) / float(n_verts),
            device=device,
            blocked=masked,
        )
        masked.update(int(idx) for idx in random_mask.nonzero(as_tuple=False).view(-1).tolist())

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
    hybrid_patch_fraction: float = 0.7,
    region_num_components: int = 4,
    positions: Optional[torch.Tensor] = None,
    region_grow_mode: str = "spatial",
    region_seed_min_hops: int = 4,
    region_size_jitter: float = 0.0,
    multiscale_region_fraction: float = 0.6,
    multiscale_patch_fraction: float = 0.25,
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
    if strategy == "region":
        return region_vertex_mask(
            n_verts=n_verts,
            ratio=ratio,
            neighbors=neighbors,
            device=device,
            region_num_components=region_num_components,
            positions=positions,
            region_grow_mode=region_grow_mode,
            region_seed_min_hops=region_seed_min_hops,
            region_size_jitter=region_size_jitter,
        )
    if strategy == "hybrid":
        return hybrid_vertex_mask(
            n_verts=n_verts,
            ratio=ratio,
            neighbors=neighbors,
            device=device,
            patch_hops=patch_hops,
            patch_num_seeds=patch_num_seeds,
            patch_fraction=hybrid_patch_fraction,
        )
    if strategy == "multiscale":
        return multiscale_vertex_mask(
            n_verts=n_verts,
            ratio=ratio,
            neighbors=neighbors,
            device=device,
            patch_hops=patch_hops,
            patch_num_seeds=patch_num_seeds,
            region_num_components=region_num_components,
            positions=positions,
            region_grow_mode=region_grow_mode,
            region_seed_min_hops=region_seed_min_hops,
            region_size_jitter=region_size_jitter,
            region_fraction=multiscale_region_fraction,
            patch_fraction=multiscale_patch_fraction,
        )
    raise ValueError(f"Unsupported mask strategy: {strategy}")


__all__ = [
    "analyze_mask_components",
    "build_neighbor_list",
    "sample_vertex_mask",
]
