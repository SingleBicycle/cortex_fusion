"""Train cortical surface graph SSL with masked feature reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset_surface import IGNORE_INDEX, SurfaceSubjectDataset
from src.data.feature_schema import get_feature_schema, list_input_modes
from src.models.adgcn import GraphBranchModel
from src.train.losses import get_recon_dim_indices, masked_reconstruction_loss
from src.train.masking import analyze_mask_components, build_neighbor_list, sample_vertex_mask
from src.train.recon_artifacts import save_recon_examples, write_json, write_per_dim_recon_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_subject(batch):
    return batch[0]


def to_device_hemi(hemi_batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "X": hemi_batch["X"].to(device=device, dtype=torch.float32),
        "X_target": hemi_batch["X_target"].to(device=device, dtype=torch.float32),
        "edge_index": hemi_batch["edge_index"].to(device=device, dtype=torch.long),
        "y": hemi_batch["y"].to(device=device, dtype=torch.long),
        "mask_valid": hemi_batch["mask_valid"].to(device=device, dtype=torch.bool),
        "recon_weights": hemi_batch["recon_weights"].to(device=device, dtype=torch.float32),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ADGCN graph branch with reconstruction-first SSL")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--res", type=str, default="fsaverage6")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default=None)

    parser.add_argument("--input_mode", type=str, default="main5", choices=list_input_modes())
    parser.add_argument("--normalization_mode", type=str, default="global_train", choices=["per_hemi", "global_train"])
    parser.add_argument(
        "--xyz_norm_mode",
        type=str,
        default="subject_rms",
        choices=["subject_rms", "center_only", "none"],
    )
    parser.add_argument(
        "--morph_norm_mode",
        type=str,
        default="subject_zscore",
        choices=["subject_zscore", "raw"],
    )
    parser.add_argument(
        "--graph_global_mode",
        type=str,
        default="none",
        choices=["none", "summary5"],
    )
    parser.add_argument(
        "--mask_strategy",
        type=str,
        default="hybrid",
        choices=["random", "patch", "hybrid", "region", "multiscale"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.35)
    parser.add_argument("--patch_hops", type=int, default=2)
    parser.add_argument("--patch_num_seeds", type=int, default=16)
    parser.add_argument("--hybrid_patch_fraction", type=float, default=0.7)
    parser.add_argument("--region_num_components", type=int, default=4)
    parser.add_argument("--region_grow_mode", type=str, default="spatial", choices=["bfs", "spatial"])
    parser.add_argument("--region_seed_min_hops", type=int, default=4)
    parser.add_argument("--region_size_jitter", type=float, default=0.0)
    parser.add_argument("--multiscale_region_fraction", type=float, default=0.6)
    parser.add_argument("--multiscale_patch_fraction", type=float, default=0.25)
    parser.add_argument("--mask_fill", type=str, default="zero", choices=["zero", "mask_token"])
    parser.add_argument("--mask_random_sub_prob", type=float, default=0.0)
    parser.add_argument("--decoder_remask_ratio", type=float, default=0.0)
    parser.add_argument("--decoder_remask_strategy", type=str, default="random", choices=["random", "region"])
    parser.add_argument("--decoder_remask_fill", type=str, default="mask_token", choices=["zero", "mask_token"])
    parser.add_argument("--decoder_remask_views", type=int, default=1)
    parser.add_argument("--latent_consistency_weight", type=float, default=0.0)
    parser.add_argument("--recon_loss", type=str, default="wmse", choices=["wmse", "huber"])
    parser.add_argument("--recon_on", type=str, default="all", choices=["all", "morph_only"])
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=0.2)
    parser.add_argument("--use_ce", type=int, default=0, choices=[0, 1])
    parser.add_argument("--alpha_ce", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dims", type=int, nargs=7, default=[32, 64, 128, 256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fuse_mode", type=str, default="sum", choices=["sum", "concat"])
    parser.add_argument("--backbone_type", type=str, default="gcn", choices=["gcn", "mlp", "mesh_hier", "graph_unet"])
    parser.add_argument("--pool_mode", type=str, default="mean", choices=["mean", "attn", "pca", "mean_pca"])
    parser.add_argument("--hierarchy_dir", type=str, default=None)
    parser.add_argument("--max_hierarchy_levels", type=int, default=2)
    parser.add_argument("--graph_unet_depth", type=int, default=2)
    parser.add_argument("--graph_unet_pool_ratios", type=float, nargs="+", default=[0.8, 0.8])

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--eval_examples", type=int, default=3)

    parser.add_argument("--edge_cache_dir", type=str, default="cache/templates")
    parser.add_argument("--random_resolution", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _resolve_lambda_ce(args: argparse.Namespace) -> float:
    if args.alpha_ce is not None:
        return float(args.alpha_ce)
    return float(args.lambda_ce)


def _build_model_config(
    args: argparse.Namespace,
    schema: Dict[str, object],
    num_classes: int,
) -> Dict[str, object]:
    return {
        "input_mode": args.input_mode,
        "in_dim": int(schema["in_dim"]),
        "geo_dim": int(schema["geo_dim"]),
        "morph_dim": int(schema["morph_dim"]),
        "hidden_dim": int(args.hidden_dim),
        "dims": [int(x) for x in args.dims],
        "num_classes": int(num_classes),
        "dropout": float(args.dropout),
        "fuse_mode": args.fuse_mode,
        "backbone_type": str(getattr(args, "backbone_type", "gcn")),
        "pool_mode": str(getattr(args, "pool_mode", "mean")),
        "hierarchy_dir": getattr(args, "hierarchy_dir", None),
        "edge_cache_dir": getattr(args, "edge_cache_dir", None),
        "max_hierarchy_levels": int(getattr(args, "max_hierarchy_levels", 2)),
        "graph_unet_depth": int(getattr(args, "graph_unet_depth", 2)),
        "graph_unet_pool_ratios": [
            float(x) for x in getattr(args, "graph_unet_pool_ratios", [0.8, 0.8])
        ],
        "graph_global_dim": 5 if str(getattr(args, "graph_global_mode", "none")) == "summary5" else 0,
    }


def _build_split_indices(
    num_items: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    if num_items <= 0:
        raise ValueError("Dataset must contain at least one subject.")
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Expected 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1.")

    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if num_items == 1:
        return {"train": indices, "val": [], "test": []}

    n_val = int(round(num_items * val_ratio))
    n_test = int(round(num_items * test_ratio))

    if val_ratio > 0.0 and num_items >= 3:
        n_val = max(1, n_val)
    if test_ratio > 0.0 and num_items >= 3:
        n_test = max(1, n_test)

    while n_val + n_test >= num_items:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    train_end = num_items - n_val - n_test
    train_indices = indices[:train_end]
    val_indices = indices[train_end: train_end + n_val]
    test_indices = indices[train_end + n_val:]

    if not train_indices:
        raise RuntimeError("Train split is empty after split construction.")

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_subject,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _load_existing_best_from_log(log_path: str) -> Tuple[float, int]:
    if not os.path.exists(log_path):
        return float("inf"), 0

    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return float("inf"), 0

    best_row = min(rows, key=lambda row: float(row["selection_recon_loss"]))
    return float(best_row["selection_recon_loss"]), int(best_row["epoch"])


def _capture_rng_state() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _extract_hemi_positions(
    x_target: torch.Tensor,
    feature_schema: Dict[str, object],
) -> torch.Tensor | None:
    geo_dim = int(feature_schema.get("geo_dim", 0))
    if geo_dim >= 6 and x_target.shape[1] >= 6:
        return 0.5 * (x_target[:, :3] + x_target[:, 3:6])
    if geo_dim >= 3 and x_target.shape[1] >= 3:
        return x_target[:, :3]
    return None


def _apply_input_mask(
    x: torch.Tensor,
    vertex_mask: torch.Tensor,
    input_keep_mask: torch.Tensor,
    mask_token: torch.Tensor | None = None,
    random_sub_prob: float = 0.0,
) -> torch.Tensor:
    x_masked = x.clone()
    masked_count = int(vertex_mask.sum().item())
    if masked_count <= 0:
        return x_masked

    visible_part = x_masked[vertex_mask] * input_keep_mask.unsqueeze(0)
    masked_dims = (1.0 - input_keep_mask).unsqueeze(0)
    replacement = visible_part
    if mask_token is not None:
        replacement = replacement + (mask_token.to(dtype=x_masked.dtype).unsqueeze(0) * masked_dims)

    random_sub_prob = max(0.0, min(1.0, float(random_sub_prob)))
    if random_sub_prob > 0.0:
        substitute_mask = torch.rand(masked_count, device=x.device) < random_sub_prob
        if bool(substitute_mask.any()):
            source_idx = torch.randint(0, x.shape[0], (int(substitute_mask.sum().item()),), device=x.device)
            replacement[substitute_mask] = (
                visible_part[substitute_mask] + (x[source_idx].to(dtype=x_masked.dtype) * masked_dims)
            )

    x_masked[vertex_mask] = replacement
    return x_masked


def _prepare_batch(
    batch: Dict,
    device: torch.device,
    args: argparse.Namespace,
    input_keep_mask: torch.Tensor,
    adjacency_cache: Dict[str, List[List[int]]],
    mask_token: torch.Tensor | None = None,
) -> Dict[str, object]:
    if batch["res"] not in adjacency_cache:
        adjacency_cache[batch["res"]] = build_neighbor_list(
            edge_index=batch["lh"]["edge_index"],
            num_nodes=int(batch["lh"]["X"].shape[0]),
        )
    neighbors = adjacency_cache[batch["res"]]

    feature_schema = batch.get("feature_schema", {})
    # Regional mask growth is CPU-side; keep positions on CPU to avoid repeated GPU->CPU syncs.
    hemi_positions_cpu = {
        hemi_name: _extract_hemi_positions(
            x_target=batch[hemi_name]["X_target"],
            feature_schema=feature_schema,
        )
        for hemi_name in ("lh", "rh")
    }
    hemi_batches = {
        "lh": to_device_hemi(batch["lh"], device=device),
        "rh": to_device_hemi(batch["rh"], device=device),
    }
    graph_global = None
    if "graph_global" in batch:
        graph_global = batch["graph_global"].to(device=device, dtype=torch.float32).view(-1)
    hemi_inputs = {}
    hemi_targets = {}
    hemi_masks = {}
    hemi_masked_inputs = {}
    hemi_mask_stats = {}
    hemi_positions = {}
    for hemi_name in ("lh", "rh"):
        hemi_b = hemi_batches[hemi_name]
        hemi_positions[hemi_name] = hemi_positions_cpu[hemi_name]
        vertex_mask = sample_vertex_mask(
            strategy=args.mask_strategy,
            n_verts=int(hemi_b["X"].shape[0]),
            ratio=args.mask_ratio,
            device=device,
            edge_index=batch[hemi_name]["edge_index"],
            neighbors=neighbors,
            patch_hops=args.patch_hops,
            patch_num_seeds=args.patch_num_seeds,
            hybrid_patch_fraction=args.hybrid_patch_fraction,
            region_num_components=args.region_num_components,
            positions=hemi_positions[hemi_name],
            region_grow_mode=args.region_grow_mode,
            region_seed_min_hops=args.region_seed_min_hops,
            region_size_jitter=args.region_size_jitter,
            multiscale_region_fraction=args.multiscale_region_fraction,
            multiscale_patch_fraction=args.multiscale_patch_fraction,
        )
        hemi_mask_stats[hemi_name] = analyze_mask_components(mask=vertex_mask, neighbors=neighbors)

        x_masked = _apply_input_mask(
            x=hemi_b["X"],
            vertex_mask=vertex_mask,
            input_keep_mask=input_keep_mask,
            mask_token=mask_token,
            random_sub_prob=args.mask_random_sub_prob,
        )

        hemi_inputs[hemi_name] = {
            "x": x_masked,
            "edge_index": hemi_b["edge_index"],
            "valid_mask": hemi_b["mask_valid"],
            "res_name": batch["res"],
        }
        hemi_targets[hemi_name] = hemi_b["X_target"]
        hemi_masks[hemi_name] = vertex_mask
        hemi_masked_inputs[hemi_name] = x_masked

    return {
        "hemi_batches": hemi_batches,
        "hemi_inputs": hemi_inputs,
        "hemi_targets": hemi_targets,
        "hemi_masks": hemi_masks,
        "hemi_masked_inputs": hemi_masked_inputs,
        "hemi_mask_stats": hemi_mask_stats,
        "hemi_positions": hemi_positions,
        "neighbors": neighbors,
        "graph_global": graph_global,
    }


def _compute_batch_recon_terms(
    out: Dict[str, object],
    hemi_batches: Dict[str, Dict[str, torch.Tensor]],
    hemi_targets: Dict[str, torch.Tensor],
    hemi_masks: Dict[str, torch.Tensor],
    active_dims: List[int],
    recon_loss_type: str,
    use_ce: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_losses = []
    ce_losses = []
    per_dim_losses = []

    for hemi_name in ("lh", "rh"):
        hemi_b = hemi_batches[hemi_name]
        recon_result = masked_reconstruction_loss(
            pred=out[hemi_name]["recon_pred"],
            target=hemi_targets[hemi_name],
            vertex_mask=hemi_masks[hemi_name],
            dim_weights=hemi_b["recon_weights"],
            active_dims=active_dims,
            loss_type=recon_loss_type,
        )
        recon_losses.append(recon_result["loss"])
        per_dim_losses.append(recon_result["per_dim_loss"])

        if use_ce:
            logits = out[hemi_name].get("label_logits", None)
            if logits is None:
                raise RuntimeError("use_ce=1 but model has no label head")
            ce_losses.append(F.cross_entropy(logits, hemi_b["y"], ignore_index=IGNORE_INDEX))

    recon_loss = torch.stack(recon_losses).mean()
    per_dim_loss = torch.stack(per_dim_losses).mean(dim=0)
    if use_ce:
        ce_loss = torch.stack(ce_losses).mean()
    else:
        ce_loss = torch.zeros((), device=recon_loss.device)
    return recon_loss, ce_loss, per_dim_loss


def _compute_decoder_remask_terms(
    model: GraphBranchModel,
    out: Dict[str, object],
    prepared: Dict[str, object],
    args: argparse.Namespace,
    active_dims: List[int],
    recon_loss_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zero_scalar = out["lh"]["H_v"].new_zeros(())
    if float(args.decoder_remask_ratio) <= 0.0 or int(args.decoder_remask_views) <= 0:
        return zero_scalar, out["lh"]["recon_pred"].new_zeros(prepared["hemi_targets"]["lh"].shape[1])

    decoder_mask_token = getattr(model, "_decoder_mask_token_runtime", None)
    remask_losses = []
    remask_per_dim_losses = []

    for _ in range(int(args.decoder_remask_views)):
        hemi_losses = []
        hemi_per_dim_losses = []
        for hemi_name in ("lh", "rh"):
            hemi_b = prepared["hemi_batches"][hemi_name]
            latent_mask = sample_vertex_mask(
                strategy=args.decoder_remask_strategy,
                n_verts=int(hemi_b["X"].shape[0]),
                ratio=args.decoder_remask_ratio,
                device=hemi_b["X"].device,
                edge_index=hemi_b["edge_index"],
                neighbors=prepared["neighbors"],
                patch_hops=args.patch_hops,
                patch_num_seeds=args.patch_num_seeds,
                hybrid_patch_fraction=args.hybrid_patch_fraction,
                region_num_components=args.region_num_components,
                positions=prepared["hemi_positions"].get(hemi_name, None),
                region_grow_mode=args.region_grow_mode,
                region_seed_min_hops=args.region_seed_min_hops,
                region_size_jitter=args.region_size_jitter,
                multiscale_region_fraction=args.multiscale_region_fraction,
                multiscale_patch_fraction=args.multiscale_patch_fraction,
            )

            latent = out[hemi_name]["H_v"]
            latent_masked = latent.clone()
            if args.decoder_remask_fill == "mask_token" and decoder_mask_token is not None:
                latent_masked[latent_mask] = decoder_mask_token.to(dtype=latent.dtype).unsqueeze(0)
            else:
                latent_masked[latent_mask] = 0.0

            remask_pred = model.recon_head(latent_masked)
            remask_result = masked_reconstruction_loss(
                pred=remask_pred,
                target=prepared["hemi_targets"][hemi_name],
                vertex_mask=prepared["hemi_masks"][hemi_name],
                dim_weights=hemi_b["recon_weights"],
                active_dims=active_dims,
                loss_type=recon_loss_type,
            )
            hemi_losses.append(remask_result["loss"])
            hemi_per_dim_losses.append(remask_result["per_dim_loss"])

        remask_losses.append(torch.stack(hemi_losses).mean())
        remask_per_dim_losses.append(torch.stack(hemi_per_dim_losses).mean(dim=0))

    return torch.stack(remask_losses).mean(), torch.stack(remask_per_dim_losses).mean(dim=0)


def _compute_latent_consistency_loss(
    model: GraphBranchModel,
    batch: Dict,
    device: torch.device,
    args: argparse.Namespace,
    input_keep_mask: torch.Tensor,
    adjacency_cache: Dict[str, List[List[int]]],
    mask_token: torch.Tensor | None,
    anchor_z_graph: torch.Tensor,
) -> torch.Tensor:
    if float(args.latent_consistency_weight) <= 0.0:
        return anchor_z_graph.new_zeros(())

    alt_prepared = _prepare_batch(
        batch=batch,
        device=device,
        args=args,
        input_keep_mask=input_keep_mask,
        adjacency_cache=adjacency_cache,
        mask_token=mask_token,
    )
    alt_out = model(
        lh=alt_prepared["hemi_inputs"]["lh"],
        rh=alt_prepared["hemi_inputs"]["rh"],
        graph_global=alt_prepared.get("graph_global", None),
    )
    return F.mse_loss(anchor_z_graph, alt_out["z_graph"])


def _mean_feature_metric(
    feature_names: List[str],
    per_dim_mse: np.ndarray,
    per_dim_count: np.ndarray,
    predicate,
) -> float:
    idxs = [idx for idx, name in enumerate(feature_names) if predicate(str(name))]
    valid = [idx for idx in idxs if int(per_dim_count[idx]) > 0]
    if not valid:
        return 0.0
    return float(per_dim_mse[valid].mean())


def _compute_named_metrics(
    feature_names: List[str],
    per_dim_mse: np.ndarray,
    per_dim_count: np.ndarray,
) -> Dict[str, float]:
    xyz_metric = _mean_feature_metric(
        feature_names=feature_names,
        per_dim_mse=per_dim_mse,
        per_dim_count=per_dim_count,
        predicate=lambda name: name.endswith("_x") or name.endswith("_y") or name.endswith("_z"),
    )
    thickness_metric = _mean_feature_metric(
        feature_names=feature_names,
        per_dim_mse=per_dim_mse,
        per_dim_count=per_dim_count,
        predicate=lambda name: name == "thickness",
    )
    curvature_metric = _mean_feature_metric(
        feature_names=feature_names,
        per_dim_mse=per_dim_mse,
        per_dim_count=per_dim_count,
        predicate=lambda name: name == "curvature",
    )
    return {
        "xyz_mse": xyz_metric,
        "thickness_mse": thickness_metric,
        "curvature_mse": curvature_metric,
    }


def _evaluate_loader(
    model: GraphBranchModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    active_dims: List[int],
    feature_names: List[str],
    recon_weights: List[float],
    input_keep_mask: torch.Tensor,
    use_ce: bool,
    seed: int,
    split_name: str,
    example_limit: int,
) -> Dict[str, object]:
    if loader is None:
        return {
            "available": False,
            "split": split_name,
            "num_subjects": 0,
            "num_hemis": 0,
            "total_vertices": 0,
            "masked_vertices": 0,
            "realized_mask_ratio": 0.0,
            "mean_num_connected_components": 0.0,
            "mean_largest_component_size": 0.0,
            "mean_component_size": 0.0,
            "mean_largest_component_ratio": 0.0,
            "weighted_mse": 0.0,
            "unweighted_mse": 0.0,
            "xyz_mse": 0.0,
            "thickness_mse": 0.0,
            "curvature_mse": 0.0,
            "rmse": 0.0,
            "ce_loss": 0.0,
            "per_dim_mse": [0.0] * len(feature_names),
            "per_dim_count": [0] * len(feature_names),
            "active_dims": list(active_dims),
            "active_feature_names": [feature_names[idx] for idx in active_dims],
            "examples": [],
        }

    rng_state = _capture_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    adjacency_cache: Dict[str, List[List[int]]] = {}
    active_set = set(active_dims)

    total_ce = 0.0
    total_batches = 0
    total_hemis = 0
    total_vertices = 0
    total_masked_vertices = 0
    mask_stat_sums = {
        "num_connected_components": 0.0,
        "largest_component_size": 0.0,
        "mean_component_size": 0.0,
        "largest_component_ratio": 0.0,
    }

    per_dim_sum = np.zeros(len(feature_names), dtype=np.float64)
    per_dim_count = np.zeros(len(feature_names), dtype=np.int64)
    examples: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            total_batches += 1
            prepared = _prepare_batch(
                batch=batch,
                device=device,
                args=args,
                input_keep_mask=input_keep_mask,
                adjacency_cache=adjacency_cache,
                mask_token=getattr(model, "_input_mask_token_runtime", None),
            )
            out = model(
                lh=prepared["hemi_inputs"]["lh"],
                rh=prepared["hemi_inputs"]["rh"],
                graph_global=prepared.get("graph_global", None),
            )
            _, ce_loss, _ = _compute_batch_recon_terms(
                out=out,
                hemi_batches=prepared["hemi_batches"],
                hemi_targets=prepared["hemi_targets"],
                hemi_masks=prepared["hemi_masks"],
                active_dims=active_dims,
                recon_loss_type="wmse",
                use_ce=use_ce,
            )

            total_ce += float(ce_loss.detach().cpu().item())

            for hemi_name in ("lh", "rh"):
                pred = out[hemi_name]["recon_pred"].detach()
                target = prepared["hemi_targets"][hemi_name]
                mask = prepared["hemi_masks"][hemi_name]

                total_hemis += 1
                total_vertices += int(mask.numel())
                masked_vertices = int(mask.sum().item())
                total_masked_vertices += masked_vertices
                for key in mask_stat_sums:
                    mask_stat_sums[key] += float(prepared["hemi_mask_stats"][hemi_name][key])
                if masked_vertices == 0:
                    continue

                sq_err = (pred[mask] - target[mask]) ** 2
                for dim_idx in active_dims:
                    per_dim_sum[dim_idx] += float(sq_err[:, dim_idx].sum().detach().cpu().item())
                    per_dim_count[dim_idx] += masked_vertices

            if len(examples) < example_limit:
                example = {
                    "split": split_name,
                    "sid": str(batch["sid"]),
                    "res": str(batch["res"]),
                    "input_mode": str(batch["input_mode"]),
                    "hemis": {},
                }
                for hemi_name in ("lh", "rh"):
                    example["hemis"][hemi_name] = {
                        "original": prepared["hemi_targets"][hemi_name].detach().cpu().numpy(),
                        "masked": prepared["hemi_masked_inputs"][hemi_name].detach().cpu().numpy(),
                        "recon": out[hemi_name]["recon_pred"].detach().cpu().numpy(),
                        "mask": prepared["hemi_masks"][hemi_name].detach().cpu().numpy(),
                        "mask_stats": dict(prepared["hemi_mask_stats"][hemi_name]),
                    }
                examples.append(example)

    _restore_rng_state(rng_state)

    per_dim_mse = np.zeros(len(feature_names), dtype=np.float64)
    valid = per_dim_count > 0
    per_dim_mse[valid] = per_dim_sum[valid] / per_dim_count[valid]
    active_weights = np.asarray(recon_weights, dtype=np.float64)
    active_weights[[idx for idx in range(len(feature_names)) if idx not in active_set]] = 0.0
    active_weight_sum = float(active_weights.sum())
    weighted_mse = 0.0
    if active_weight_sum > 0.0:
        weighted_mse = float((per_dim_mse * active_weights).sum() / active_weight_sum)
    unweighted_mse = float(per_dim_mse[active_dims].mean()) if active_dims else 0.0
    realized_ratio = 0.0 if total_vertices == 0 else float(total_masked_vertices) / float(total_vertices)
    mean_mask_stats = {
        key: (value / float(max(total_hemis, 1)))
        for key, value in mask_stat_sums.items()
    }
    named_metrics = _compute_named_metrics(
        feature_names=feature_names,
        per_dim_mse=per_dim_mse,
        per_dim_count=per_dim_count,
    )

    return {
        "available": True,
        "split": split_name,
        "num_subjects": len(loader.dataset),
        "num_hemis": total_hemis,
        "total_vertices": total_vertices,
        "masked_vertices": total_masked_vertices,
        "realized_mask_ratio": realized_ratio,
        "mean_num_connected_components": mean_mask_stats["num_connected_components"],
        "mean_largest_component_size": mean_mask_stats["largest_component_size"],
        "mean_component_size": mean_mask_stats["mean_component_size"],
        "mean_largest_component_ratio": mean_mask_stats["largest_component_ratio"],
        "weighted_mse": weighted_mse,
        "unweighted_mse": unweighted_mse,
        "xyz_mse": named_metrics["xyz_mse"],
        "thickness_mse": named_metrics["thickness_mse"],
        "curvature_mse": named_metrics["curvature_mse"],
        "rmse": math.sqrt(max(unweighted_mse, 0.0)),
        "ce_loss": total_ce / max(total_batches, 1),
        "per_dim_mse": per_dim_mse.tolist(),
        "per_dim_count": per_dim_count.tolist(),
        "active_dims": list(active_dims),
        "active_feature_names": [feature_names[idx] for idx in active_dims],
        "examples": examples,
    }


def main() -> None:
    args = build_argparser().parse_args()
    args.lambda_ce = _resolve_lambda_ce(args)
    if not (0.0 <= float(args.mask_random_sub_prob) <= 1.0):
        raise ValueError("Expected 0 <= mask_random_sub_prob <= 1.")
    if int(args.decoder_remask_views) <= 0:
        raise ValueError("decoder_remask_views must be >= 1.")
    if float(args.decoder_remask_ratio) < 0.0:
        raise ValueError("decoder_remask_ratio must be >= 0.")
    if float(args.latent_consistency_weight) < 0.0:
        raise ValueError("latent_consistency_weight must be >= 0.")
    if float(args.multiscale_region_fraction) < 0.0 or float(args.multiscale_patch_fraction) < 0.0:
        raise ValueError("multiscale fractions must be >= 0.")
    if float(args.multiscale_region_fraction) + float(args.multiscale_patch_fraction) > 1.0:
        raise ValueError("Expected multiscale_region_fraction + multiscale_patch_fraction <= 1.")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("This trainer expects batch_size=1 (subject-level batches).")
    resume_ckpt = None
    if args.resume_ckpt is not None:
        args.resume_ckpt = os.path.abspath(args.resume_ckpt)
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_ckpt}")
        resume_ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        resume_norm_mode = str((resume_ckpt.get("args", {}) or {}).get("normalization_mode", "per_hemi"))
        if resume_norm_mode != str(args.normalization_mode):
            raise ValueError(
                f"resume checkpoint normalization_mode={resume_norm_mode!r} does not match "
                f"requested normalization_mode={args.normalization_mode!r}"
            )

    schema = get_feature_schema(args.input_mode)
    recon_dim_indices = get_recon_dim_indices(
        in_dim=int(schema["in_dim"]),
        geo_dim=int(schema["geo_dim"]),
        morph_dim=int(schema["morph_dim"]),
        recon_on=args.recon_on,
    )
    feature_names = list(schema["feature_names"])
    recon_feature_names = [feature_names[idx] for idx in recon_dim_indices]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    use_ce = bool(args.use_ce)

    dataset = SurfaceSubjectDataset(
        manifest_csv=args.manifest,
        res=args.res,
        random_resolution=bool(args.random_resolution),
        edge_cache_dir=args.edge_cache_dir,
        seed=args.seed,
        in_memory_cache=False,
        input_mode=args.input_mode,
        normalization_mode=args.normalization_mode,
        xyz_norm_mode=args.xyz_norm_mode,
        morph_norm_mode=args.morph_norm_mode,
        graph_global_mode=args.graph_global_mode,
    )
    if use_ce and dataset.num_classes <= 0:
        raise ValueError(
            "use_ce=1 requires parcel-level annotation classes. "
            "The current manifest only provides cortex masks (.label), so use --use_ce 0."
        )

    if resume_ckpt is not None and "split_indices" in resume_ckpt:
        split_indices = {
            split_name: [int(idx) for idx in resume_ckpt["split_indices"].get(split_name, [])]
            for split_name in ("train", "val", "test")
        }
        for split_name, indices in split_indices.items():
            bad = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
            if bad:
                raise ValueError(
                    f"resume split {split_name} has indices outside dataset range: "
                    f"{bad[:5]} (dataset_len={len(dataset)})"
                )
    else:
        split_indices = _build_split_indices(
            num_items=len(dataset),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    normalization_stats = None
    if args.normalization_mode == "global_train":
        if resume_ckpt is not None and resume_ckpt.get("normalization_stats") is not None:
            normalization_stats = resume_ckpt["normalization_stats"]
            print("Loaded train-split global normalization stats from checkpoint.")
        else:
            print("Computing train-split global normalization stats...")
            normalization_stats = dataset.compute_global_normalization_stats(split_indices["train"])
        dataset.set_normalization_stats(normalization_stats)

    split_datasets = {
        split_name: (Subset(dataset, indices) if indices else None)
        for split_name, indices in split_indices.items()
    }

    train_loader = _build_loader(
        split_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = (
        _build_loader(split_datasets["val"], batch_size=1, shuffle=False, num_workers=args.num_workers, device=device)
        if split_datasets["val"] is not None
        else None
    )
    test_loader = (
        _build_loader(split_datasets["test"], batch_size=1, shuffle=False, num_workers=args.num_workers, device=device)
        if split_datasets["test"] is not None
        else None
    )

    num_classes = dataset.num_classes if use_ce else 0
    model_config = _build_model_config(args=args, schema=schema, num_classes=num_classes)
    model = GraphBranchModel(**model_config).to(device)
    mask_token = None
    decoder_mask_token = None
    optimizer_params = list(model.parameters())
    if args.mask_fill == "mask_token":
        mask_token = torch.nn.Parameter(torch.zeros(int(schema["in_dim"]), device=device, dtype=torch.float32))
        optimizer_params.append(mask_token)
    if args.decoder_remask_fill == "mask_token" and float(args.decoder_remask_ratio) > 0.0:
        decoder_mask_token = torch.nn.Parameter(
            torch.zeros(int(args.hidden_dim), device=device, dtype=torch.float32)
        )
        optimizer_params.append(decoder_mask_token)
    model._input_mask_token_runtime = mask_token
    model._decoder_mask_token_runtime = decoder_mask_token
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        _optimizer_to_device(optimizer=optimizer, device=device)
        if mask_token is not None and resume_ckpt.get("mask_token_state") is not None:
            with torch.no_grad():
                mask_token.copy_(resume_ckpt["mask_token_state"].to(device=device, dtype=torch.float32))
        if decoder_mask_token is not None and resume_ckpt.get("decoder_mask_token_state") is not None:
            with torch.no_grad():
                decoder_mask_token.copy_(
                    resume_ckpt["decoder_mask_token_state"].to(device=device, dtype=torch.float32)
                )
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        if start_epoch > int(args.epochs):
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but requested total epochs={args.epochs}"
            )
        print(f"Resuming from checkpoint: {args.resume_ckpt} (next_epoch={start_epoch})")

    run_config = {
        **vars(args),
        "num_subjects": len(dataset),
        "num_classes": dataset.num_classes,
        "class_names": dataset.class_names,
        "ignore_index": IGNORE_INDEX,
        "feature_schema": schema,
        "normalization_mode": args.normalization_mode,
        "normalization": {
            "normalization_mode": args.normalization_mode,
            "xyz_norm_mode": args.xyz_norm_mode,
            "morph_norm_mode": args.morph_norm_mode,
            "graph_global_mode": args.graph_global_mode,
        },
        "normalization_stats": normalization_stats,
        "model_config": model_config,
        "recon_dim_indices": recon_dim_indices,
        "recon_feature_names": recon_feature_names,
        "masking": {
            "strategy": args.mask_strategy,
            "ratio": args.mask_ratio,
            "patch_hops": args.patch_hops,
            "patch_num_seeds": args.patch_num_seeds,
            "hybrid_patch_fraction": args.hybrid_patch_fraction,
            "region_num_components": args.region_num_components,
            "region_grow_mode": args.region_grow_mode,
            "region_seed_min_hops": args.region_seed_min_hops,
            "region_size_jitter": args.region_size_jitter,
            "multiscale_region_fraction": args.multiscale_region_fraction,
            "multiscale_patch_fraction": args.multiscale_patch_fraction,
            "mask_fill": args.mask_fill,
            "mask_random_sub_prob": args.mask_random_sub_prob,
            "decoder_remask_ratio": args.decoder_remask_ratio,
            "decoder_remask_strategy": args.decoder_remask_strategy,
            "decoder_remask_fill": args.decoder_remask_fill,
            "decoder_remask_views": args.decoder_remask_views,
            "latent_consistency_weight": args.latent_consistency_weight,
            "recon_on": args.recon_on,
        },
        "recon_weights": list(schema["default_recon_weights"]),
        "splits": {name: len(indices) for name, indices in split_indices.items()},
        "start_epoch": start_epoch,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    if start_epoch == 1 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_total_loss",
                    "train_recon_loss",
                    "train_decoder_remask_loss",
                    "train_latent_consistency_loss",
                    "train_ce_loss",
                    "val_recon_loss",
                    "selection_recon_loss",
                    "time_sec",
                ]
                + [f"recon_{name}" for name in feature_names]
            )

    best_recon, best_epoch = _load_existing_best_from_log(log_path=log_path)
    selection_split = "val" if val_loader is not None else "train"
    adjacency_cache: Dict[str, List[List[int]]] = {}

    input_keep_mask = torch.ones(int(schema["in_dim"]), dtype=torch.float32, device=device)
    input_keep_mask[recon_dim_indices] = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_decoder_remask = 0.0
        total_latent_consistency = 0.0
        total_ce = 0.0
        total_per_dim = np.zeros(int(schema["in_dim"]), dtype=np.float64)
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=140)
        for batch in pbar:
            n_steps += 1
            prepared = _prepare_batch(
                batch=batch,
                device=device,
                args=args,
                input_keep_mask=input_keep_mask,
                adjacency_cache=adjacency_cache,
                mask_token=mask_token,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(
                    lh=prepared["hemi_inputs"]["lh"],
                    rh=prepared["hemi_inputs"]["rh"],
                    graph_global=prepared.get("graph_global", None),
                )
                base_recon_loss, ce_loss, per_dim_loss = _compute_batch_recon_terms(
                    out=out,
                    hemi_batches=prepared["hemi_batches"],
                    hemi_targets=prepared["hemi_targets"],
                    hemi_masks=prepared["hemi_masks"],
                    active_dims=recon_dim_indices,
                    recon_loss_type=args.recon_loss,
                    use_ce=use_ce,
                )
                decoder_remask_loss, decoder_remask_per_dim = _compute_decoder_remask_terms(
                    model=model,
                    out=out,
                    prepared=prepared,
                    args=args,
                    active_dims=recon_dim_indices,
                    recon_loss_type=args.recon_loss,
                )
                recon_loss = base_recon_loss
                if float(args.decoder_remask_ratio) > 0.0:
                    recon_loss = 0.5 * (base_recon_loss + decoder_remask_loss)
                    per_dim_loss = 0.5 * (per_dim_loss + decoder_remask_per_dim)
                latent_consistency_loss = _compute_latent_consistency_loss(
                    model=model,
                    batch=batch,
                    device=device,
                    args=args,
                    input_keep_mask=input_keep_mask,
                    adjacency_cache=adjacency_cache,
                    mask_token=mask_token,
                    anchor_z_graph=out["z_graph"],
                )
                loss = (
                    (args.lambda_recon * recon_loss)
                    + (args.lambda_ce * ce_loss)
                    + (args.latent_consistency_weight * latent_consistency_loss)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            total_recon += float(recon_loss.detach().cpu().item())
            total_decoder_remask += float(decoder_remask_loss.detach().cpu().item())
            total_latent_consistency += float(latent_consistency_loss.detach().cpu().item())
            total_ce += float(ce_loss.detach().cpu().item())
            total_per_dim += per_dim_loss.detach().cpu().numpy().astype(np.float64)
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / n_steps:.4f}",
                    "recon": f"{total_recon / n_steps:.4f}",
                    "remask": f"{total_decoder_remask / n_steps:.4f}",
                    "ce": f"{total_ce / n_steps:.4f}",
                }
            )

        avg_loss = total_loss / max(n_steps, 1)
        avg_recon = total_recon / max(n_steps, 1)
        avg_decoder_remask = total_decoder_remask / max(n_steps, 1)
        avg_latent_consistency = total_latent_consistency / max(n_steps, 1)
        avg_ce = total_ce / max(n_steps, 1)
        avg_per_dim = total_per_dim / max(n_steps, 1)
        epoch_time = time.time() - t0

        val_metric = _evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            active_dims=recon_dim_indices,
            feature_names=feature_names,
            recon_weights=list(schema["default_recon_weights"]),
            input_keep_mask=input_keep_mask,
            use_ce=use_ce,
            seed=args.seed + epoch + 1000,
            split_name="val",
            example_limit=0,
        )
        val_recon = float(val_metric["weighted_mse"]) if val_metric["available"] else avg_recon
        selection_recon = val_recon if val_metric["available"] else avg_recon

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    avg_loss,
                    avg_recon,
                    avg_decoder_remask,
                    avg_latent_consistency,
                    avg_ce,
                    val_recon,
                    selection_recon,
                    epoch_time,
                ]
                + avg_per_dim.tolist()
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "schema": schema,
            "model_config": model_config,
            "feature_names": feature_names,
            "recon_dim_indices": recon_dim_indices,
            "num_classes": num_classes,
            "class_names": dataset.class_names,
            "ignore_index": IGNORE_INDEX,
            "split_indices": split_indices,
            "normalization_stats": normalization_stats,
        }
        if mask_token is not None:
            ckpt["mask_token_state"] = mask_token.detach().cpu()
        if decoder_mask_token is not None:
            ckpt["decoder_mask_token_state"] = decoder_mask_token.detach().cpu()

        torch.save(ckpt, os.path.join(args.out_dir, "ckpt_last.pt"))

        if selection_recon < best_recon:
            best_recon = selection_recon
            best_epoch = epoch
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best_recon.pt"))

        print(
            f"[epoch {epoch}] train_total={avg_loss:.6f} train_recon={avg_recon:.6f} "
            f"val_recon={val_recon:.6f} remask={avg_decoder_remask:.6f} time={epoch_time:.1f}s"
        )

    best_ckpt_path = os.path.join(args.out_dir, "ckpt_best_recon.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    model = model.to(device)
    if mask_token is not None and best_ckpt.get("mask_token_state") is not None:
        with torch.no_grad():
            mask_token.copy_(best_ckpt["mask_token_state"].to(device=device, dtype=torch.float32))
    model._input_mask_token_runtime = mask_token
    if decoder_mask_token is not None and best_ckpt.get("decoder_mask_token_state") is not None:
        with torch.no_grad():
            decoder_mask_token.copy_(best_ckpt["decoder_mask_token_state"].to(device=device, dtype=torch.float32))
    model._decoder_mask_token_runtime = decoder_mask_token

    val_artifacts = _evaluate_loader(
        model=model,
        loader=val_loader,
        device=device,
        args=args,
        active_dims=recon_dim_indices,
        feature_names=feature_names,
        recon_weights=list(schema["default_recon_weights"]),
        input_keep_mask=input_keep_mask,
        use_ce=use_ce,
        seed=args.seed + 5000,
        split_name="val",
        example_limit=max(0, int(args.eval_examples)),
    )
    test_artifacts = _evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        args=args,
        active_dims=recon_dim_indices,
        feature_names=feature_names,
        recon_weights=list(schema["default_recon_weights"]),
        input_keep_mask=input_keep_mask,
        use_ce=use_ce,
        seed=args.seed + 6000,
        split_name="test",
        example_limit=max(0, int(args.eval_examples)),
    )

    split_metrics = {
        "val": {k: v for k, v in val_artifacts.items() if k != "examples"},
        "test": {k: v for k, v in test_artifacts.items() if k != "examples"},
    }
    write_json(
        os.path.join(args.out_dir, "recon_metrics.json"),
        {
            "best_epoch": best_epoch,
            "best_selection_split": selection_split,
            "best_selection_recon_loss": best_recon,
            "feature_names": feature_names,
            "recon_feature_names": recon_feature_names,
            "splits": split_metrics,
        },
    )
    write_per_dim_recon_csv(
        os.path.join(args.out_dir, "per_dim_recon_mse.csv"),
        feature_names=feature_names,
        split_metrics=split_metrics,
    )
    write_json(
        os.path.join(args.out_dir, "masking_summary.json"),
        {
            "strategy": args.mask_strategy,
            "requested_mask_ratio": args.mask_ratio,
            "patch_hops": args.patch_hops,
            "patch_num_seeds": args.patch_num_seeds,
            "hybrid_patch_fraction": args.hybrid_patch_fraction,
            "region_num_components": args.region_num_components,
            "region_grow_mode": args.region_grow_mode,
            "region_seed_min_hops": args.region_seed_min_hops,
            "region_size_jitter": args.region_size_jitter,
            "multiscale_region_fraction": args.multiscale_region_fraction,
            "multiscale_patch_fraction": args.multiscale_patch_fraction,
            "mask_fill": args.mask_fill,
            "mask_random_sub_prob": args.mask_random_sub_prob,
            "decoder_remask_ratio": args.decoder_remask_ratio,
            "decoder_remask_strategy": args.decoder_remask_strategy,
            "decoder_remask_fill": args.decoder_remask_fill,
            "decoder_remask_views": args.decoder_remask_views,
            "latent_consistency_weight": args.latent_consistency_weight,
            "recon_on": args.recon_on,
            "active_feature_names": recon_feature_names,
            "splits": {
                "val": {
                    "available": val_artifacts["available"],
                    "total_vertices": val_artifacts["total_vertices"],
                    "masked_vertices": val_artifacts["masked_vertices"],
                    "realized_mask_ratio": val_artifacts["realized_mask_ratio"],
                    "mean_num_connected_components": val_artifacts["mean_num_connected_components"],
                    "mean_largest_component_size": val_artifacts["mean_largest_component_size"],
                    "mean_component_size": val_artifacts["mean_component_size"],
                    "mean_largest_component_ratio": val_artifacts["mean_largest_component_ratio"],
                    "weighted_mse": val_artifacts["weighted_mse"],
                    "thickness_mse": val_artifacts["thickness_mse"],
                    "curvature_mse": val_artifacts["curvature_mse"],
                },
                "test": {
                    "available": test_artifacts["available"],
                    "total_vertices": test_artifacts["total_vertices"],
                    "masked_vertices": test_artifacts["masked_vertices"],
                    "realized_mask_ratio": test_artifacts["realized_mask_ratio"],
                    "mean_num_connected_components": test_artifacts["mean_num_connected_components"],
                    "mean_largest_component_size": test_artifacts["mean_largest_component_size"],
                    "mean_component_size": test_artifacts["mean_component_size"],
                    "mean_largest_component_ratio": test_artifacts["mean_largest_component_ratio"],
                    "weighted_mse": test_artifacts["weighted_mse"],
                    "thickness_mse": test_artifacts["thickness_mse"],
                    "curvature_mse": test_artifacts["curvature_mse"],
                },
            },
        },
    )
    save_recon_examples(
        os.path.join(args.out_dir, "recon_examples"),
        examples=val_artifacts["examples"] + test_artifacts["examples"],
        feature_names=feature_names,
    )

    print(f"Training done. logs={log_path}")
    print(f"Best reconstruction loss={best_recon:.6f} ({selection_split}, epoch={best_epoch})")


if __name__ == "__main__":
    main()
