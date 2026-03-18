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
from src.train.masking import build_neighbor_list, sample_vertex_mask
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

    parser.add_argument("--input_mode", type=str, default="main5", choices=list_input_modes())
    parser.add_argument("--mask_strategy", type=str, default="hybrid", choices=["random", "patch", "hybrid"])
    parser.add_argument("--mask_ratio", type=float, default=0.35)
    parser.add_argument("--patch_hops", type=int, default=2)
    parser.add_argument("--patch_num_seeds", type=int, default=16)
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_subject,
        pin_memory=(device.type == "cuda"),
    )


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


def _prepare_batch(
    batch: Dict,
    device: torch.device,
    args: argparse.Namespace,
    input_keep_mask: torch.Tensor,
    adjacency_cache: Dict[str, List[List[int]]],
) -> Dict[str, object]:
    if batch["res"] not in adjacency_cache:
        adjacency_cache[batch["res"]] = build_neighbor_list(
            edge_index=batch["lh"]["edge_index"],
            num_nodes=int(batch["lh"]["X"].shape[0]),
        )
    neighbors = adjacency_cache[batch["res"]]

    hemi_batches = {
        "lh": to_device_hemi(batch["lh"], device=device),
        "rh": to_device_hemi(batch["rh"], device=device),
    }
    hemi_inputs = {}
    hemi_targets = {}
    hemi_masks = {}
    hemi_masked_inputs = {}

    for hemi_name in ("lh", "rh"):
        hemi_b = hemi_batches[hemi_name]
        vertex_mask = sample_vertex_mask(
            strategy=args.mask_strategy,
            n_verts=int(hemi_b["X"].shape[0]),
            ratio=args.mask_ratio,
            device=device,
            edge_index=batch[hemi_name]["edge_index"],
            neighbors=neighbors,
            patch_hops=args.patch_hops,
            patch_num_seeds=args.patch_num_seeds,
        )

        x_masked = hemi_b["X"].clone()
        x_masked[vertex_mask] = x_masked[vertex_mask] * input_keep_mask.unsqueeze(0)

        hemi_inputs[hemi_name] = {
            "x": x_masked,
            "edge_index": hemi_b["edge_index"],
            "valid_mask": hemi_b["mask_valid"],
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
            "weighted_mse": 0.0,
            "unweighted_mse": 0.0,
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
            )
            out = model(lh=prepared["hemi_inputs"]["lh"], rh=prepared["hemi_inputs"]["rh"])
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

    return {
        "available": True,
        "split": split_name,
        "num_subjects": len(loader.dataset),
        "num_hemis": total_hemis,
        "total_vertices": total_vertices,
        "masked_vertices": total_masked_vertices,
        "realized_mask_ratio": realized_ratio,
        "weighted_mse": weighted_mse,
        "unweighted_mse": unweighted_mse,
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

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("This trainer expects batch_size=1 (subject-level batches).")

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
    )
    if use_ce and dataset.num_classes <= 0:
        raise ValueError(
            "use_ce=1 requires parcel-level annotation classes. "
            "The current manifest only provides cortex masks (.label), so use --use_ce 0."
        )

    split_indices = _build_split_indices(
        num_items=len(dataset),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_config = {
        **vars(args),
        "num_subjects": len(dataset),
        "num_classes": dataset.num_classes,
        "class_names": dataset.class_names,
        "ignore_index": IGNORE_INDEX,
        "feature_schema": schema,
        "model_config": model_config,
        "recon_dim_indices": recon_dim_indices,
        "recon_feature_names": recon_feature_names,
        "masking": {
            "strategy": args.mask_strategy,
            "ratio": args.mask_ratio,
            "patch_hops": args.patch_hops,
            "patch_num_seeds": args.patch_num_seeds,
            "recon_on": args.recon_on,
        },
        "recon_weights": list(schema["default_recon_weights"]),
        "splits": {name: len(indices) for name, indices in split_indices.items()},
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_total_loss",
                "train_recon_loss",
                "train_ce_loss",
                "val_recon_loss",
                "selection_recon_loss",
                "time_sec",
            ]
            + [f"recon_{name}" for name in feature_names]
        )

    best_recon = float("inf")
    best_epoch = 0
    selection_split = "val" if val_loader is not None else "train"
    adjacency_cache: Dict[str, List[List[int]]] = {}

    input_keep_mask = torch.ones(int(schema["in_dim"]), dtype=torch.float32, device=device)
    input_keep_mask[recon_dim_indices] = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_recon = 0.0
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
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(lh=prepared["hemi_inputs"]["lh"], rh=prepared["hemi_inputs"]["rh"])
                recon_loss, ce_loss, per_dim_loss = _compute_batch_recon_terms(
                    out=out,
                    hemi_batches=prepared["hemi_batches"],
                    hemi_targets=prepared["hemi_targets"],
                    hemi_masks=prepared["hemi_masks"],
                    active_dims=recon_dim_indices,
                    recon_loss_type=args.recon_loss,
                    use_ce=use_ce,
                )
                loss = (args.lambda_recon * recon_loss) + (args.lambda_ce * ce_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            total_recon += float(recon_loss.detach().cpu().item())
            total_ce += float(ce_loss.detach().cpu().item())
            total_per_dim += per_dim_loss.detach().cpu().numpy().astype(np.float64)
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / n_steps:.4f}",
                    "recon": f"{total_recon / n_steps:.4f}",
                    "ce": f"{total_ce / n_steps:.4f}",
                }
            )

        avg_loss = total_loss / max(n_steps, 1)
        avg_recon = total_recon / max(n_steps, 1)
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
                [epoch, avg_loss, avg_recon, avg_ce, val_recon, selection_recon, epoch_time]
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
        }

        torch.save(ckpt, os.path.join(args.out_dir, "ckpt_last.pt"))

        if selection_recon < best_recon:
            best_recon = selection_recon
            best_epoch = epoch
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best_recon.pt"))

        print(
            f"[epoch {epoch}] train_total={avg_loss:.6f} train_recon={avg_recon:.6f} "
            f"val_recon={val_recon:.6f} time={epoch_time:.1f}s"
        )

    best_ckpt_path = os.path.join(args.out_dir, "ckpt_best_recon.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    model = model.to(device)

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
            "recon_on": args.recon_on,
            "active_feature_names": recon_feature_names,
            "splits": {
                "val": {
                    "available": val_artifacts["available"],
                    "total_vertices": val_artifacts["total_vertices"],
                    "masked_vertices": val_artifacts["masked_vertices"],
                    "realized_mask_ratio": val_artifacts["realized_mask_ratio"],
                },
                "test": {
                    "available": test_artifacts["available"],
                    "total_vertices": test_artifacts["total_vertices"],
                    "masked_vertices": test_artifacts["masked_vertices"],
                    "realized_mask_ratio": test_artifacts["realized_mask_ratio"],
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
