"""Train v1 cortical graph branch (MFM + optional label CE)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_surface import IGNORE_INDEX, SurfaceSubjectDataset
from src.models.adgcn import GraphBranchModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_subject(batch):
    return batch[0]


def sample_vertex_mask(n_verts: int, ratio: float, device: torch.device) -> torch.Tensor:
    if ratio <= 0.0:
        mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
        ridx = torch.randint(0, n_verts, (1,), device=device)
        mask[ridx] = True
        return mask

    if ratio >= 1.0:
        return torch.ones(n_verts, dtype=torch.bool, device=device)

    mask = torch.rand(n_verts, device=device) < ratio
    if int(mask.sum().item()) == 0:
        ridx = torch.randint(0, n_verts, (1,), device=device)
        mask[ridx] = True
    return mask


def to_device_hemi(hemi_batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "X": hemi_batch["X"].to(device=device, dtype=torch.float32),
        "edge_index": hemi_batch["edge_index"].to(device=device, dtype=torch.long),
        "y": hemi_batch["y"].to(device=device, dtype=torch.long),
        "mask_valid": hemi_batch["mask_valid"].to(device=device, dtype=torch.bool),
        "thickness_gt": hemi_batch["thickness_gt"].to(device=device, dtype=torch.float32),
        "curv_gt": hemi_batch["curv_gt"].to(device=device, dtype=torch.float32),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ADGCN graph branch v1")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--res", type=str, default="fsaverage6")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--alpha_ce", type=float, default=0.2)
    parser.add_argument("--use_ce", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--edge_cache_dir", type=str, default="cache/templates")
    parser.add_argument("--random_resolution", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("This v1 trainer expects batch_size=1 (subject-level batches).")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    use_ce = bool(args.use_ce)

    dataset = SurfaceSubjectDataset(
        manifest_csv=args.manifest,
        res=args.res,
        random_resolution=bool(args.random_resolution),
        edge_cache_dir=args.edge_cache_dir,
        seed=args.seed,
        in_memory_cache=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_subject,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = dataset.num_classes if use_ce else 0
    model = GraphBranchModel(num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_config = {
        **vars(args),
        "num_subjects": len(dataset),
        "num_classes": dataset.num_classes,
        "class_names": dataset.class_names,
        "ignore_index": IGNORE_INDEX,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "mfm", "ce", "time_sec"])

    best_mfm = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_mfm = 0.0
        total_ce = 0.0
        n_steps = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", ncols=120)
        for batch in pbar:
            n_steps += 1

            lh_b = to_device_hemi(batch["lh"], device=device)
            rh_b = to_device_hemi(batch["rh"], device=device)

            hemi_inputs = {}
            targets = {}
            masks = {}

            for hemi_name, hemi_b in (("lh", lh_b), ("rh", rh_b)):
                x = hemi_b["X"]
                n_verts = x.shape[0]
                mask = sample_vertex_mask(n_verts=n_verts, ratio=args.mask_ratio, device=device)

                x_masked = x.clone()
                # Only mask morphometry channels; keep xyz channels intact.
                x_masked[mask, 6:8] = 0.0

                target = torch.stack([hemi_b["thickness_gt"], hemi_b["curv_gt"]], dim=-1)
                masks[hemi_name] = mask
                targets[hemi_name] = target
                hemi_inputs[hemi_name] = {
                    "x": x_masked,
                    "edge_index": hemi_b["edge_index"],
                    "valid_mask": hemi_b["mask_valid"],
                }

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(lh=hemi_inputs["lh"], rh=hemi_inputs["rh"])

                mfm_losses = []
                ce_losses = []

                for hemi_name, hemi_b in (("lh", lh_b), ("rh", rh_b)):
                    pred = out[hemi_name]["morph_pred"]
                    target = targets[hemi_name]
                    mask = masks[hemi_name]
                    mfm_losses.append(F.huber_loss(pred[mask], target[mask], reduction="mean", delta=1.0))

                    if use_ce:
                        logits = out[hemi_name].get("label_logits", None)
                        if logits is None:
                            raise RuntimeError("use_ce=1 but model has no label head")
                        ce_losses.append(
                            F.cross_entropy(logits, hemi_b["y"], ignore_index=IGNORE_INDEX)
                        )

                mfm_loss = torch.stack(mfm_losses).mean()
                if use_ce:
                    ce_loss = torch.stack(ce_losses).mean()
                else:
                    ce_loss = torch.zeros((), device=device)

                loss = mfm_loss + args.alpha_ce * ce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            total_mfm += float(mfm_loss.detach().cpu().item())
            total_ce += float(ce_loss.detach().cpu().item())
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / n_steps:.4f}",
                    "mfm": f"{total_mfm / n_steps:.4f}",
                    "ce": f"{total_ce / n_steps:.4f}",
                }
            )

        avg_loss = total_loss / max(n_steps, 1)
        avg_mfm = total_mfm / max(n_steps, 1)
        avg_ce = total_ce / max(n_steps, 1)
        epoch_time = time.time() - t0

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_mfm, avg_ce, epoch_time])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
            "class_names": dataset.class_names,
            "ignore_index": IGNORE_INDEX,
        }

        last_path = os.path.join(args.out_dir, "ckpt_last.pt")
        torch.save(ckpt, last_path)

        if avg_mfm < best_mfm:
            best_mfm = avg_mfm
            best_path = os.path.join(args.out_dir, "ckpt_best_mfm.pt")
            torch.save(ckpt, best_path)

        print(
            f"[epoch {epoch}] loss={avg_loss:.6f} mfm={avg_mfm:.6f} "
            f"ce={avg_ce:.6f} time={epoch_time:.1f}s"
        )

    print(f"Training done. logs={log_path}")
    print(f"Best MFM={best_mfm:.6f}")


if __name__ == "__main__":
    main()
