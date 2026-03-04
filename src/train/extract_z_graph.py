"""Extract subject-level z_graph embeddings from trained graph branch."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_surface import SurfaceSubjectDataset
from src.models.adgcn import GraphBranchModel


def collate_subject(batch):
    return batch[0]


def to_device_hemi(hemi_batch, device: torch.device):
    return {
        "x": hemi_batch["X"].to(device=device, dtype=torch.float32),
        "edge_index": hemi_batch["edge_index"].to(device=device, dtype=torch.long),
        "valid_mask": hemi_batch["mask_valid"].to(device=device, dtype=torch.bool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract z_graph embeddings")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--res", type=str, default="fsaverage6")
    parser.add_argument("--edge_cache_dir", type=str, default="cache/templates")
    parser.add_argument("--out_dir", type=str, default="z_graph_cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt.get("class_names", None)
    num_classes = int(ckpt.get("num_classes", 0))

    dataset = SurfaceSubjectDataset(
        manifest_csv=args.manifest,
        res=args.res,
        random_resolution=False,
        edge_cache_dir=args.edge_cache_dir,
        class_names=class_names,
        in_memory_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_subject,
        pin_memory=(device.type == "cuda"),
    )

    model = GraphBranchModel(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract_z_graph", ncols=120):
            sid = str(batch["sid"])

            lh = to_device_hemi(batch["lh"], device=device)
            rh = to_device_hemi(batch["rh"], device=device)

            out = model(lh=lh, rh=rh)
            z_graph = out["z_graph"].detach().cpu().numpy().astype(np.float32)

            out_path = os.path.join(args.out_dir, f"{sid}.npy")
            np.save(out_path, z_graph)

    print(f"Saved z_graph embeddings to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
