"""Extract subject-level z_graph embeddings and PCA summaries."""

from __future__ import annotations

import argparse
import csv
import os
import struct
import zlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_surface import SurfaceSubjectDataset
from src.data.feature_schema import get_feature_schema
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
    parser.add_argument("--res", type=str, default=None)
    parser.add_argument("--edge_cache_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="z_graph_cache")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _resolve_model_config(ckpt: Dict) -> Dict[str, object]:
    ckpt_args = ckpt.get("args", {}) or {}
    if ckpt.get("model_config"):
        model_config = dict(ckpt["model_config"])
        return {
            "input_mode": str(model_config["input_mode"]),
            "in_dim": int(model_config["in_dim"]),
            "geo_dim": int(model_config["geo_dim"]),
            "morph_dim": int(model_config["morph_dim"]),
            "hidden_dim": int(model_config["hidden_dim"]),
            "dims": [int(x) for x in model_config["dims"]],
            "num_classes": int(model_config["num_classes"]),
            "dropout": float(model_config["dropout"]),
            "fuse_mode": str(model_config["fuse_mode"]),
        }

    input_mode = str(ckpt_args.get("input_mode", "baseline8"))
    schema = ckpt.get("schema") or get_feature_schema(input_mode)
    hidden_dim = int(ckpt_args.get("hidden_dim", 16))
    dims = ckpt_args.get("dims", [16, 32, 64, 128, 64, 32, 16])
    return {
        "input_mode": input_mode,
        "in_dim": int(schema["in_dim"]),
        "geo_dim": int(schema["geo_dim"]),
        "morph_dim": int(schema["morph_dim"]),
        "hidden_dim": hidden_dim,
        "dims": [int(x) for x in dims],
        "num_classes": int(ckpt.get("num_classes", 0)),
        "dropout": float(ckpt_args.get("dropout", 0.1)),
        "fuse_mode": str(ckpt_args.get("fuse_mode", "sum")),
    }


def _resolve_value(cli_value, ckpt_args: Dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def _compute_pca_2d(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}")
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32)

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    try:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=n_components).fit_transform(embeddings).astype(np.float32)
    except ImportError:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords = (u[:, :n_components] * s[:n_components]).astype(np.float32)

    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return coords[:, :2].astype(np.float32, copy=False)


def _write_subject_ids(path: str, subject_ids: Iterable[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid"])
        for sid in subject_ids:
            writer.writerow([sid])


def _write_embeddings_csv(path: str, subject_ids: List[str], embeddings: np.ndarray) -> None:
    dim = embeddings.shape[1]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid"] + [f"z{i:03d}" for i in range(dim)])
        for sid, row in zip(subject_ids, embeddings):
            writer.writerow([sid] + row.astype(float).tolist())


def _write_pca_csv(path: str, subject_ids: List[str], pca_xy: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid", "pc1", "pc2"])
        for sid, row in zip(subject_ids, pca_xy):
            writer.writerow([sid, float(row[0]), float(row[1])])


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)


def _save_png_rgb(path: str, image: np.ndarray) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    with open(path, "wb") as f:
        f.write(png)


def _draw_disk(image: np.ndarray, x: int, y: int, radius: int, color: Tuple[int, int, int]) -> None:
    height, width, _ = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dx * dx) + (dy * dy) > (radius * radius):
                continue
            xx = x + dx
            yy = y + dy
            if 0 <= xx < width and 0 <= yy < height:
                image[yy, xx] = color


def _save_basic_scatter(path: str, pca_xy: np.ndarray) -> None:
    width, height = 1000, 800
    margin = 60
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[margin: height - margin, margin] = (180, 180, 180)
    image[margin: height - margin, width - margin] = (180, 180, 180)
    image[margin, margin: width - margin] = (180, 180, 180)
    image[height - margin, margin: width - margin] = (180, 180, 180)

    if pca_xy.shape[0] > 0:
        xs = pca_xy[:, 0]
        ys = pca_xy[:, 1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        if abs(xmax - xmin) < 1e-6:
            xmin -= 1.0
            xmax += 1.0
        if abs(ymax - ymin) < 1e-6:
            ymin -= 1.0
            ymax += 1.0

        for x_val, y_val in zip(xs, ys):
            x_norm = (float(x_val) - xmin) / (xmax - xmin)
            y_norm = (float(y_val) - ymin) / (ymax - ymin)
            x_px = int(round(margin + x_norm * (width - (2 * margin))))
            y_px = int(round((height - margin) - y_norm * (height - (2 * margin))))
            _draw_disk(image, x_px, y_px, radius=4, color=(32, 92, 196))

    _save_png_rgb(path=path, image=image)


def _save_pca_scatter(path: str, pca_xy: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pca_xy[:, 0], pca_xy[:, 1], s=24, c="#205cc4", alpha=0.9)
        ax.set_title("z_graph PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    except Exception:
        _save_basic_scatter(path=path, pca_xy=pca_xy)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    model_config = _resolve_model_config(ckpt)
    class_names = ckpt.get("class_names", None)

    res = _resolve_value(args.res, ckpt_args, "res", "fsaverage6")
    edge_cache_dir = _resolve_value(args.edge_cache_dir, ckpt_args, "edge_cache_dir", "cache/templates")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    dataset = SurfaceSubjectDataset(
        manifest_csv=args.manifest,
        res=res,
        random_resolution=False,
        edge_cache_dir=edge_cache_dir,
        class_names=class_names,
        in_memory_cache=False,
        input_mode=model_config["input_mode"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_subject,
        pin_memory=(device.type == "cuda"),
    )

    model = GraphBranchModel(**model_config)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    subject_ids: List[str] = []
    embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract_z_graph", ncols=140):
            sid = str(batch["sid"])

            lh = to_device_hemi(batch["lh"], device=device)
            rh = to_device_hemi(batch["rh"], device=device)

            out = model(lh=lh, rh=rh)
            z_graph = out["z_graph"].detach().cpu().numpy().astype(np.float32).reshape(-1)

            np.save(os.path.join(args.out_dir, f"{sid}.npy"), z_graph)
            subject_ids.append(sid)
            embeddings.append(z_graph)

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    embedding_matrix = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    pca_xy = _compute_pca_2d(embedding_matrix)

    np.save(os.path.join(args.out_dir, "embeddings.npy"), embedding_matrix)
    _write_subject_ids(os.path.join(args.out_dir, "subject_ids.csv"), subject_ids)
    _write_embeddings_csv(
        os.path.join(args.out_dir, "embeddings_with_sid.csv"),
        subject_ids=subject_ids,
        embeddings=embedding_matrix,
    )
    _write_pca_csv(os.path.join(args.out_dir, "pca_2d.csv"), subject_ids=subject_ids, pca_xy=pca_xy)
    _save_pca_scatter(os.path.join(args.out_dir, "pca_scatter.png"), pca_xy=pca_xy)

    print(f"Saved z_graph embeddings to: {os.path.abspath(args.out_dir)}")
    print(f"embeddings.npy shape: {tuple(embedding_matrix.shape)}")


if __name__ == "__main__":
    main()
