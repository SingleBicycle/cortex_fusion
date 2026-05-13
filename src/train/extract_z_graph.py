"""Extract subject-level z_graph embeddings and lightweight diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
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


def to_device_hemi(hemi_batch, device: torch.device, res_name: str):
    return {
        "x": hemi_batch["X"].to(device=device, dtype=torch.float32),
        "edge_index": hemi_batch["edge_index"].to(device=device, dtype=torch.long),
        "valid_mask": hemi_batch["mask_valid"].to(device=device, dtype=torch.bool),
        "res_name": str(res_name),
    }


def to_device_graph_global(batch, device: torch.device) -> torch.Tensor | None:
    if "graph_global" not in batch:
        return None
    return batch["graph_global"].to(device=device, dtype=torch.float32).view(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract z_graph embeddings")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--res", type=str, default=None)
    parser.add_argument("--edge_cache_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="z_graph_cache")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_umap", type=int, default=0, choices=[0, 1])
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
            "backbone_type": str(model_config.get("backbone_type", "gcn")),
            "pool_mode": str(model_config.get("pool_mode", "mean")),
            "hierarchy_dir": model_config.get("hierarchy_dir", None),
            "edge_cache_dir": model_config.get("edge_cache_dir", None),
            "max_hierarchy_levels": int(model_config.get("max_hierarchy_levels", 2)),
            "graph_unet_depth": int(model_config.get("graph_unet_depth", 2)),
            "graph_unet_pool_ratios": [
                float(x) for x in model_config.get("graph_unet_pool_ratios", [0.8, 0.8])
            ],
            "graph_global_dim": int(model_config.get("graph_global_dim", 0)),
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
        "backbone_type": str(ckpt_args.get("backbone_type", "gcn")),
        "pool_mode": str(ckpt_args.get("pool_mode", "mean")),
        "hierarchy_dir": ckpt_args.get("hierarchy_dir", None),
        "edge_cache_dir": ckpt_args.get("edge_cache_dir", None),
        "max_hierarchy_levels": int(ckpt_args.get("max_hierarchy_levels", 2)),
        "graph_unet_depth": int(ckpt_args.get("graph_unet_depth", 2)),
        "graph_unet_pool_ratios": [
            float(x) for x in ckpt_args.get("graph_unet_pool_ratios", [0.8, 0.8])
        ],
        "graph_global_dim": 5 if str(ckpt_args.get("graph_global_mode", "none")) == "summary5" else 0,
    }


def _resolve_value(cli_value, ckpt_args: Dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def _compute_pca_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}")
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(embeddings).astype(np.float32)
        explained = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    except ImportError:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords = (u[:, :n_components] * s[:n_components]).astype(np.float32)
        denom = np.square(s).sum()
        explained = (
            (np.square(s[:n_components]) / max(float(denom), 1e-12)).astype(np.float32, copy=False)
            if denom > 0
            else np.zeros(n_components, dtype=np.float32)
        )

    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    if explained.shape[0] < 2:
        explained = np.pad(explained, (0, 2 - explained.shape[0]), mode="constant")
    return coords[:, :2].astype(np.float32, copy=False), explained[:2].astype(np.float32, copy=False)


def _compute_optional_umap_2d(embeddings: np.ndarray, skip_umap: bool) -> Tuple[np.ndarray | None, str | None]:
    if skip_umap:
        return None, "skip_umap=1"
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, embeddings.shape[0] - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings).astype(np.float32, copy=False)
        return coords, None
    except Exception as exc:
        return None, str(exc)


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


def _write_subject_stats_csv(path: str, subject_ids: List[str], subject_stats: Dict[str, np.ndarray]) -> None:
    keys = list(subject_stats.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid"] + keys)
        for idx, sid in enumerate(subject_ids):
            writer.writerow([sid] + [float(subject_stats[key][idx]) for key in keys])


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


def _save_scatter(path: str, xy: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=24, c="#205cc4", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    except Exception:
        _save_basic_scatter(path=path, pca_xy=xy)


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    power = np.square(s).astype(np.float64, copy=False)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probs = power / total
    probs = probs[probs > 1e-12]
    entropy = -float((probs * np.log(probs)).sum())
    return float(np.exp(entropy))


def _pairwise_distance_stats(embeddings: np.ndarray, max_points: int = 512) -> Dict[str, float]:
    if embeddings.shape[0] <= 1:
        return {"mean": 0.0, "std": 0.0}
    rng = np.random.default_rng(42)
    if embeddings.shape[0] > max_points:
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt(np.maximum(np.square(diff).sum(axis=-1), 0.0))
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    return {
        "mean": float(tri.mean()) if tri.size else 0.0,
        "std": float(tri.std()) if tri.size else 0.0,
    }


def _build_stats_payload(
    embedding_matrix: np.ndarray,
    subject_stats: Dict[str, np.ndarray],
    model_config: Dict[str, object],
    ckpt_args: Dict[str, object],
    skip_umap: bool,
) -> tuple[Dict[str, object], np.ndarray, np.ndarray | None]:
    pca_xy, pca_explained = _compute_pca_2d(embedding_matrix)
    umap_xy, umap_error = _compute_optional_umap_2d(embedding_matrix, skip_umap=skip_umap)
    norms = np.linalg.norm(embedding_matrix, axis=1)
    per_dim_std = embedding_matrix.std(axis=0)
    pairwise_stats = _pairwise_distance_stats(embedding_matrix)
    stats_payload = {
        "num_subjects": int(embedding_matrix.shape[0]),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "pool_mode": str(model_config.get("pool_mode", "mean")),
        "backbone_type": str(model_config.get("backbone_type", "gcn")),
        "graph_global_dim": int(model_config.get("graph_global_dim", 0)),
        "normalization_mode": str(ckpt_args.get("normalization_mode", "per_hemi")),
        "xyz_norm_mode": str(ckpt_args.get("xyz_norm_mode", "subject_rms")),
        "morph_norm_mode": str(ckpt_args.get("morph_norm_mode", "subject_zscore")),
        "graph_global_mode": str(ckpt_args.get("graph_global_mode", "none")),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_min": float(per_dim_std.min()),
        "per_dim_std_max": float(per_dim_std.max()),
        "effective_rank": _effective_rank(embedding_matrix),
        "pairwise_distance_mean": pairwise_stats["mean"],
        "pairwise_distance_std": pairwise_stats["std"],
        "pca_explained_variance_ratio": [float(x) for x in pca_explained.tolist()],
        "pc1_curvature_abs_mean_corr": _safe_corrcoef(pca_xy[:, 0], subject_stats["curvature_abs_mean"]),
        "pc2_curvature_abs_mean_corr": _safe_corrcoef(pca_xy[:, 1], subject_stats["curvature_abs_mean"]),
        "pc1_thickness_mean_corr": _safe_corrcoef(pca_xy[:, 0], subject_stats["thickness_mean"]),
        "pc2_thickness_mean_corr": _safe_corrcoef(pca_xy[:, 1], subject_stats["thickness_mean"]),
        "pc1_xyz_scale_corr": _safe_corrcoef(pca_xy[:, 0], subject_stats["xyz_scale"]),
        "pc2_xyz_scale_corr": _safe_corrcoef(pca_xy[:, 1], subject_stats["xyz_scale"]),
        "umap_available": bool(umap_xy is not None),
        "umap_error": umap_error,
    }
    return stats_payload, pca_xy, umap_xy


def _write_embedding_outputs(
    out_dir: str,
    prefix: str,
    scatter_title: str,
    subject_ids: List[str],
    embedding_matrix: np.ndarray,
    subject_stats: Dict[str, np.ndarray],
    model_config: Dict[str, object],
    ckpt_args: Dict[str, object],
    skip_umap: bool,
) -> Dict[str, object]:
    stats_payload, pca_xy, umap_xy = _build_stats_payload(
        embedding_matrix=embedding_matrix,
        subject_stats=subject_stats,
        model_config=model_config,
        ckpt_args=ckpt_args,
        skip_umap=skip_umap,
    )
    np.save(os.path.join(out_dir, f"{prefix}embeddings.npy"), embedding_matrix)
    _write_embeddings_csv(
        os.path.join(out_dir, f"{prefix}embeddings_with_sid.csv"),
        subject_ids=subject_ids,
        embeddings=embedding_matrix,
    )
    _write_pca_csv(
        os.path.join(out_dir, f"{prefix}pca_2d.csv"),
        subject_ids=subject_ids,
        pca_xy=pca_xy,
    )
    _save_scatter(
        os.path.join(out_dir, f"{prefix}pca_scatter.png"),
        xy=pca_xy,
        title=scatter_title,
    )
    if umap_xy is not None:
        _write_pca_csv(
            os.path.join(out_dir, f"{prefix}umap_2d.csv"),
            subject_ids=subject_ids,
            pca_xy=umap_xy,
        )
        _save_scatter(
            os.path.join(out_dir, f"{prefix}umap_scatter.png"),
            xy=umap_xy,
            title=scatter_title.replace("PCA", "UMAP"),
        )
    with open(os.path.join(out_dir, f"{prefix}embedding_stats.json"), "w") as f:
        json.dump(stats_payload, f, indent=2)
    return {
        "stats": stats_payload,
        "pca_xy": pca_xy,
        "umap_xy": umap_xy,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    model_config = _resolve_model_config(ckpt)
    class_names = ckpt.get("class_names", None)

    res = _resolve_value(args.res, ckpt_args, "res", "fsaverage6")
    edge_cache_dir = _resolve_value(args.edge_cache_dir, ckpt_args, "edge_cache_dir", "cache/templates")
    normalization_mode = str(ckpt_args.get("normalization_mode", "per_hemi"))
    normalization_stats = ckpt.get("normalization_stats", None)
    if normalization_mode == "global_train" and normalization_stats is None:
        raise RuntimeError("checkpoint was trained with global_train normalization but has no normalization_stats")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    dataset = SurfaceSubjectDataset(
        manifest_csv=args.manifest,
        res=res,
        random_resolution=False,
        edge_cache_dir=edge_cache_dir,
        class_names=class_names,
        in_memory_cache=False,
        input_mode=model_config["input_mode"],
        normalization_mode=normalization_mode,
        normalization_stats=normalization_stats,
        xyz_norm_mode=str(ckpt_args.get("xyz_norm_mode", "subject_rms")),
        morph_norm_mode=str(ckpt_args.get("morph_norm_mode", "subject_zscore")),
        graph_global_mode=str(ckpt_args.get("graph_global_mode", "none")),
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
    model_state = dict(ckpt["model_state"])
    model_state.pop("_input_mask_token_runtime", None)
    model_state.pop("_decoder_mask_token_runtime", None)
    model.load_state_dict(model_state, strict=True)
    model = model.to(device)
    model.eval()

    subject_ids: List[str] = []
    post_head_embeddings: List[np.ndarray] = []
    pre_head_embeddings: List[np.ndarray] = []
    thickness_mean: List[float] = []
    thickness_std: List[float] = []
    curv_mean: List[float] = []
    curv_std: List[float] = []
    curv_abs_mean: List[float] = []
    xyz_scale: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract_z_graph", ncols=140):
            sid = str(batch["sid"])

            lh = to_device_hemi(batch["lh"], device=device, res_name=batch["res"])
            rh = to_device_hemi(batch["rh"], device=device, res_name=batch["res"])
            graph_global = to_device_graph_global(batch, device=device)

            out = model(lh=lh, rh=rh, graph_global=graph_global)
            pre_head_parts = [out["lh"]["z_hemi"], out["rh"]["z_hemi"]]
            if int(model_config.get("graph_global_dim", 0)) > 0:
                if graph_global is None:
                    raise RuntimeError("graph_global_dim > 0 but batch did not provide graph_global")
                if getattr(model, "global_encoder", None) is None:
                    raise RuntimeError("graph_global_dim > 0 but model.global_encoder is not initialized")
                encoded_global = model.global_encoder(
                    graph_global.view(-1).to(dtype=out["lh"]["z_hemi"].dtype)
                )
                pre_head_parts.append(encoded_global)
            pre_head = torch.cat(pre_head_parts, dim=-1).detach().cpu().numpy().astype(np.float32).reshape(-1)
            z_graph = out["z_graph"].detach().cpu().numpy().astype(np.float32).reshape(-1)

            lh_stats = batch["lh"].get("summary_stats", batch["lh"]["morph_stats"])
            rh_stats = batch["rh"].get("summary_stats", batch["rh"]["morph_stats"])
            lh_xyz_stats = batch["lh"]["xyz_stats"]
            rh_xyz_stats = batch["rh"]["xyz_stats"]

            np.save(os.path.join(args.out_dir, f"{sid}.npy"), z_graph)
            subject_ids.append(sid)
            post_head_embeddings.append(z_graph)
            pre_head_embeddings.append(pre_head)
            thickness_mean.append(0.5 * (float(lh_stats["thickness_mean"]) + float(rh_stats["thickness_mean"])))
            thickness_std.append(0.5 * (float(lh_stats["thickness_std"]) + float(rh_stats["thickness_std"])))
            curv_mean.append(0.5 * (float(lh_stats["curv_mean"]) + float(rh_stats["curv_mean"])))
            curv_std.append(0.5 * (float(lh_stats["curv_std"]) + float(rh_stats["curv_std"])))
            curv_abs_mean.append(
                0.5
                * (
                    float(lh_stats.get("curv_abs_mean_raw", batch["lh"]["morph_stats"]["curv_abs_mean_raw"]))
                    + float(rh_stats.get("curv_abs_mean_raw", batch["rh"]["morph_stats"]["curv_abs_mean_raw"]))
                )
            )
            xyz_scale.append(
                0.5
                * (
                    float(lh_stats.get("xyz_scale", lh_xyz_stats["scale"]))
                    + float(rh_stats.get("xyz_scale", rh_xyz_stats["scale"]))
                )
            )

    if not post_head_embeddings:
        raise RuntimeError("No embeddings were extracted.")

    post_head_matrix = np.stack(post_head_embeddings, axis=0).astype(np.float32, copy=False)
    pre_head_matrix = np.stack(pre_head_embeddings, axis=0).astype(np.float32, copy=False)
    subject_stats = {
        "thickness_mean": np.asarray(thickness_mean, dtype=np.float32),
        "thickness_std": np.asarray(thickness_std, dtype=np.float32),
        "curvature_mean": np.asarray(curv_mean, dtype=np.float32),
        "curvature_std": np.asarray(curv_std, dtype=np.float32),
        "curvature_abs_mean": np.asarray(curv_abs_mean, dtype=np.float32),
        "xyz_scale": np.asarray(xyz_scale, dtype=np.float32),
    }

    _write_subject_ids(os.path.join(args.out_dir, "subject_ids.csv"), subject_ids)
    _write_subject_stats_csv(
        os.path.join(args.out_dir, "subject_stats.csv"),
        subject_ids=subject_ids,
        subject_stats=subject_stats,
    )
    post_bundle = _write_embedding_outputs(
        out_dir=args.out_dir,
        prefix="post_head_",
        scatter_title="post-head z_graph PCA",
        subject_ids=subject_ids,
        embedding_matrix=post_head_matrix,
        subject_stats=subject_stats,
        model_config=model_config,
        ckpt_args=ckpt_args,
        skip_umap=bool(args.skip_umap),
    )
    pre_bundle = _write_embedding_outputs(
        out_dir=args.out_dir,
        prefix="pre_head_",
        scatter_title="pre-head embedding PCA",
        subject_ids=subject_ids,
        embedding_matrix=pre_head_matrix,
        subject_stats=subject_stats,
        model_config=model_config,
        ckpt_args=ckpt_args,
        skip_umap=bool(args.skip_umap),
    )

    # Backward-compatible aliases: keep legacy filenames pointing to post-head z_graph.
    np.save(os.path.join(args.out_dir, "embeddings.npy"), post_head_matrix)
    _write_embeddings_csv(
        os.path.join(args.out_dir, "embeddings_with_sid.csv"),
        subject_ids=subject_ids,
        embeddings=post_head_matrix,
    )
    _write_pca_csv(
        os.path.join(args.out_dir, "pca_2d.csv"),
        subject_ids=subject_ids,
        pca_xy=post_bundle["pca_xy"],
    )
    _save_scatter(
        os.path.join(args.out_dir, "pca_scatter.png"),
        xy=post_bundle["pca_xy"],
        title="z_graph PCA",
    )
    if post_bundle["umap_xy"] is not None:
        _write_pca_csv(
            os.path.join(args.out_dir, "umap_2d.csv"),
            subject_ids=subject_ids,
            pca_xy=post_bundle["umap_xy"],
        )
        _save_scatter(
            os.path.join(args.out_dir, "umap_scatter.png"),
            xy=post_bundle["umap_xy"],
            title="z_graph UMAP",
        )
    with open(os.path.join(args.out_dir, "embedding_stats.json"), "w") as f:
        json.dump(post_bundle["stats"], f, indent=2)

    head_comparison = {
        "pre_head": pre_bundle["stats"],
        "post_head": post_bundle["stats"],
        "delta": {
            "effective_rank_post_minus_pre": float(
                post_bundle["stats"]["effective_rank"] - pre_bundle["stats"]["effective_rank"]
            ),
            "pairwise_distance_mean_post_minus_pre": float(
                post_bundle["stats"]["pairwise_distance_mean"] - pre_bundle["stats"]["pairwise_distance_mean"]
            ),
            "abs_pc1_curvature_corr_post_minus_pre": float(
                abs(post_bundle["stats"]["pc1_curvature_abs_mean_corr"])
                - abs(pre_bundle["stats"]["pc1_curvature_abs_mean_corr"])
            ),
            "abs_pc1_xyz_scale_corr_post_minus_pre": float(
                abs(post_bundle["stats"]["pc1_xyz_scale_corr"])
                - abs(pre_bundle["stats"]["pc1_xyz_scale_corr"])
            ),
            "abs_pc1_thickness_mean_corr_post_minus_pre": float(
                abs(post_bundle["stats"]["pc1_thickness_mean_corr"])
                - abs(pre_bundle["stats"]["pc1_thickness_mean_corr"])
            ),
        },
    }
    with open(os.path.join(args.out_dir, "head_comparison.json"), "w") as f:
        json.dump(head_comparison, f, indent=2)

    print(f"Saved z_graph embeddings to: {os.path.abspath(args.out_dir)}")
    print(f"pre_head_embeddings.npy shape: {tuple(pre_head_matrix.shape)}")
    print(f"post_head_embeddings.npy shape: {tuple(post_head_matrix.shape)}")


if __name__ == "__main__":
    main()
