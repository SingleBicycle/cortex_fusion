"""PyTorch dataset for cortical template-aligned surface graph branch."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.feature_schema import get_feature_schema
from src.data.io_fs import read_annot, read_label_vertices, read_surface, robust_read_morph

IGNORE_INDEX = -1
_IGNORE_NAME_TOKENS = (
    "unknown",
    "medial",
    "wall",
    "corpuscallosum",
    "corpus_callosum",
)


REQUIRED_COLUMNS = {
    "sid",
    "hemi",
    "res",
    "pial_path",
    "white_path",
    "thickness_path",
    "curv_path",
    "N",
    "F",
}


def _decode_name(name_obj) -> str:
    if isinstance(name_obj, bytes):
        return name_obj.decode("utf-8", errors="ignore").strip()
    return str(name_obj).strip()


def _canonical_name(name_obj) -> str:
    return _decode_name(name_obj).lower()


def _is_ignored_label_name(name_obj) -> bool:
    s = _canonical_name(name_obj)
    if not s:
        return True
    if s in {"?", "-1", "none"}:
        return True
    return any(tok in s for tok in _IGNORE_NAME_TOKENS)


def _zscore(vec: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
    mean = float(vec.mean())
    std = float(vec.std())
    if std < eps:
        std = 1.0
    out = (vec - mean) / std
    return out.astype(np.float32), mean, std


def _normalize_xyz(pial_xyz: np.ndarray, white_xyz: np.ndarray, eps: float = 1e-6):
    both = np.concatenate([pial_xyz, white_xyz], axis=0)
    center = both.mean(axis=0, keepdims=True)
    p = pial_xyz - center
    w = white_xyz - center

    # RMS scale per hemisphere/subject to make magnitudes comparable across subjects.
    rms = float(np.sqrt(np.mean(np.sum(np.concatenate([p, w], axis=0) ** 2, axis=1))))
    if rms < eps:
        rms = 1.0
    return (p / rms).astype(np.float32), (w / rms).astype(np.float32), center.squeeze(0), rms


def _build_feature_tensor(
    input_mode: str,
    pial_norm: np.ndarray,
    white_norm: np.ndarray,
    mid_norm: np.ndarray,
    thickness_norm: np.ndarray,
    curv_norm: np.ndarray,
) -> np.ndarray:
    if input_mode == "baseline8":
        x = np.concatenate(
            [
                pial_norm,
                white_norm,
                thickness_norm[:, None],
                curv_norm[:, None],
            ],
            axis=1,
        )
    elif input_mode == "main5":
        x = np.concatenate(
            [
                mid_norm,
                thickness_norm[:, None],
                curv_norm[:, None],
            ],
            axis=1,
        )
    elif input_mode == "ablation2":
        x = np.stack([thickness_norm, curv_norm], axis=1)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return x.astype(np.float32, copy=False)


class SurfaceSubjectDataset(Dataset):
    """Subject-level dataset returning both hemispheres for one resolution."""

    def __init__(
        self,
        manifest_csv: str,
        res: str = "fsaverage6",
        random_resolution: bool = False,
        edge_cache_dir: str = "cache/templates",
        seed: int = 42,
        class_names: Optional[Sequence[str]] = None,
        in_memory_cache: bool = False,
        input_mode: str = "baseline8",
    ) -> None:
        super().__init__()
        self.manifest_csv = os.path.abspath(manifest_csv)
        self.res = res
        self.random_resolution = bool(random_resolution)
        self.edge_cache_dir = os.path.abspath(edge_cache_dir)
        self.seed = int(seed)
        self.in_memory_cache = bool(in_memory_cache)
        self.input_mode = str(input_mode).lower()
        self.feature_schema = get_feature_schema(self.input_mode)

        self._rng = random.Random(self.seed)
        self._sample_cache: Dict[Tuple[str, str], Dict] = {}

        df = pd.read_csv(self.manifest_csv)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError(f"manifest is empty: {self.manifest_csv}")

        self.df = df.copy()
        if "annot_path" not in self.df.columns:
            self.df["annot_path"] = ""
        if "label_path" not in self.df.columns:
            self.df["label_path"] = ""
        if "label_format" not in self.df.columns:
            self.df["label_format"] = ""
        if "topology_path" not in self.df.columns:
            self.df["topology_path"] = ""

        self.df["annot_path"] = self.df["annot_path"].fillna("").astype(str)
        self.df["label_path"] = self.df["label_path"].fillna("").astype(str)
        self.df["label_format"] = self.df["label_format"].fillna("").astype(str)
        self.df["topology_path"] = self.df["topology_path"].fillna("").astype(str)

        annot_present = self.df["annot_path"].str.len() > 0
        label_present = self.df["label_path"].str.len() > 0
        self.df.loc[~label_present & annot_present, "label_path"] = self.df.loc[
            ~label_present & annot_present, "annot_path"
        ]

        label_format_missing = self.df["label_format"].str.len() == 0
        self.df.loc[label_format_missing & annot_present, "label_format"] = "annot"
        self.df.loc[label_format_missing & ~annot_present & label_present, "label_format"] = "cortex_label"

        topology_missing = self.df["topology_path"].str.len() == 0
        self.df.loc[topology_missing, "topology_path"] = self.df.loc[topology_missing, "pial_path"]

        self.sid_to_rows: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(dict))

        for row in self.df.to_dict(orient="records"):
            sid = str(row["sid"])
            row_res = str(row["res"])
            hemi = str(row["hemi"])
            if hemi not in {"lh", "rh"}:
                continue
            if hemi in self.sid_to_rows[sid][row_res]:
                # Keep deterministic first row if duplicated in manifest.
                continue
            self.sid_to_rows[sid][row_res][hemi] = row

        self.edge_index_by_res: Dict[str, torch.Tensor] = {}
        self.sid_to_available_res: Dict[str, List[str]] = {}
        for sid, res_map in self.sid_to_rows.items():
            available: List[str] = []
            for row_res, hemi_map in res_map.items():
                if "lh" not in hemi_map or "rh" not in hemi_map:
                    continue
                edge_path = os.path.join(self.edge_cache_dir, f"{row_res}_edge_index.pt")
                if os.path.exists(edge_path):
                    available.append(row_res)
            if available:
                self.sid_to_available_res[sid] = sorted(available)

        if not self.sid_to_available_res:
            raise RuntimeError(
                "No subjects have both hemispheres + cached edge_index. "
                "Run cache_edge_index.py first."
            )

        if self.random_resolution:
            self.subject_ids = sorted(self.sid_to_available_res.keys())
        else:
            self.subject_ids = sorted(
                sid for sid, res_list in self.sid_to_available_res.items() if self.res in res_list
            )
            if not self.subject_ids:
                raise RuntimeError(f"No subjects available for fixed resolution: {self.res}")

        needed_res = set()
        if self.random_resolution:
            for res_list in self.sid_to_available_res.values():
                needed_res.update(res_list)
        else:
            needed_res.add(self.res)

        for row_res in sorted(needed_res):
            edge_path = os.path.join(self.edge_cache_dir, f"{row_res}_edge_index.pt")
            if not os.path.exists(edge_path):
                raise FileNotFoundError(f"Missing edge cache: {edge_path}")
            edge_index = torch.load(edge_path, map_location="cpu")
            if not isinstance(edge_index, torch.Tensor):
                raise TypeError(f"edge cache is not a tensor: {edge_path}")
            self.edge_index_by_res[row_res] = edge_index.long().contiguous()

        if class_names is None:
            self.class_names = self._build_class_names()
        else:
            self.class_names = [str(x).lower() for x in class_names]

        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def _build_class_names(self) -> List[str]:
        annot_paths = sorted(
            {
                path
                for path, label_format in zip(
                    self.df["annot_path"].astype(str).tolist(),
                    self.df["label_format"].astype(str).tolist(),
                )
                if path and str(label_format).lower() == "annot"
            }
        )
        if not annot_paths:
            return []

        names_set = set()
        for path in annot_paths:
            labels, ctab, names = read_annot(path)
            del labels, ctab
            for name in names:
                if _is_ignored_label_name(name):
                    continue
                names_set.add(_canonical_name(name))

        return sorted(names_set)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _choose_resolution(self, sid: str) -> str:
        if not self.random_resolution:
            return self.res
        return self._rng.choice(self.sid_to_available_res[sid])

    def _map_labels(self, labels: np.ndarray, names: Sequence) -> np.ndarray:
        y = np.full(labels.shape, IGNORE_INDEX, dtype=np.int64)

        for local_idx, name_obj in enumerate(names):
            if _is_ignored_label_name(name_obj):
                continue
            key = _canonical_name(name_obj)
            global_idx = self.class_to_index.get(key)
            if global_idx is None:
                continue
            y[labels == local_idx] = global_idx

        y[labels < 0] = IGNORE_INDEX
        return y

    def _load_hemi(self, row: Dict, row_res: str) -> Dict:
        pial_xyz, pial_faces = read_surface(str(row["pial_path"]))
        white_xyz, white_faces = read_surface(str(row["white_path"]))

        if pial_xyz.shape != white_xyz.shape:
            raise ValueError(
                f"white/pial vertex mismatch: {row['sid']} {row['hemi']} {row_res} "
                f"pial={pial_xyz.shape} white={white_xyz.shape}"
            )
        if (
            pial_faces.size > 0
            and white_faces.size > 0
            and (pial_faces.shape != white_faces.shape or not np.array_equal(pial_faces, white_faces))
        ):
            raise ValueError(f"faces topology mismatch: {row['sid']} {row['hemi']} {row_res}")

        n_verts = pial_xyz.shape[0]

        label_format = str(row.get("label_format", "annot")).lower()
        label_path = str(row.get("label_path", "")) or str(row.get("annot_path", ""))
        if label_format == "annot":
            labels, ctab, names = read_annot(label_path)
            del ctab
            labels = np.asarray(labels)
            if labels.shape[0] != n_verts:
                raise ValueError(
                    f"annot length mismatch: {row['sid']} {row['hemi']} {row_res} "
                    f"labels={labels.shape[0]} N={n_verts}"
                )
            y = self._map_labels(labels=labels, names=names)
            mask_valid = y != IGNORE_INDEX
        elif label_format == "cortex_label":
            verts = read_label_vertices(label_path)
            valid = verts[(verts >= 0) & (verts < n_verts)]
            y = np.full(n_verts, IGNORE_INDEX, dtype=np.int64)
            mask_valid = np.zeros(n_verts, dtype=bool)
            mask_valid[valid] = True
        else:
            raise ValueError(
                f"Unsupported label_format={label_format!r} for {row['sid']} {row['hemi']} {row_res}"
            )

        thickness_raw = robust_read_morph(str(row["thickness_path"]), expected_len=n_verts)
        curv_raw = robust_read_morph(str(row["curv_path"]), expected_len=n_verts)

        pial_norm, white_norm, xyz_center, xyz_scale = _normalize_xyz(pial_xyz, white_xyz)
        mid_norm = 0.5 * (pial_norm + white_norm)
        thickness_norm, th_mean, th_std = _zscore(thickness_raw)
        curv_norm, cu_mean, cu_std = _zscore(curv_raw)

        x_target_np = _build_feature_tensor(
            input_mode=self.input_mode,
            pial_norm=pial_norm,
            white_norm=white_norm,
            mid_norm=mid_norm,
            thickness_norm=thickness_norm,
            curv_norm=curv_norm,
        )
        x_target = torch.from_numpy(x_target_np)

        out = {
            "X": x_target.clone(),
            "X_target": x_target.clone(),
            "edge_index": self.edge_index_by_res[row_res],
            "y": torch.from_numpy(y),
            "mask_valid": torch.from_numpy(mask_valid),
            "feature_names": list(self.feature_schema["feature_names"]),
            "recon_weights": torch.tensor(
                self.feature_schema["default_recon_weights"],
                dtype=torch.float32,
            ),
            "input_mode": self.input_mode,
            "thickness_gt": torch.from_numpy(thickness_norm.astype(np.float32)),
            "curv_gt": torch.from_numpy(curv_norm.astype(np.float32)),
            "morph_stats": {
                "thickness_mean": th_mean,
                "thickness_std": th_std,
                "curv_mean": cu_mean,
                "curv_std": cu_std,
            },
            "xyz_stats": {
                "center": xyz_center.astype(np.float32),
                "scale": float(xyz_scale),
            },
        }
        return out

    def __getitem__(self, idx: int) -> Dict:
        sid = self.subject_ids[idx]
        row_res = self._choose_resolution(sid)

        cache_key = (sid, row_res)
        if self.in_memory_cache and cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        rows = self.sid_to_rows[sid][row_res]
        lh = self._load_hemi(rows["lh"], row_res=row_res)
        rh = self._load_hemi(rows["rh"], row_res=row_res)

        sample = {
            "sid": sid,
            "res": row_res,
            "input_mode": self.input_mode,
            "feature_schema": dict(self.feature_schema),
            "lh": lh,
            "rh": rh,
        }

        if self.in_memory_cache:
            self._sample_cache[cache_key] = sample
        return sample


__all__ = [
    "SurfaceSubjectDataset",
    "IGNORE_INDEX",
]
