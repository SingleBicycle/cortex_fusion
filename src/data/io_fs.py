"""FreeSurfer IO utilities for cortical surface graph branch."""

from __future__ import annotations

import gzip
import os
import struct
import tempfile
from typing import Iterable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_annot as nib_read_annot
from nibabel.freesurfer.io import read_geometry as nib_read_geometry
from nibabel.freesurfer.io import read_label as nib_read_label
from nibabel.freesurfer.io import read_morph_data as nib_read_morph_data

_MGH_HEADER_BYTES = 284
_MGH_DTYPE_BY_CODE = {
    0: np.dtype(">u1"),   # MRI_UCHAR
    1: np.dtype(">i4"),   # MRI_INT
    2: np.dtype(">i4"),   # MRI_LONG (stored as int32 in common dumps)
    3: np.dtype(">f4"),   # MRI_FLOAT
    4: np.dtype(">i2"),   # MRI_SHORT
}


def read_surface(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read FreeSurfer surface geometry.

    Returns:
        verts: (N, 3) float32
        faces: (F, 3) int64. Coordinate-only MGH surfaces return an empty `(0, 3)` array.
    """
    try:
        verts, faces = nib_read_geometry(path)
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)
    except Exception:  # noqa: BLE001
        img = nib.load(path)
        data = np.asarray(img.get_fdata(), dtype=np.float32)

        if data.ndim == 4 and data.shape[1:] == (1, 1, 3):
            verts = data[:, 0, 0, :]
        elif data.ndim == 2 and data.shape[1] == 3:
            verts = data
        else:
            raise

        faces = np.zeros((0, 3), dtype=np.int64)
        return np.asarray(verts, dtype=np.float32), faces


def read_annot(path: str, orig_ids: bool = False):
    """Read FreeSurfer annotation file.

    Args:
        path: annot file path
        orig_ids: whether to return original annotation ids

    Returns:
        labels, ctab, names
    """
    return nib_read_annot(path, orig_ids=orig_ids)


def read_label_vertices(path: str) -> np.ndarray:
    """Read FreeSurfer `.label` file and return selected vertex indices."""
    verts = nib_read_label(path, read_scalars=False)
    return np.asarray(verts, dtype=np.int64).reshape(-1)


def _is_gzip_bytes(blob: bytes) -> bool:
    return len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _maybe_decompress_gzip_bytes(blob: bytes) -> bytes:
    if _is_gzip_bytes(blob):
        return gzip.decompress(blob)
    return blob


def _validate_expected_len(vec: np.ndarray, expected_len: Optional[int], source: str) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).reshape(-1)
    if expected_len is not None and out.size != expected_len:
        raise ValueError(f"{source}: len={out.size} != expected_len={expected_len}")
    return out


def _manual_read_mgh_from_bytes(blob: bytes) -> np.ndarray:
    """Parse MGH/MGZ payload directly with header-aware binary read."""
    data = _maybe_decompress_gzip_bytes(blob)
    if len(data) < _MGH_HEADER_BYTES:
        raise ValueError("manual MGH parse failed: file shorter than header")

    version, width, height, depth, nframes, mri_type, _dof = struct.unpack(
        ">7i", data[:28]
    )
    if width <= 0 or height <= 0 or depth <= 0 or nframes <= 0:
        raise ValueError(
            "manual MGH parse failed: invalid dims "
            f"(version={version}, dims={[width, height, depth, nframes]})"
        )

    dtype = _MGH_DTYPE_BY_CODE.get(mri_type)
    if dtype is None:
        raise ValueError(f"manual MGH parse failed: unsupported mri_type={mri_type}")

    count = int(width) * int(height) * int(depth) * int(nframes)
    payload_bytes = count * dtype.itemsize
    start = _MGH_HEADER_BYTES
    end = start + payload_bytes
    if len(data) < end:
        raise ValueError(
            "manual MGH parse failed: truncated payload "
            f"(need={end}, got={len(data)})"
        )

    vec = np.frombuffer(data, dtype=dtype, count=count, offset=start)
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def _try_read_morph_nib(path: str) -> np.ndarray:
    return np.asarray(nib_read_morph_data(path), dtype=np.float32).reshape(-1)


def _try_read_morph_nib_from_bytes(blob: bytes) -> np.ndarray:
    # nibabel API accepts path-like; write a temp payload for retry.
    with tempfile.NamedTemporaryFile(suffix=".mgh", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        return _try_read_morph_nib(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def robust_read_morph(path: str, expected_len: Optional[int] = None) -> np.ndarray:
    """Robust morphometry loader for thickness/curvature vectors.

    Strategy:
      1) Try nibabel.freesurfer.io.read_morph_data directly.
      2) If failed and file appears gzip payload, decompress and retry nibabel.
      3) If still failed, fallback to manual MGH/MGZ header-aware parse.

    Args:
        path: morph file path (mgh/mgz/curv style)
        expected_len: if set, enforce vector length

    Returns:
        vec: (N,) float32
    """
    errors = []

    try:
        return _validate_expected_len(_try_read_morph_nib(path), expected_len, "nib_read_morph_data")
    except Exception as e:  # noqa: BLE001
        errors.append(f"step1_nib_direct: {type(e).__name__}: {e}")

    raw = None
    try:
        raw = _read_file_bytes(path)
    except Exception as e:  # noqa: BLE001
        errors.append(f"read_file_bytes: {type(e).__name__}: {e}")

    if raw is not None and _is_gzip_bytes(raw):
        try:
            decompressed = gzip.decompress(raw)
            vec = _try_read_morph_nib_from_bytes(decompressed)
            return _validate_expected_len(vec, expected_len, "nib_read_morph_data[gzip-retry]")
        except Exception as e:  # noqa: BLE001
            errors.append(f"step2_nib_gzip_retry: {type(e).__name__}: {e}")

    try:
        if raw is None:
            raw = _read_file_bytes(path)
        vec = _manual_read_mgh_from_bytes(raw)
        return _validate_expected_len(vec, expected_len, "manual_mgh_parse")
    except Exception as e:  # noqa: BLE001
        errors.append(f"step3_manual_parse: {type(e).__name__}: {e}")

    err_msg = " | ".join(errors)
    raise RuntimeError(f"robust_read_morph failed for {path}: {err_msg}")


__all__ = [
    "read_surface",
    "read_annot",
    "read_label_vertices",
    "robust_read_morph",
]
