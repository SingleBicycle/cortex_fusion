"""Build subject-hemi-resolution manifest for cortical surface graph training."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.data.io_fs import read_annot, read_surface, robust_read_morph

PIAL_RE = re.compile(r"^(?P<hemi>lh|rh)\.pial(?:\.resampled)?\.(?P<res>[A-Za-z0-9_]+)$")
WHITE_RE = re.compile(r"^(?P<hemi>lh|rh)\.white(?:\.resampled)?\.(?P<res>[A-Za-z0-9_]+)$")
ANNOT_RE = re.compile(
    r"^(?P<hemi>lh|rh)\.aparc(?:\.resampled)?\.(?P<res>[A-Za-z0-9_]+)\.annot$"
)
THICK_RE = re.compile(r"^(?P<hemi>lh|rh)\.thickness(?:\.[A-Za-z0-9_]+)*\.(?:mgh|mgz)$")
CURV_RE = re.compile(r"^(?P<hemi>lh|rh)\.curv(?:\.[A-Za-z0-9_]+)*\.(?:mgh|mgz)$")


@dataclass(frozen=True)
class ScanResult:
    pial: Dict[Tuple[str, str, str], Set[str]]
    white: Dict[Tuple[str, str, str], Set[str]]
    annot: Dict[Tuple[str, str, str], Set[str]]
    thick: Dict[Tuple[str, str], Set[str]]
    curv: Dict[Tuple[str, str], Set[str]]


def _collect_scan_roots(root: str, mode: Optional[str]) -> List[str]:
    root = os.path.abspath(root)
    if mode is None:
        return [root]

    aliases = [f"SURF_{mode.upper()}", f"FEAT_{mode.upper()}"]
    roots: List[str] = []
    for alias in aliases:
        p = os.path.join(root, alias)
        if os.path.exists(p):
            roots.append(os.path.realpath(p))

    if not roots:
        roots = [root]

    deduped: List[str] = []
    seen = set()
    for p in roots:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)
    return deduped


def _scan_files(scan_roots: Sequence[str]) -> ScanResult:
    pial: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    white: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    annot: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    thick: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    curv: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for base in scan_roots:
        for dirpath, _, filenames in os.walk(base, followlinks=False):
            sid = os.path.basename(dirpath)
            if not sid.startswith("SUBJ_"):
                continue

            for fname in filenames:
                fpath = os.path.abspath(os.path.join(dirpath, fname))

                m = PIAL_RE.match(fname)
                if m:
                    key = (sid, m.group("hemi"), m.group("res"))
                    pial[key].add(fpath)
                    continue

                m = WHITE_RE.match(fname)
                if m:
                    key = (sid, m.group("hemi"), m.group("res"))
                    white[key].add(fpath)
                    continue

                m = ANNOT_RE.match(fname)
                if m:
                    key = (sid, m.group("hemi"), m.group("res"))
                    annot[key].add(fpath)
                    continue

                m = THICK_RE.match(fname)
                if m:
                    key = (sid, m.group("hemi"))
                    thick[key].add(fpath)
                    continue

                m = CURV_RE.match(fname)
                if m:
                    key = (sid, m.group("hemi"))
                    curv[key].add(fpath)
                    continue

    return ScanResult(pial=pial, white=white, annot=annot, thick=thick, curv=curv)


def _pick_single_path(paths: Set[str]) -> str:
    # Deterministic pick if there are duplicate copies/symlinked mirrors.
    return sorted(paths)[0]


def _pick_morph_path(
    candidates: Sequence[str],
    expected_len: int,
    morph_cache: Dict[str, Optional[np.ndarray]],
) -> Optional[str]:
    for path in sorted(candidates):
        vec = morph_cache.get(path, None)
        if path not in morph_cache:
            try:
                vec = robust_read_morph(path, expected_len=None)
            except Exception:  # noqa: BLE001
                vec = None
            morph_cache[path] = vec

        if vec is not None and int(vec.shape[0]) == int(expected_len):
            return path
    return None


def _iter_target_sids(scan: ScanResult, res: str) -> List[str]:
    sids = sorted({sid for (sid, _hemi, r) in scan.pial.keys() if r == res})
    return sids


def build_manifest(
    root: str,
    out_csv: str,
    res: str = "fsaverage6",
    mode: Optional[str] = None,
    max_subjects: Optional[int] = None,
) -> pd.DataFrame:
    scan_roots = _collect_scan_roots(root=root, mode=mode)
    scan = _scan_files(scan_roots)

    target_sids = _iter_target_sids(scan=scan, res=res)
    if max_subjects is not None:
        target_sids = target_sids[: max_subjects]

    rows: List[Dict[str, object]] = []
    skipped = Counter()
    morph_cache: Dict[str, Optional[np.ndarray]] = {}

    for sid in target_sids:
        for hemi in ("lh", "rh"):
            key = (sid, hemi, res)
            pial_paths = scan.pial.get(key, set())
            if not pial_paths:
                skipped["missing_pial"] += 1
                continue

            white_paths = scan.white.get(key, set())
            if not white_paths:
                skipped["missing_white"] += 1
                continue

            annot_paths = scan.annot.get(key, set())
            if not annot_paths:
                skipped["missing_annot"] += 1
                continue

            pial_path = _pick_single_path(pial_paths)
            white_path = _pick_single_path(white_paths)
            annot_path = _pick_single_path(annot_paths)

            try:
                pial_verts, pial_faces = read_surface(pial_path)
            except Exception:  # noqa: BLE001
                skipped["bad_pial"] += 1
                continue

            try:
                white_verts, white_faces = read_surface(white_path)
            except Exception:  # noqa: BLE001
                skipped["bad_white"] += 1
                continue

            n_verts = int(pial_verts.shape[0])
            n_faces = int(pial_faces.shape[0])

            if int(white_verts.shape[0]) != n_verts:
                skipped["white_n_mismatch"] += 1
                continue

            if white_faces.shape != pial_faces.shape or not np.array_equal(white_faces, pial_faces):
                skipped["faces_topology_mismatch"] += 1
                continue

            try:
                annot_labels, _ctab, _names = read_annot(annot_path)
            except Exception:  # noqa: BLE001
                skipped["bad_annot"] += 1
                continue

            if int(np.asarray(annot_labels).shape[0]) != n_verts:
                skipped["annot_n_mismatch"] += 1
                continue

            thick_candidates = list(scan.thick.get((sid, hemi), set()))
            curv_candidates = list(scan.curv.get((sid, hemi), set()))

            thickness_path = _pick_morph_path(
                candidates=thick_candidates,
                expected_len=n_verts,
                morph_cache=morph_cache,
            )
            if thickness_path is None:
                skipped["no_thickness_len_match"] += 1
                continue

            curv_path = _pick_morph_path(
                candidates=curv_candidates,
                expected_len=n_verts,
                morph_cache=morph_cache,
            )
            if curv_path is None:
                skipped["no_curv_len_match"] += 1
                continue

            rows.append(
                {
                    "sid": sid,
                    "hemi": hemi,
                    "res": res,
                    "pial_path": pial_path,
                    "white_path": white_path,
                    "annot_path": annot_path,
                    "thickness_path": thickness_path,
                    "curv_path": curv_path,
                    "N": n_verts,
                    "F": n_faces,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sid", "hemi", "res"]).reset_index(drop=True)

    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    n_subjects = int(df["sid"].nunique()) if not df.empty else 0
    print(f"scan_roots: {scan_roots}")
    print(f"target_res: {res}")
    print(f"manifest_rows: {len(df)}")
    print(f"subjects_kept: {n_subjects}")
    print(f"output: {out_csv}")

    if skipped:
        print("skipped_reasons:")
        for reason, cnt in skipped.most_common():
            print(f"  - {reason}: {cnt}")

    # Quick integrity signal for downstream paired-hemi training.
    if not df.empty:
        hemi_counts = df.groupby("sid")["hemi"].nunique()
        n_pairable = int((hemi_counts == 2).sum())
        print(f"subjects_with_both_hemis: {n_pairable}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cortical graph branch manifest")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parser.add_argument("--res", type=str, default="fsaverage6", help="Target resolution")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["low", "full"],
        help="Optional mode hint for scan roots",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Optional cap for quick tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        root=args.root,
        out_csv=args.out,
        res=args.res,
        mode=args.mode,
        max_subjects=args.max_subjects,
    )


if __name__ == "__main__":
    main()
