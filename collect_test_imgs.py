#!/usr/bin/env python3
"""Collect OneVL test-set images into ``onevl_opensource_test_imgs/{roadwork,impromptu,navsim,ar1}/``.

Image paths are the same relative paths as in test JSON/JSONL (no hostname). The output zip
uses only those relative paths under the bundle root.

Default layout:
  {out_dir}/roadwork/<path as in test> ...
  {out_dir}/impromptu/...
  {out_dir}/navsim/...
  {out_dir}/ar1/...

Defaults match scripts: opendata base for roadwork/impromptu/ar1, evad-osc for navsim.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Set, Tuple


def _roadwork_images(test_data: Path) -> Set[str]:
    data = json.loads((test_data / "roadwork_test.json").read_text(encoding="utf-8"))
    out: Set[str] = set()
    for item in data:
        for im in item.get("images") or []:
            if isinstance(im, str) and im.strip():
                out.add(im.strip())
    return out


def _navsim_images(test_data: Path) -> Set[str]:
    data = json.loads((test_data / "navsim_test.json").read_text(encoding="utf-8"))
    out: Set[str] = set()
    for item in data:
        for im in item.get("images") or []:
            if isinstance(im, str) and im.strip():
                out.add(im.strip())
    return out


def _jsonl_images(test_data: Path, name: str) -> Set[str]:
    out: Set[str] = set()
    with open(test_data / name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for im in item.get("images") or []:
                if isinstance(im, str) and im.strip():
                    out.add(im.strip())
    return out


def _copy_dataset(
    label: str,
    rel_paths: Set[str],
    base: Path,
    out_root: Path,
    missing_log: List[str],
) -> Tuple[int, int]:
    ok = 0
    for rel in sorted(rel_paths):
        src = base / rel
        dst = out_root / label / rel
        if not src.is_file():
            missing_log.append(f"{label}\t{rel}\t{src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ok += 1
    return ok, len(rel_paths)


def _zip_tree(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.name != "_missing_sources.txt":
                arc = p.relative_to(out_dir)
                zf.write(p, arc.as_posix())


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Collect OneVL test images + zip")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=script_dir.parent / "onevl_opensource_test_imgs",
        help="Directory to populate (default: sibling of onevl_opensource/)",
    )
    ap.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help="Zip file path (default: {out_dir}.zip next to out_dir parent)",
    )
    ap.add_argument(
        "--test_data",
        type=Path,
        default=script_dir / "test_data",
    )
    ap.add_argument(
        "--base_opendata",
        type=Path,
        default=Path("/e2e-data/embodied-research-data/opendata"),
    )
    ap.add_argument(
        "--base_navsim",
        type=Path,
        default=Path("/e2e-data/evad-osc-datasets/datasets"),
    )
    ap.add_argument("--skip_zip", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = args.test_data
    datasets: List[Tuple[str, Set[str], Path]] = [
        ("roadwork", _roadwork_images(test_data), args.base_opendata),
        ("impromptu", _jsonl_images(test_data, "impromptu_test.jsonl"), args.base_opendata),
        ("navsim", _navsim_images(test_data), args.base_navsim),
        ("ar1", _jsonl_images(test_data, "ar1_test.jsonl"), args.base_opendata),
    ]

    missing: List[str] = []
    total_ok = 0
    total_decl = 0
    for label, rels, base in datasets:
        n_ok, n_all = _copy_dataset(label, rels, base, out_dir, missing)
        total_ok += n_ok
        total_decl += n_all
        print(f"{label}: copied {n_ok} / {n_all} files (base={base})")

    miss_path = out_dir / "_missing_sources.txt"
    if missing:
        miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"WARN: {len(missing)} missing files -> {miss_path}")
    else:
        if miss_path.is_file():
            miss_path.unlink()

    if args.skip_zip:
        return

    zip_path = args.zip_path
    if zip_path is None:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
    print(f"Writing zip (paths relative to bundle root): {zip_path}")
    _zip_tree(out_dir, zip_path)
    print(f"Done. Files in archive: {total_ok} (approx); declared rel paths: {total_decl}")


if __name__ == "__main__":
    main()
