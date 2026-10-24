"""Audit the apple-tree dataset before any modelling.

Usage (from the repository root)::

    python scripts/audit_dataset.py "path/to/Dataset (With Summer GT)" "path/to/Dataset" --out audit

Writes to ``--out``:

* ``frames.csv``  - one row per (tree, season) with file paths and problems
* ``split.csv``   - train/val/test label per frame (the saved selection record)
* ``overlays/``   - GT drawn over RGB and depth for a few labelled frames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import data  # noqa: E402


def describe_images(frames: pd.DataFrame, n_sample: int, seed: int) -> pd.DataFrame:
    """Report size, mode and value range for a random sample of frames."""
    sample = frames.sample(min(n_sample, len(frames)), random_state=seed)
    rows = []
    for _, frame in sample.iterrows():
        for modality in ("D", "RGB", "GT"):
            path = frame[f"path_{modality}"]
            if path is None or pd.isna(path):
                continue
            image = Image.open(path)
            array = np.asarray(image)
            row = {"tree_id": frame.tree_id, "season": frame.season, "modality": modality,
                   "mode": image.mode, "size": image.size, "dtype": str(array.dtype),
                   "min": int(array.min()), "max": int(array.max())}
            if modality == "D":
                depth = data.load_depth(path)
                row["no_return_frac"] = round(float((depth == 0).mean()), 3)
                row["distinct_levels"] = int(np.unique(depth[depth > 0]).size)
            rows.append(row)
    return pd.DataFrame(rows)


def save_overlays(frames: pd.DataFrame, out: Path, n: int, seed: int) -> None:
    """Save RGB|depth overlays of the GT mask for ``n`` labelled frames."""
    labelled = frames[frames["path_GT"].notna()]
    out.mkdir(parents=True, exist_ok=True)
    for _, frame in labelled.sample(min(n, len(labelled)), random_state=seed).iterrows():
        rgb = data.load_rgb(frame.path_RGB)
        depth = data.load_depth(frame.path_D)
        gt = data.load_gt(frame.path_GT, size=(rgb.shape[1], rgb.shape[0]))
        depth_rgb = np.repeat(depth[..., None], 3, axis=2)
        depth_rgb[depth == 0] = (0, 90, 160)          # no-return pixels in blue
        pair = np.concatenate([data.overlay_mask(rgb, gt),
                               np.full((rgb.shape[0], 10, 3), 255, np.uint8),
                               data.overlay_mask(depth_rgb, gt)], axis=1)
        Image.fromarray(pair).save(out / f"{frame.tree_id}_{frame.season}.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="dataset folders to scan")
    parser.add_argument("--out", default="audit", help="output folder")
    parser.add_argument("--group-by", default="tree_id", choices=["tree_id", "row"])
    parser.add_argument("--sample", type=int, default=20, help="frames to describe")
    parser.add_argument("--overlays", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files, unmatched = data.index_files(args.folders)
    others = sorted({p.name for folder in args.folders for p in Path(folder).rglob("*")
                     if p.is_file() and p.suffix.lower() != ".png"})
    print(f"{len(files)} dataset files, {len(unmatched)} unmatched PNGs")
    print(f"{int(files['source_id'].notna().sum())} files use the re-matched winter name form")
    print(f"non-PNG files ignored: {others}")
    for path in unmatched[:10]:
        print("  unmatched:", path)
    if files.empty:
        return 1

    frames = data.build_frames(files)
    frames.to_csv(out / "frames.csv", index=False)
    print("\nFrames by season and label status:")
    print(frames.groupby(["season", "labelled"]).size().to_string())
    print(f"\nTrees: {frames.tree_id.nunique()}  rows: {frames.row.nunique()}")

    bad = frames[frames["problems"] != ""]
    print(f"\nFrames with problems: {len(bad)}")
    if len(bad):
        print(bad[["tree_id", "season", "problems"]].head(30).to_string(index=False))

    print("\nImage properties (random sample):")
    print(describe_images(frames, args.sample, args.seed).to_string(index=False))

    labelled_summer = frames[frames["path_GT"].notna() & (frames["problems"] == "")]
    n_groups = labelled_summer[args.group_by].nunique()
    if n_groups < 10:
        print(f"\nOnly {n_groups} clean labelled group(s): skipping the train/val/test split.")
    else:
        split = data.grouped_split(labelled_summer, group_by=args.group_by, seed=args.seed)
        record = labelled_summer[["tree_id", "row", "season"]].assign(split=split)
        record.to_csv(out / "split.csv", index=False)
        print(f"\nSplit of {len(record)} labelled summer frames by {args.group_by}:")
        print(record["split"].value_counts().to_string())

    save_overlays(frames, out / "overlays", args.overlays, args.seed)
    print(f"\nWrote results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
