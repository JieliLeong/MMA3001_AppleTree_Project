"""Indexing, pairing and loading for the MMA3001 RGB-D apple-tree dataset.

File names follow the pattern ``R01N01_Summer_GT-12-40-50.png``:

==========  ==============================================================
``R01``     orchard row
``N``/``U`` ``N`` = tree with a summer ground-truth mask, ``U`` = unlabelled
``01``      tree number within the row
``Summer``  season of capture
``GT``      modality: ``D`` (depth), ``RGB`` (colour) or ``GT`` (scaffold mask)
``12-40-50`` capture time h-m-s. **Not zero-padded** (e.g. ``12-41-4``), so
            names must be parsed with a regular expression, not by position.
==========  ==============================================================

One *frame* is the set of files for one tree in one season. Findings from the
R01N01 sample (23 Oct 2026) that shape the loaders below:

* Depth PNGs are 8-bit greyscale saved as three identical RGB channels.
  ``0`` means no sensor return; larger values are farther away. Values are a
  rescaled encoding, **not millimetres**, so metric 3D needs the scale from
  the dataset provider.
* GT masks are 256 x 256 binary images, whereas RGB and depth are 640 x 480.
  The mask must be resized (non-uniformly: 2.5x in x, 1.875x in y) before it
  can be compared with an RGB or depth pixel.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

FILENAME_RE = re.compile(
    r"^(?P<row>R\d+)(?P<kind>[NU])(?P<tree>\d+)_(?P<season>[A-Za-z]+)_"
    r"(?P<modality>D|RGB|GT)-(?P<h>\d{1,2})-(?P<m>\d{1,2})-(?P<s>\d{1,2})\.png$"
)
"""Regular expression for a dataset file name (see module docstring)."""

FRAME_SIZE = (640, 480)
"""(width, height) of the RGB and depth images."""

GT_SEASONS = ("Summer",)
"""Seasons in which labelled (``N``) trees are expected to have a GT mask."""


# --------------------------------------------------------------------------
# Indexing and pairing
# --------------------------------------------------------------------------
def parse_filename(name: str) -> dict | None:
    """Split a dataset file name into its fields.

    Parameters
    ----------
    name : str
        Bare file name, e.g. ``"R01N03_Summer_GT-12-41-4.png"``.

    Returns
    -------
    dict or None
        Keys ``tree_id`` (e.g. ``"R01N03"``), ``row``, ``labelled`` (bool),
        ``season``, ``modality`` and ``time_s`` (seconds after midnight).
        ``None`` if the name does not match the dataset pattern.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        return None
    g = match.groupdict()
    return {
        "tree_id": f"{g['row']}{g['kind']}{g['tree']}",
        "row": g["row"],
        "labelled": g["kind"] == "N",
        "season": g["season"],
        "modality": g["modality"],
        "time_s": int(g["h"]) * 3600 + int(g["m"]) * 60 + int(g["s"]),
    }


def index_files(folders: Iterable[str | Path]) -> tuple[pd.DataFrame, list[Path]]:
    """List every dataset image in one or more folders (searched recursively).

    Returns
    -------
    files : pandas.DataFrame
        One row per matching file: the fields from :func:`parse_filename`
        plus ``path``.
    unmatched : list of pathlib.Path
        PNG files whose names did not match, so they can be inspected rather
        than silently ignored.
    """
    rows, unmatched = [], []
    for folder in folders:
        for path in sorted(Path(folder).rglob("*.png")):
            fields = parse_filename(path.name)
            if fields is None:
                unmatched.append(path)
            else:
                rows.append({**fields, "path": path})
    columns = ["tree_id", "row", "labelled", "season", "modality", "time_s", "path"]
    return pd.DataFrame(rows, columns=columns), unmatched


def build_frames(files: pd.DataFrame, max_time_gap_s: int = 2,
                 gt_seasons: Iterable[str] = GT_SEASONS) -> pd.DataFrame:
    """Group files into frames (one tree, one season) and flag problems.

    Parameters
    ----------
    files : pandas.DataFrame
        Output of :func:`index_files`.
    max_time_gap_s : int
        Largest allowed difference between the capture times of the
        modalities in one frame before it is flagged.
    gt_seasons : iterable of str
        Seasons in which a labelled tree should have a GT mask.

    Returns
    -------
    pandas.DataFrame
        One row per frame with ``path_D``, ``path_RGB``, ``path_GT`` (``None``
        when absent), ``time_gap_s`` and ``problems`` (a ``"; "``-joined
        string, empty when the frame is complete and consistent).
    """
    gt_seasons = set(gt_seasons)
    frames = []
    for (tree_id, season), group in files.groupby(["tree_id", "season"], sort=True):
        labelled = bool(group["labelled"].iloc[0])
        expected = {"D", "RGB"} | ({"GT"} if labelled and season in gt_seasons else set())
        problems = []
        record = {"tree_id": tree_id, "row": group["row"].iloc[0],
                  "labelled": labelled, "season": season}
        for modality in ("D", "RGB", "GT"):
            hits = group[group["modality"] == modality]
            if len(hits) > 1:
                problems.append(f"{len(hits)} {modality} files")
            record[f"path_{modality}"] = hits["path"].iloc[0] if len(hits) else None
            if len(hits) == 0 and modality in expected:
                problems.append(f"missing {modality}")
            if len(hits) and modality not in expected:
                problems.append(f"unexpected {modality}")
        gap = int(group["time_s"].max() - group["time_s"].min())
        record["time_gap_s"] = gap
        if gap > max_time_gap_s:
            problems.append(f"modalities {gap}s apart")
        record["problems"] = "; ".join(problems)
        frames.append(record)
    return pd.DataFrame(frames)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_depth(path: str | Path) -> np.ndarray:
    """Load a depth image as a 2-D array of raw encoded values.

    ``0`` means no sensor return. Use ``depth > 0`` as the validity mask, and
    do not treat the values as millimetres (see module docstring).

    Raises
    ------
    ValueError
        If a 3-channel file has channels that differ, which would mean the
        depth is colour-mapped and needs decoding rather than slicing.
    """
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., :3]
        if not (np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 0], array[..., 2])):
            raise ValueError(f"{path}: depth channels differ (colour-mapped depth?)")
        array = array[..., 0]
    return array.copy()


def load_rgb(path: str | Path) -> np.ndarray:
    """Load a colour image as a ``uint8`` array of shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB")).copy()


def load_gt(path: str | Path, size: tuple[int, int] | None = FRAME_SIZE) -> np.ndarray:
    """Load a scaffold mask as a boolean array.

    Parameters
    ----------
    path : str or pathlib.Path
        GT PNG (stored as 256 x 256, values 0 and 255).
    size : (width, height) or None
        Resize to this size with nearest-neighbour sampling so the mask stays
        binary. Default matches the RGB/depth frames. ``None`` keeps the
        native 256 x 256 grid, e.g. to evaluate predictions at GT resolution.
    """
    image = Image.open(path).convert("L")
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.NEAREST)
    return np.asarray(image) > 127


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, colour=(255, 0, 255),
                 alpha: float = 0.55) -> np.ndarray:
    """Blend ``colour`` into ``rgb`` where ``mask`` is True (for visual checks)."""
    out = rgb.astype(float).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(colour, dtype=float)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------
# Leakage-safe splitting (MMA3001 Week 5.6)
# --------------------------------------------------------------------------
def grouped_split(frames: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                  group_by: str = "tree_id", seed: int = 0) -> pd.Series:
    """Assign each frame to ``"train"``, ``"val"`` or ``"test"`` by group.

    All frames sharing a ``group_by`` value land in the same split, so a tree
    (or, with ``group_by="row"``, a whole orchard row) is never seen in both
    training and testing. Splitting by row is the stricter test: neighbouring
    trees in a row were captured minutes apart with the same background and
    lighting.

    Fractions refer to groups, not frames.

    Returns
    -------
    pandas.Series
        Split label for each frame, aligned to ``frames.index``.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = frames[group_by].to_numpy()
    labels = pd.Series("train", index=frames.index, dtype=object)

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(outer.split(frames, groups=groups))
    labels.iloc[test_idx] = "test"

    if val_size > 0:
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                                  random_state=seed + 1)
        _, val_rel = next(inner.split(rest_idx, groups=groups[rest_idx]))
        labels.iloc[rest_idx[val_rel]] = "val"
    return labels
