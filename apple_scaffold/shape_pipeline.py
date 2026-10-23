"""Shared steps of the shape method (Stages 4b, 4c and 4d).

The method, end to end:

1. each scaffold mask -> two centreline curves -> ``K`` SVD mode coefficients
   (:mod:`apple_scaffold.shapes`);
2. each photo -> a fixed-length summary (:func:`apple_scaffold.shapes.image_descriptor`,
   optionally plus a bark-probability-map summary);
3. ridge regression from summary to coefficients;
4. a Monte Carlo "fuzzy band" around the predicted shape, thresholded.

This module holds the steps the scripts share, so each stage runs exactly the
same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apple_scaffold import data, evaluate, shapes
from apple_scaffold.features import (FEATURE_NAMES, POSITION_FEATURES, load_frame,
                                     pixel_features, sample_pixels)

SIZE = 256
"""Side of the square grid everything is computed on (the masks' native size)."""

IMAGE_ONLY = [i for i, n in enumerate(FEATURE_NAMES) if n not in POSITION_FEATURES]
"""Pixel-feature columns without x and y (used by the bark classifier)."""

RIDGE_ALPHAS = np.logspace(-1, 5, 25)
"""Regularisation strengths searched by ridge's built-in leave-one-out CV."""


def split_frames(folders, seed: int = 0) -> dict:
    """Clean labelled summer frames split by tree (same split as the audit and Stage 3).

    Returns ``{"train": frames, "val": frames, "test": frames}``.
    """
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    frames = frames[(frames["problems"] == "") & frames["path_GT"].notna()].reset_index(drop=True)
    labels = data.grouped_split(frames, seed=seed)
    return {name: frames[labels == name].reset_index(drop=True) for name in ("train", "val", "test")}


def first_pass(frames: pd.DataFrame, rng: np.random.Generator, pixels_per_tree: int,
               grid: int = 16):
    """Masks, coarse photo summaries and a balanced image-only pixel sample per tree.

    Returns
    -------
    masks : list of bool arrays (SIZE, SIZE)
    coarse : ndarray (n_trees, n_descriptor)
    Xpix, ypix : lists of per-tree pixel samples for the bark classifier
    """
    masks, coarse, Xpix, ypix = [], [], [], []
    for frame in frames.itertuples():
        rgb, depth, gt = load_frame(frame, SIZE)
        feats = pixel_features(rgb, depth)
        idx = sample_pixels(gt, pixels_per_tree, rng)
        masks.append(gt)
        coarse.append(shapes.image_descriptor(rgb, depth, grid=grid))
        Xpix.append(feats[idx][:, IMAGE_ONLY])
        ypix.append(gt.ravel()[idx])
    return masks, np.stack(coarse), Xpix, ypix


def coarse_descriptors(frames: pd.DataFrame, grid: int = 16) -> np.ndarray:
    """Only the coarse photo summaries (for sweeps over the summary grid)."""
    out = []
    for frame in frames.itertuples():
        rgb, depth, _ = load_frame(frame, SIZE)
        out.append(shapes.image_descriptor(rgb, depth, grid=grid))
    return np.stack(out)


def bark_classifier(seed: int = 0) -> RandomForestClassifier:
    """Image-only pixel classifier whose probability map summarises visible bark."""
    return RandomForestClassifier(n_estimators=40, max_depth=14, min_samples_leaf=10,
                                  n_jobs=-1, random_state=seed)


def bark_maps(frames: pd.DataFrame, indices, model) -> dict:
    """Bark-probability maps for the trees at ``indices`` (``{index: map}``)."""
    maps = {}
    for i in indices:
        frame = frames.iloc[i]
        rgb, depth, _ = load_frame(frame, SIZE)
        prob = model.predict_proba(pixel_features(rgb, depth)[:, IMAGE_ONLY])[:, 1]
        maps[i] = prob.reshape(SIZE, SIZE).astype(np.float32)
    return maps


def out_of_fold_bark_maps(frames, Xpix, ypix, folds, seed: int = 0, log=print) -> dict:
    """Bark maps in which no tree's map comes from a classifier trained on that tree."""
    maps = {}
    for f, (fit_idx, held_idx) in enumerate(folds.split(np.arange(len(frames))), start=1):
        clf = bark_classifier(seed).fit(np.concatenate([Xpix[i] for i in fit_idx]),
                                        np.concatenate([ypix[i] for i in fit_idx]))
        maps.update(bark_maps(frames, held_idx, clf))
        log(f"  bark maps: fold {f}/{folds.get_n_splits()} done")
    return maps


def ridge() -> Pipeline:
    """Standardise inputs, then ridge regression with alpha chosen by built-in CV."""
    return Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))])


def describe_masks(masks, rows: int = 32):
    """Centreline vectors (n_trees, 2*rows) and the mean width profile (rows,)."""
    described = [shapes.describe(m, rows) for m in masks]
    vectors = np.stack([d[0] for d in described])
    widths = np.nanmean(np.stack([d[1] for d in described]), axis=0)
    return vectors, widths


def band_probabilities(coeffs, cov, model, widths, n_samples: int, seed: int) -> list:
    """Fuzzy-band probability map for each row of ``coeffs`` (tree ``i`` uses seed + i)."""
    return [shapes.probability_map(c, cov, model, widths, n_samples, seed + i)
            for i, c in enumerate(coeffs)]


def band_masks(coeffs, cov, model, widths, threshold: float, n_samples: int, seed: int) -> list:
    """Thresholded, de-speckled fuzzy bands."""
    return [evaluate.clean_mask(p >= threshold)
            for p in band_probabilities(coeffs, cov, model, widths, n_samples, seed)]


def tune_band_threshold(coeffs, cov, model, widths, masks, n_samples: int, seed: int) -> float:
    """Band threshold with the best mean Dice on the given (training) trees."""
    probs = band_probabilities(coeffs, cov, model, widths, n_samples, seed)
    return evaluate.best_threshold(probs, masks)[0]


def per_tree_scores(pred_masks, true_masks, tolerances=(3, 5, 10)) -> pd.DataFrame:
    """One row of scores per tree (Dice, IoU, centreline P/R/F1)."""
    return pd.DataFrame([evaluate.score_mask(p, t, tolerances) for p, t in zip(pred_masks, true_masks)])
