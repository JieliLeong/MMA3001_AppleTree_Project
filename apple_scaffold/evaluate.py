"""Scoring a predicted scaffold mask against the ground truth.

Two views of "how good":

* **Area overlap** - Dice and IoU of the predicted and true masks. Strict:
  the true masks are thick lines, so a correct but thinner
  prediction loses points.
* **Centreline agreement** - both masks are thinned to one-pixel skeletons.
  A predicted centreline pixel counts as correct if a true centreline pixel
  lies within ``tol`` pixels (precision); a true centreline pixel counts as
  found if a predicted one lies within ``tol`` (recall). F1 combines them.
  This measures what a pruning robot needs: where the branches run.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def clean_mask(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Remove connected blobs smaller than ``min_size`` pixels (speckle)."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    size = np.bincount(labels.ravel())
    keep = size >= min_size
    keep[0] = False
    return keep[labels]


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient ``2|P & G| / (|P| + |G|)``; 1 when both are empty."""
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(pred, gt).sum() / total)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union; 1 when both are empty."""
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def centreline_scores(pred: np.ndarray, gt: np.ndarray, tolerances=(5,)) -> dict:
    """Precision, recall and F1 of the predicted centreline, per tolerance.

    Skeletons and distance maps are computed once and reused for every
    tolerance. Returns ``{tol: {"precision", "recall", "f1"}}``.
    """
    skel_pred, skel_gt = skeletonize(pred), skeletonize(gt)
    if not skel_pred.any() or not skel_gt.any():
        return {t: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for t in tolerances}
    to_gt = ndimage.distance_transform_edt(~skel_gt)[skel_pred]
    to_pred = ndimage.distance_transform_edt(~skel_pred)[skel_gt]
    scores = {}
    for tol in tolerances:
        precision = float(np.mean(to_gt <= tol))
        recall = float(np.mean(to_pred <= tol))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores[tol] = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def score_mask(pred: np.ndarray, gt: np.ndarray, tolerances=(3, 5, 10)) -> dict:
    """All scores for one image: Dice, IoU and centreline P/R/F1 per tolerance."""
    scores = {"dice": dice(pred, gt), "iou": iou(pred, gt)}
    for tol, cl in centreline_scores(pred, gt, tolerances).items():
        scores.update({f"cl_precision_t{tol}": cl["precision"],
                       f"cl_recall_t{tol}": cl["recall"], f"cl_f1_t{tol}": cl["f1"]})
    return scores


def best_threshold(probabilities, masks, thresholds=None, min_size: int = 30) -> tuple:
    """Threshold on predicted probability that maximises mean Dice.

    Tuned on validation trees only, never on the test trees.

    Returns
    -------
    (threshold, mean Dice at that threshold)
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = (0.5, -1.0)
    for t in thresholds:
        mean_dice = float(np.mean([dice(clean_mask(p >= t, min_size), m)
                                   for p, m in zip(probabilities, masks)]))
        if mean_dice > best[1]:
            best = (float(t), mean_dice)
    return best


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple:
    """Bootstrap confidence interval for the mean over test trees (Week 5.5.11)."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def covered_fraction(rgb: np.ndarray, mask: np.ndarray, min_saturation: float = 0.3) -> float:
    """Share of the scaffold that looks covered by leaves or fruit (occlusion proxy).

    Leaves (green) and apples (red) are strongly saturated colours, while bark,
    soil and sky are not, so a scaffold pixel whose colour saturation exceeds
    ``min_saturation`` is counted as covered. This is a proxy, not a measured
    occlusion.
    """
    colour = rgb.astype(np.float32) / 255.0
    brightest = colour.max(axis=-1)
    saturation = (brightest - colour.min(axis=-1)) / (brightest + 1e-6)
    return float(np.mean(saturation[mask] > min_saturation)) if mask.any() else float("nan")
