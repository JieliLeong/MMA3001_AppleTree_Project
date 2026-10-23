"""Describing a scaffold mask as two centreline curves, and drawing it back.

Stage 3 showed that per-pixel models mostly learn *where scaffolds usually
are*, because most of the scaffold is hidden by leaves. This module instead
treats the scaffold as one shape: a trunk that splits into two leaders.

**Description.** Scanning the mask from the bottom row upwards, each row
crosses one run of scaffold pixels below the fork (the trunk) and two above it
(left and right leaders). At ``n_rows`` sampled rows we record

* ``x_left(y)``  - centre of the left leader (or trunk), and
* ``x_right(y)`` - centre of the right leader (or trunk),

so a tree becomes a vector of ``2 * n_rows`` numbers (equal along the trunk).
Rows where a leader is missing (for example it has left the side of the
image) are filled by linear interpolation along ``y`` (constant beyond the
ends).

**Drawing.** The centrelines are interpolated to every row with a PCHIP
spline (Week 3.4: shape-preserving, so no overshoot between samples) and
thickened by a width profile ``w(y)``.

**Shape model.** :class:`ShapeModel` finds the main ways trees differ with the
singular value decomposition (Week 2.3) of the centred training vectors: each
tree is approximated by the mean shape plus a few "modes".
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def sample_rows(height: int = 256, n_rows: int = 32) -> np.ndarray:
    """Row indices (top to bottom) at which centrelines are recorded."""
    return np.round(np.linspace(0, height - 1, n_rows)).astype(int)


def _runs(row: np.ndarray) -> list:
    """(start, end) column pairs of the runs of True in one mask row."""
    padded = np.concatenate([[False], row, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2] - 1))


def _fill(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation along rows (constant beyond the ends)."""
    ok = ~np.isnan(values)
    if ok.sum() == 0:
        raise ValueError("no scaffold found in any sampled row")
    return np.interp(rows, rows[ok], values[ok])


def describe(mask: np.ndarray, n_rows: int = 32):
    """Centreline vector and run widths of one scaffold mask.

    Returns
    -------
    vector : ndarray (2 * n_rows,)
        ``[x_left at each sampled row, x_right at each sampled row]``,
        rows ordered top to bottom, in pixels.
    widths : ndarray (n_rows,)
        Mean horizontal run width at each sampled row (NaN where empty).
    fork_row : int
        Lowest sampled row (largest index) at which two leaders were seen,
        or -1 if the mask never splits.
    """
    rows = sample_rows(mask.shape[0], n_rows)
    left = np.full(n_rows, np.nan)
    right = np.full(n_rows, np.nan)
    widths = np.full(n_rows, np.nan)
    forked, fork_row = False, -1
    last_left = last_right = None
    for i in range(n_rows - 1, -1, -1):                       # bottom to top
        runs = _runs(mask[rows[i]])
        if not runs:
            continue
        centres = [(a + b) / 2 for a, b in runs]
        sizes = [b - a + 1 for a, b in runs]
        if len(runs) >= 2:
            left[i], right[i] = centres[0], centres[-1]
            widths[i] = (sizes[0] + sizes[-1]) / 2
            if not forked:
                forked, fork_row = True, int(rows[i])
        elif not forked:                                      # trunk
            left[i] = right[i] = centres[0]
            widths[i] = sizes[0]
        else:                                                 # one leader visible
            c = centres[0]
            if last_left is None or abs(c - last_left) <= abs(c - last_right):
                left[i] = c
            else:
                right[i] = c
            widths[i] = sizes[0]
        last_left = left[i] if not np.isnan(left[i]) else last_left
        last_right = right[i] if not np.isnan(right[i]) else last_right
    vector = np.concatenate([_fill(left, rows.astype(float)), _fill(right, rows.astype(float))])
    return vector, widths, fork_row


def draw(vector: np.ndarray, widths: np.ndarray, shape=(256, 256)) -> np.ndarray:
    """Boolean mask of the two thickened centrelines described by ``vector``."""
    height, width = shape
    n_rows = len(widths)
    rows = sample_rows(height, n_rows)
    every = np.arange(height)
    x_left = PchipInterpolator(rows, vector[:n_rows])(every)
    x_right = PchipInterpolator(rows, vector[n_rows:])(every)
    half = np.interp(every, rows, _fill(np.asarray(widths, float), rows.astype(float))) / 2
    cols = np.arange(width)[None, :]
    return ((np.abs(cols - x_left[:, None]) <= half[:, None])
            | (np.abs(cols - x_right[:, None]) <= half[:, None]))


def coverage(vectors: np.ndarray, widths: np.ndarray, shape=(256, 256)) -> np.ndarray:
    """Fraction of the given shapes covering each pixel (vectorised :func:`draw`).

    Parameters
    ----------
    vectors : ndarray (n_shapes, 2 * n_rows)
    widths : ndarray (n_rows,)

    Returns
    -------
    ndarray (height, width) with values in [0, 1].
    """
    vectors = np.atleast_2d(vectors)
    height, width = shape
    n_rows = len(widths)
    rows = sample_rows(height, n_rows)
    every = np.arange(height)
    # PCHIP along the row axis for every shape at once: result (height, n_shapes)
    x_left = PchipInterpolator(rows, vectors[:, :n_rows].T, axis=0)(every)
    x_right = PchipInterpolator(rows, vectors[:, n_rows:].T, axis=0)(every)
    half = np.interp(every, rows, _fill(np.asarray(widths, float), rows.astype(float))) / 2
    lo_l = np.ceil(x_left - half[:, None]).astype(int)
    hi_l = np.floor(x_left + half[:, None]).astype(int)
    lo_r = np.ceil(x_right - half[:, None]).astype(int)
    hi_r = np.floor(x_right + half[:, None]).astype(int)
    # union of the two intervals per (row, shape); counts via difference arrays
    counts = np.zeros((height, width + 1), np.int32)
    row_index = np.repeat(every[:, None], vectors.shape[0], axis=1)

    def add(lo, hi, sign):
        lo = np.clip(lo, 0, width)
        hi = np.clip(hi + 1, 0, width)
        ok = hi > lo
        np.add.at(counts, (row_index[ok], lo[ok]), sign)
        np.add.at(counts, (row_index[ok], hi[ok]), -sign)

    add(lo_l, hi_l, 1)
    add(lo_r, hi_r, 1)
    # where the two intervals overlap, remove the double count
    add(np.maximum(lo_l, lo_r), np.minimum(hi_l, hi_r), -1)
    return np.cumsum(counts, axis=1)[:, :width] / vectors.shape[0]


def probability_map(coefficients: np.ndarray, covariance: np.ndarray, model: "ShapeModel",
                    widths: np.ndarray, n_samples: int = 200, seed: int = 0,
                    shape=(256, 256)) -> np.ndarray:
    """Monte Carlo "fuzzy band": probability that each pixel is scaffold.

    Shapes are sampled from a normal distribution of mode coefficients with the
    predicted mean and the model's error covariance, drawn, and averaged. A
    confident prediction gives a narrow band; an uncertain one a wide band.
    With mean 0 and the training covariance (no information from the photo)
    the result approaches the average training mask.

    Assumption: prediction errors of the mode coefficients are normally
    distributed. ``n_samples`` controls the Monte Carlo error, which falls
    roughly like ``1 / sqrt(n_samples)``.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(coefficients, float),
                                    np.asarray(covariance, float), size=n_samples,
                                    method="eigh")
    return coverage(model.decode(draws), widths, shape)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of the total shape variation predicted: R^2 pooled over all modes."""
    residual = ((y_true - y_pred) ** 2).sum()
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return float(1 - residual / total)


def map_descriptor(prob: np.ndarray, grid: int = 16, columns: int = 64, bands: int = 4) -> np.ndarray:
    """Summary of a per-pixel probability map: a ``grid x grid`` block average plus
    column profiles in ``bands`` horizontal bands (top to bottom)."""
    height, width = prob.shape
    parts = [prob.reshape(grid, height // grid, grid, width // grid).mean(axis=(1, 3)).ravel()]
    for b in range(bands):
        band = prob[b * height // bands:(b + 1) * height // bands]
        parts.append(band.mean(axis=0).reshape(columns, -1).mean(axis=1))
    return np.concatenate(parts).astype(np.float32)


class ShapeModel:
    """Mean shape plus principal modes, from the SVD of training centrelines.

    Attributes (after :meth:`fit`)
    ------------------------------
    mean_ : ndarray (d,)
    components_ : ndarray (d, d) - rows are modes, strongest first
    singular_values_ : ndarray
    explained_variance_ratio_ : ndarray - share of shape variance per mode
    """

    def fit(self, vectors: np.ndarray) -> "ShapeModel":
        vectors = np.asarray(vectors, float)
        self.mean_ = vectors.mean(axis=0)
        _, s, vt = np.linalg.svd(vectors - self.mean_, full_matrices=False)
        self.components_ = vt
        self.singular_values_ = s
        self.explained_variance_ratio_ = s ** 2 / np.sum(s ** 2)
        return self

    def encode(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """Coefficients of the first ``k`` modes, shape (n, k)."""
        return (np.atleast_2d(vectors) - self.mean_) @ self.components_[:k].T

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Shape vectors from mode coefficients, shape (n, d)."""
        coefficients = np.atleast_2d(coefficients)
        k = coefficients.shape[1]
        return self.mean_ + coefficients @ self.components_[:k]


def image_descriptor(rgb: np.ndarray, depth: np.ndarray, grid: int = 16,
                     strip_columns: int = 64) -> np.ndarray:
    """Whole-image summary used to predict the shape of the scaffold.

    * a ``grid x grid`` block average of brightness, excess green,
      saturation, depth and depth-validity over the whole image, and
    * column profiles of the same quantities across the bottom quarter, where
      the trunk is usually visible below the leaves.
    """
    colour = rgb.astype(np.float32) / 255.0
    r, g, b = colour[..., 0], colour[..., 1], colour[..., 2]
    brightest = colour.max(axis=-1)
    maps = [colour.mean(axis=-1),
            (2 * g - r - b) / (r + g + b + 1e-6),
            (brightest - colour.min(axis=-1)) / (brightest + 1e-6),
            depth.astype(np.float32) / 255.0,
            (depth > 0).astype(np.float32)]
    height, width = depth.shape
    parts = []
    for m in maps:
        parts.append(m.reshape(grid, height // grid, grid, width // grid).mean(axis=(1, 3)).ravel())
    strip = slice(3 * height // 4, height)
    for m in maps:
        parts.append(m[strip].mean(axis=0).reshape(strip_columns, -1).mean(axis=1))
    return np.concatenate(parts).astype(np.float32)
