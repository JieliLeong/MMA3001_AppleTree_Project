"""Do the supplied masks line up with the images?

The masks are 256 x 256 images (how they were made is not documented) and the
photos are squashed to that grid (:mod:`apple_scaffold.data`). If the masks had been drawn in a
different frame - the depth camera's view, a crop of the photo, or the winter
photo - the truth would be offset from the summer photo by roughly the same
shift on every tree. If instead the masks are rough sketches, the error differs
from tree to tree and has no consistent direction.

This module measures the offset by statistical registration. For each candidate
shift (and stretch) of the masks it asks how much a quantised image value (a
colour or depth "code") tells us about the moved label, **within the same small
block of the image**, so that position alone (sky at the top, ground at the
bottom) cannot create the dependence. The basic quantity is the conditional
mutual information

.. math::

    I(C; L \\mid B) = \\sum_b p(b) \\sum_{c,l} p(c, l \\mid b)
    \\log_2 \\frac{p(c, l \\mid b)}{p(c \\mid b)\\, p(l \\mid b)}

in bits, where :math:`C` is the code, :math:`L` the label and :math:`B` the
block. Divided by the label entropy :math:`H(L \\mid B)` it becomes the share of
the label's uncertainty that the image explains (the uncertainty coefficient,
:func:`uncertainty_coefficient`). Unlike the raw information it does not grow
just because a stretch makes the label region bigger, so shifts and stretches
can be compared. The transform that maximises it is where the masks agree best
with the image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def colour_codes(rgb: np.ndarray, hue_bins: int = 8, sat_bins: int = 2,
                 val_bins: int = 4) -> np.ndarray:
    """Quantise a colour image into ``hue_bins * sat_bins * val_bins`` HSV codes (uint8)."""
    hsv = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).convert("HSV")).astype(np.int32)
    h = hsv[..., 0] * hue_bins >> 8
    s = hsv[..., 1] * sat_bins >> 8
    v = hsv[..., 2] * val_bins >> 8
    return ((h * sat_bins + s) * val_bins + v).astype(np.uint8)


def depth_codes(depth: np.ndarray, bins: int = 15, top: float = 180.0) -> np.ndarray:
    """Quantise an 8-bit depth image: code 0 = no return, codes 1..``bins`` = distance bands."""
    depth = np.asarray(depth, float)
    codes = 1 + np.minimum(bins - 1, (depth * bins / top).astype(int))
    return np.where(depth > 0, codes, 0).astype(np.uint8)


def block_ids(size: int = 256, block: int = 32) -> np.ndarray:
    """Index of the ``block x block`` tile containing each pixel of a ``size x size`` grid."""
    tiles = size // block
    index = np.arange(size) // block
    return (index[:, None] * tiles + index[None, :]).astype(np.int32)


def conditional_mi(counts: np.ndarray) -> np.ndarray:
    """Per-block terms of I(code; label | block) in bits.

    Parameters
    ----------
    counts : ndarray (..., n_blocks, n_codes, 2)
        Joint pixel counts of (block, code, label).

    Returns
    -------
    ndarray (..., n_blocks)
        ``p(b) * I(C; L | B=b)``; summing over blocks gives the conditional
        mutual information.
    """
    counts = np.asarray(counts, float)
    total = counts.sum(axis=(-3, -2, -1), keepdims=True)
    p = counts / total
    p_block = p.sum(axis=(-2, -1), keepdims=True)
    p_code = p.sum(axis=-1, keepdims=True)
    p_label = p.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p * p_block / (p_code * p_label)), 0.0)
    return terms.sum(axis=(-2, -1))


def _prepare(codes: np.ndarray, n_codes: int, margin: int, block: int, groups):
    """Pixel keys (group, block, code) inside the margin and their totals."""
    codes = np.asarray(codes)
    n, height, width = codes.shape
    if height != width:
        raise ValueError("square images expected")
    groups = np.zeros(n, int) if groups is None else np.asarray(groups, int)
    n_groups = int(groups.max()) + 1
    blocks = block_ids(height, block)
    n_blocks = int(blocks.max()) + 1
    cells = n_blocks * n_codes
    keys = (blocks[None] * n_codes + codes.astype(np.int32)
            + (groups[:, None, None] * cells).astype(np.int32))
    inner = (slice(None), slice(margin, height - margin), slice(margin, width - margin))
    total = np.bincount(keys[inner].ravel(), minlength=n_groups * cells)
    return {"keys": keys, "flat_keys": keys.ravel(), "total": total,
            "shape": (n_groups, n_blocks, n_codes),
            "margin": margin, "size": height, "inner": inner}


def _counts(prepared: dict, ones_keys: np.ndarray) -> np.ndarray:
    """(n_groups, n_blocks, n_codes, 2) counts from the keys of the label pixels."""
    n_groups, n_blocks, n_codes = prepared["shape"]
    ones = np.bincount(ones_keys, minlength=n_groups * n_blocks * n_codes)
    out = np.empty((n_groups * n_blocks * n_codes, 2), np.int64)
    out[:, 1] = ones
    out[:, 0] = prepared["total"] - ones
    return out.reshape(n_groups, n_blocks, n_codes, 2)


def _label_pixels(masks: np.ndarray, size: int) -> tuple:
    """Rows, columns and flat indices of every label pixel (for fast shifting)."""
    tree, qy, qx = np.nonzero(np.asarray(masks, bool))
    flat = (tree.astype(np.int64) * size + qy) * size + qx
    return qy.astype(np.int32), qx.astype(np.int32), flat


def _shifted_counts(prepared: dict, label_pixels, dx: int, dy: int) -> np.ndarray:
    """Counts with the masks moved ``dx`` right and ``dy`` down."""
    qy, qx, flat = label_pixels
    m, size = prepared["margin"], prepared["size"]
    ok = (qy >= m - dy) & (qy < size - m - dy) & (qx >= m - dx) & (qx < size - m - dx)
    return _counts(prepared, prepared["flat_keys"][flat[ok] + (dy * size + dx)])


def shift_counts(codes: np.ndarray, masks: np.ndarray, shifts, n_codes: int, margin: int,
                 block: int = 32, groups=None) -> np.ndarray:
    """Joint (block, code, label) counts for every shift of the masks.

    A shift ``(dx, dy)`` moves each mask ``dx`` pixels right and ``dy`` pixels
    down before it is compared with its image. Only pixels at least ``margin``
    from the edge are counted, so every shift is scored on the same pixels.

    Parameters
    ----------
    codes : uint8 array (n, H, W) - quantised images
    masks : bool array (n, H, W) - truth masks
    shifts : sequence of (dx, dy)
    n_codes : int - number of possible codes
    margin : int - must be at least the largest shift
    block : int - side of the conditioning blocks
    groups : int array (n,), optional - e.g. cross-validation folds of trees

    Returns
    -------
    int64 array (n_groups, n_shifts, n_blocks, n_codes, 2)
    """
    shifts = list(shifts)
    if max(max(abs(dx), abs(dy)) for dx, dy in shifts) > margin:
        raise ValueError("margin must be at least the largest shift")
    prepared = _prepare(codes, n_codes, margin, block, groups)
    label_pixels = _label_pixels(masks, prepared["size"])
    return np.stack([_shifted_counts(prepared, label_pixels, dx, dy) for dx, dy in shifts], axis=1)


def best_shift(scores: np.ndarray, shifts):
    """The shift with the highest score, as ``(dx, dy)``."""
    return tuple(int(v) for v in shifts[int(np.argmax(scores))])


def shift_scores(codes: np.ndarray, masks: np.ndarray, shifts, n_codes: int, margin: int,
                 block: int = 32, groups=None) -> dict:
    """Uncertainty coefficient (:func:`uncertainty_coefficient`) for every shift.

    Same as scoring :func:`shift_counts`, without keeping every shift's counts
    in memory.

    Returns
    -------
    dict with ``"groups"`` : array (n_groups, n_shifts), one row per group of
    trees, and ``"pooled"`` : array (n_shifts,) from the counts summed over groups.
    """
    shifts = list(shifts)
    if max(max(abs(dx), abs(dy)) for dx, dy in shifts) > margin:
        raise ValueError("margin must be at least the largest shift")
    prepared = _prepare(codes, n_codes, margin, block, groups)
    label_pixels = _label_pixels(masks, prepared["size"])
    per_group = np.zeros((prepared["shape"][0], len(shifts)))
    pooled = np.zeros(len(shifts))
    for k, (dx, dy) in enumerate(shifts):
        counts = _shifted_counts(prepared, label_pixels, dx, dy)
        per_group[:, k] = uncertainty_coefficient(counts)
        pooled[k] = uncertainty_coefficient(counts.sum(axis=0))
    return {"groups": per_group, "pooled": pooled}


def warp_masks(masks: np.ndarray, sx: float = 1.0, sy: float = 1.0, dx: float = 0.0,
               dy: float = 0.0) -> np.ndarray:
    """Masks scaled by ``(sx, sy)`` about the image centre, then moved ``dx`` right and ``dy`` down.

    Nearest-neighbour inverse mapping: each output pixel takes the mask value at
    ``centre + (p - centre - shift) / scale``; pixels that map outside the
    frame are False. With ``sx = sy = 1`` this is an exact integer shift.
    """
    masks = np.asarray(masks, bool)
    _, height, width = masks.shape
    cy, cx = (height - 1) / 2, (width - 1) / 2
    ys = np.rint(cy + (np.arange(height) - cy - dy) / sy).astype(int)
    xs = np.rint(cx + (np.arange(width) - cx - dx) / sx).astype(int)
    inside = (((ys >= 0) & (ys < height))[:, None]) & (((xs >= 0) & (xs < width))[None, :])
    out = masks[:, np.clip(ys, 0, height - 1)][:, :, np.clip(xs, 0, width - 1)]
    return out & inside[None]


def warp_counts(codes: np.ndarray, masks: np.ndarray, transforms, n_codes: int, margin: int,
                block: int = 32, groups=None) -> np.ndarray:
    """Like :func:`shift_counts`, for general ``(sx, sy, dx, dy)`` transforms of the masks.

    Slower than :func:`shift_counts` (every mask is resampled for every
    transform), so it is used on small grids of scales.
    """
    prepared = _prepare(codes, n_codes, margin, block, groups)
    keys = prepared["keys"][prepared["inner"]]
    out = []
    for sx, sy, dx, dy in transforms:
        label = warp_masks(masks, sx, sy, dx, dy)[prepared["inner"]]
        out.append(_counts(prepared, keys[label]))
    return np.stack(out, axis=1)


def uncertainty_coefficient(counts: np.ndarray) -> np.ndarray:
    """Share of the label's uncertainty explained by the image, I(C; L | B) / H(L | B).

    Unlike the mutual information itself this does not grow just because a
    transform makes the label region larger (a stretched truth covers more
    pixels and so carries more entropy), so different scales can be compared.

    Parameters
    ----------
    counts : ndarray (..., n_blocks, n_codes, 2)

    Returns
    -------
    ndarray (...) between 0 and 1.
    """
    counts = np.asarray(counts, float)
    information = conditional_mi(counts).sum(axis=-1)
    per_block = counts.sum(axis=-2)                                  # (..., n_blocks, 2)
    weight = per_block.sum(axis=-1) / per_block.sum(axis=(-2, -1))[..., None]
    p = per_block / np.maximum(per_block.sum(axis=-1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=-1)
    entropy = (weight * h).sum(axis=-1)
    return np.where(entropy > 0, information / np.where(entropy > 0, entropy, 1), 0.0)
