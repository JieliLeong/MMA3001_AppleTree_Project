"""Per-pixel features for classifying scaffold (trunk + main leaders) pixels.

Everything is computed on the ground-truth grid. The RGB and depth images
(640 x 480) are squashed to the mask's square grid (256 x 256 by default),
which is how the masks appear to have been drawn (checked visually in the
audit overlays). Predictions are therefore compared with the labels without
resampling the labels.

Each pixel gets the features in :data:`FEATURE_NAMES`:

=============  =================================================================
``r, g, b``    colour, scaled to 0-1
``exg``        excess green ``(2g - r - b) / (r + g + b)``: high for leaves
``saturation`` ``(max - min) / max`` of the colour: bark is pale and unsaturated
``texture``    local standard deviation of brightness (7 x 7 window at 256 px)
``exg_ctx``    mean excess green in a 15 x 15 neighbourhood (context)
``sat_ctx``    mean saturation in a 15 x 15 neighbourhood (context)
``depth``      encoded depth / 255 (0 where there was no return)
``depth_ok``   1 where the depth sensor returned a value
``depth_grad`` magnitude of the depth gradient (``np.gradient`` of smoothed
               depth; differentiation amplifies noise, so smooth first - Week 7.1)
``x, y``       pixel position, 0-1 across and down the image
=============  =================================================================

Window sizes scale with the grid size so they cover the same part of the tree
at every resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from apple_scaffold import data, holes

FEATURE_NAMES = ("r", "g", "b", "exg", "saturation", "texture", "exg_ctx", "sat_ctx",
                 "depth", "depth_ok", "depth_grad", "x", "y")
"""Column order of :func:`pixel_features`."""

POSITION_FEATURES = ("x", "y")


def load_frame(frame, size: int = 256, fill_depth: bool = False):
    """Load one frame's RGB, depth and (if present) mask on a ``size`` x ``size`` grid.

    Parameters
    ----------
    frame : row of the frames table from :func:`apple_scaffold.data.build_frames`
    size : int
        Side of the square grid. 256 is the masks' native size.
    fill_depth : bool
        Fill enclosed depth holes first (Laplace, direct solve - Stage 2).

    Returns
    -------
    rgb : uint8 array (size, size, 3)
    depth : float32 array (size, size); 0 where there is no return
    gt : bool array (size, size), or None for unlabelled frames
    """
    rgb = data.load_rgb(frame.path_RGB)
    depth = data.load_depth(frame.path_D).astype(np.float32)
    if fill_depth:
        hole = holes.find_holes(depth)
        if hole.any():
            depth = holes.solve_direct(depth, hole).depth.astype(np.float32)
    rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    # nearest-neighbour keeps 'no return' pixels as exact zeros
    depth = np.asarray(Image.fromarray(depth).resize((size, size), Image.NEAREST))
    gt = None
    path = getattr(frame, "path_GT", None)
    if path is not None and not pd.isna(path):
        gt = data.load_gt(path, size=(size, size))
    return rgb, depth, gt


def _window(base: int, width: int) -> int:
    """Odd window size equivalent to ``base`` pixels on a 256-wide grid."""
    return max(3, int(round(base * width / 256)) | 1)


def pixel_features(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Feature matrix for every pixel, shape ``(H * W, len(FEATURE_NAMES))``.

    Rows are in row-major pixel order, so ``features.reshape(H, W, -1)``
    recovers the image layout.
    """
    height, width = depth.shape
    colour = rgb.astype(np.float32) / 255.0
    r, g, b = colour[..., 0], colour[..., 1], colour[..., 2]
    total = r + g + b + 1e-6
    exg = (2 * g - r - b) / total
    brightest = colour.max(axis=-1)
    saturation = (brightest - colour.min(axis=-1)) / (brightest + 1e-6)
    grey = colour.mean(axis=-1)

    def local_mean(a, base):
        """Mean over a square window equivalent to ``base`` pixels at 256 px."""
        return ndimage.uniform_filter(a, size=_window(base, width), mode="nearest")

    texture = np.sqrt(np.maximum(local_mean(grey ** 2, 7) - local_mean(grey, 7) ** 2, 0))
    exg_ctx = local_mean(exg, 15)
    sat_ctx = local_mean(saturation, 15)

    valid = depth > 0
    d = depth.astype(np.float32) / 255.0
    # normalised smoothing: average only the valid neighbours
    weight = local_mean(valid.astype(np.float32), 5)
    smooth = np.where(weight > 0, local_mean(d * valid, 5) / np.maximum(weight, 1e-6), 0)
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy) * width / 256          # per 256-grid pixel
    gradient[~ndimage.binary_erosion(valid, iterations=2)] = 0.0

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    columns = [r, g, b, exg, saturation, texture, exg_ctx, sat_ctx,
               d, valid.astype(np.float32), gradient,
               x / max(width - 1, 1), y / max(height - 1, 1)]
    return np.stack([c.ravel() for c in columns], axis=1).astype(np.float32)


def sample_pixels(gt: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``n`` pixels, half scaffold and half background where possible.

    Scaffold is only ~8 % of pixels, so balanced sampling stops the models
    from learning "always background". The probability threshold is tuned
    afterwards on validation trees to undo the shift in class balance.
    """
    flat = gt.ravel()
    positive, negative = np.flatnonzero(flat), np.flatnonzero(~flat)
    n_pos = min(len(positive), n // 2)
    n_neg = min(len(negative), n - n_pos)
    return np.concatenate([rng.choice(positive, n_pos, replace=False),
                           rng.choice(negative, n_neg, replace=False)])
