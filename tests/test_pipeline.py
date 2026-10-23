"""Tests for apple_scaffold.shape_pipeline (shared steps of the shape method)."""
import numpy as np

from apple_scaffold import shape_pipeline as sp
from apple_scaffold import shapes


def y_mask(size=256, trunk_x=128, fork_row=170, spread=1.0, width=10):
    mask = np.zeros((size, size), bool)
    for row in range(size):
        centres = [trunk_x] if row >= fork_row else [
            trunk_x - spread * (fork_row - row) * 0.6, trunk_x + spread * (fork_row - row) * 0.6]
        for c in centres:
            lo = int(round(c)) - width // 2
            mask[row, max(lo, 0):max(min(lo + width, size), 0)] = True
    return mask


def synthetic_trees(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [y_mask(trunk_x=int(rng.normal(128, 10)), spread=rng.normal(1.0, 0.1)) for _ in range(n)]


def test_describe_masks_shapes():
    vectors, widths = sp.describe_masks(synthetic_trees(5), rows=16)
    assert vectors.shape == (5, 32) and widths.shape == (16,)


def test_band_threshold_and_masks_recover_known_shapes():
    """With the true mode coefficients and tiny uncertainty the band must match the masks."""
    masks = synthetic_trees(30)
    vectors, widths = sp.describe_masks(masks, rows=32)
    model = shapes.ShapeModel().fit(vectors)
    coeffs = model.encode(vectors, 4)
    cov = np.eye(4) * 1e-6
    threshold = sp.tune_band_threshold(coeffs[:10], cov, model, widths, masks[:10], 20, 0)
    predicted = sp.band_masks(coeffs[10:], cov, model, widths, threshold, 20, 0)
    scores = sp.per_tree_scores(predicted, masks[10:], tolerances=(5,))
    assert scores["dice"].mean() > 0.85
    assert {"dice", "iou", "cl_f1_t5"} <= set(scores.columns)


def test_ridge_pipeline_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    Y = X[:, :3] @ np.array([[1.0, 0.0], [0.5, 2.0], [0.0, -1.0]])
    model = sp.ridge().fit(X, Y)
    assert shapes.weighted_r2(Y, model.predict(X)) > 0.99
