"""Tests for apple_scaffold.shapes (centreline description, drawing, SVD shape model)."""
import numpy as np
import pytest

from apple_scaffold import evaluate, shapes


def y_mask(size=128, trunk_x=64, fork_row=80, spread=1.0, width=8):
    """Synthetic scaffold: vertical trunk below ``fork_row``, two straight leaders above."""
    mask = np.zeros((size, size), bool)
    half = width // 2
    for row in range(size):
        if row >= fork_row:
            centres = [trunk_x]
        else:
            offset = spread * (fork_row - row) * 0.6
            centres = [trunk_x - offset, trunk_x + offset]
        for c in centres:
            lo, hi = int(round(c)) - half, int(round(c)) + half
            mask[row, max(lo, 0):max(min(hi, size), 0)] = True
    return mask


def test_describe_then_draw_recovers_the_mask():
    mask = y_mask()
    vector, widths, fork = shapes.describe(mask, n_rows=32)
    redrawn = shapes.draw(vector, widths, mask.shape)
    assert evaluate.dice(redrawn, mask) > 0.9
    assert 70 <= fork <= 84


def test_trunk_only_mask_has_equal_centrelines_and_no_fork():
    mask = np.zeros((64, 64), bool)
    mask[:, 30:36] = True
    vector, widths, fork = shapes.describe(mask, n_rows=16)
    assert fork == -1
    assert np.allclose(vector[:16], vector[16:])
    assert np.allclose(vector[:16], 32.5)


def test_leader_leaving_the_side_is_assigned_to_the_correct_side():
    mask = y_mask(spread=2.2)            # right leader exits the image before the top
    vector, _, _ = shapes.describe(mask, n_rows=32)
    left, right = vector[:32], vector[32:]
    assert np.all(left <= right + 1e-9)
    assert left[0] < 20                  # left leader still tracked to the top


def test_shape_model_finds_two_hidden_parameters():
    rng = np.random.default_rng(0)
    masks = [y_mask(trunk_x=rng.uniform(50, 78), spread=rng.uniform(0.7, 1.2)) for _ in range(40)]
    vectors = np.stack([shapes.describe(m, 32)[0] for m in masks])
    model = shapes.ShapeModel().fit(vectors)
    assert model.explained_variance_ratio_[:2].sum() > 0.97
    full = vectors.shape[1]
    assert np.allclose(model.decode(model.encode(vectors, full)), vectors)


def test_image_descriptor_length():
    rgb = np.zeros((256, 256, 3), np.uint8)
    depth = np.ones((256, 256), np.float32)
    assert shapes.image_descriptor(rgb, depth).shape == (5 * 16 * 16 + 5 * 64,)


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        shapes.describe(np.zeros((32, 32), bool), n_rows=8)
