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


# ---------------------------------------------------------------- probabilistic drawing
def training_shapes(n=60, seed=0):
    rng = np.random.default_rng(seed)
    masks = [y_mask(trunk_x=rng.uniform(50, 78), fork_row=int(rng.uniform(70, 90)),
                    spread=rng.uniform(0.7, 1.2)) for _ in range(n)]
    described = [shapes.describe(m, 32) for m in masks]
    vectors = np.stack([d[0] for d in described])
    widths = np.nanmean(np.stack([d[1] for d in described]), axis=0)
    return masks, vectors, widths


def test_coverage_equals_averaging_individual_drawings():
    _, vectors, widths = training_shapes(20)
    fast = shapes.coverage(vectors, widths, (128, 128))
    slow = np.mean([shapes.draw(v, widths, (128, 128)) for v in vectors], axis=0)
    assert np.array_equal(fast, slow)


def test_zero_uncertainty_band_is_the_predicted_shape_itself():
    _, vectors, widths = training_shapes(20)
    model = shapes.ShapeModel().fit(vectors)
    coeff = model.encode(vectors[3], 5)[0]
    band = shapes.probability_map(coeff, np.zeros((5, 5)), model, widths, n_samples=20,
                                  shape=(128, 128))
    assert np.array_equal(band > 0.5, shapes.draw(model.decode(coeff[None])[0], widths, (128, 128)))
    assert set(np.unique(band)) <= {0.0, 1.0}


def test_band_without_photo_information_reproduces_the_average_shape_map():
    """Verification: mean 0 + training covariance should give the average of the training shapes.

    The band assumes normally distributed mode coefficients, so the synthetic
    tree parameters here are drawn from normal distributions. (With uniform
    parameters the agreement drops to a correlation of about 0.96.)
    """
    rng = np.random.default_rng(0)
    masks = [y_mask(trunk_x=rng.normal(64, 6), fork_row=int(round(rng.normal(80, 3))),
                    spread=rng.normal(0.95, 0.1)) for _ in range(80)]
    described = [shapes.describe(m, 32) for m in masks]
    vectors = np.stack([d[0] for d in described])
    widths = np.nanmean(np.stack([d[1] for d in described]), axis=0)
    model = shapes.ShapeModel().fit(vectors)
    k = 6
    coeffs = model.encode(vectors, k)
    band = shapes.probability_map(np.zeros(k), np.cov(coeffs.T), model, widths,
                                  n_samples=3000, shape=(128, 128))
    average = shapes.coverage(model.decode(coeffs), widths, (128, 128))
    assert np.corrcoef(band.ravel(), average.ravel())[0, 1] > 0.98
    assert np.abs(band - average).mean() < 0.03


def test_weighted_r2_limits():
    y = np.random.default_rng(1).normal(size=(50, 3)) * [3.0, 1.0, 0.1]
    assert shapes.weighted_r2(y, y) == pytest.approx(1.0)
    assert shapes.weighted_r2(y, np.tile(y.mean(axis=0), (50, 1))) == pytest.approx(0.0, abs=1e-12)


def test_map_descriptor_length():
    assert shapes.map_descriptor(np.zeros((256, 256))).shape == (16 * 16 + 4 * 64,)
