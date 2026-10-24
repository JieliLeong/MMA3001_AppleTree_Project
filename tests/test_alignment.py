"""Tests for the mask-to-image alignment check."""
import numpy as np
import pytest

from apple_scaffold import alignment


def test_colour_codes_range_and_separation():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 255, 0)
    rgb[1, 0] = (255, 255, 255)
    codes = alignment.colour_codes(rgb)
    assert codes.dtype == np.uint8 and codes.max() < 8 * 2 * 4
    assert len({int(codes[0, 0]), int(codes[0, 1]), int(codes[1, 0]), int(codes[1, 1])}) == 4


def test_depth_codes_keep_no_return_separate():
    depth = np.array([[0, 1, 90, 179, 255]])
    codes = alignment.depth_codes(depth, bins=15, top=180)
    assert codes[0, 0] == 0
    assert codes[0, 1:].min() >= 1 and codes[0, 1:].max() == 15
    assert list(codes[0, 1:]) == sorted(codes[0, 1:])


def test_block_ids_tile_the_grid():
    ids = alignment.block_ids(64, 16)
    assert ids.shape == (64, 64) and ids.max() == 15
    assert np.bincount(ids.ravel()).tolist() == [256] * 16


def test_conditional_mi_zero_when_independent_and_one_bit_when_determined():
    independent = np.ones((4, 6, 2))                  # 4 blocks, 6 codes, label 50/50 everywhere
    assert alignment.conditional_mi(independent).sum() == pytest.approx(0.0, abs=1e-12)
    determined = np.zeros((4, 6, 2))
    determined[:, :3, 0] = 5                          # codes 0-2 always label 0
    determined[:, 3:, 1] = 5                          # codes 3-5 always label 1
    assert alignment.conditional_mi(determined).sum() == pytest.approx(1.0, rel=1e-12)


def test_conditional_mi_ignores_position_only_dependence():
    # label depends on the block only (e.g. scaffold at the bottom), not on the code
    counts = np.zeros((2, 3, 2))
    counts[0, :, 0] = 10                              # top block: never scaffold
    counts[1, :, 1] = 10                              # bottom block: always scaffold
    assert alignment.conditional_mi(counts).sum() == pytest.approx(0.0, abs=1e-12)


def _planted(dx_true, dy_true, n=20, size=64, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), bool)
    codes = rng.integers(2, 10, size=(n, size, size)).astype(np.uint8)
    for i in range(n):
        x0, y0 = rng.integers(16, 40, size=2)
        masks[i, y0:y0 + 12, x0:x0 + 5] = True
        shifted = np.roll(np.roll(masks[i], dy_true, axis=0), dx_true, axis=1)
        codes[i][shifted & (rng.random((size, size)) < 0.7)] = 1
    return codes, masks


def test_shift_scan_recovers_a_planted_offset():
    codes, masks = _planted(dx_true=4, dy_true=-2)
    shifts = [(dx, dy) for dy in range(-6, 7) for dx in range(-6, 7)]
    counts = alignment.shift_counts(codes, masks, shifts, n_codes=10, margin=6, block=16)
    scores = alignment.conditional_mi(counts[0]).sum(axis=-1)
    assert alignment.best_shift(scores, shifts) == (4, -2)
    assert scores.max() > 5 * np.median(scores)


def test_every_shift_is_scored_on_the_same_pixels():
    codes, masks = _planted(dx_true=0, dy_true=0, n=6)
    shifts = [(-5, 0), (0, 0), (3, 5)]
    groups = np.arange(6) % 2
    counts = alignment.shift_counts(codes, masks, shifts, n_codes=10, margin=5, block=16,
                                    groups=groups)
    assert counts.shape == (2, 3, 16, 10, 2)
    per_shift = counts.sum(axis=(0, 2, 3, 4))
    assert (per_shift == per_shift[0]).all()
    assert per_shift[0] == 6 * (64 - 2 * 5) ** 2


def test_shift_larger_than_margin_is_rejected():
    codes, masks = _planted(0, 0, n=2)
    with pytest.raises(ValueError):
        alignment.shift_counts(codes, masks, [(8, 0)], n_codes=10, margin=5)


def test_shift_scores_match_scoring_the_counts():
    codes, masks = _planted(dx_true=2, dy_true=0, n=8)
    shifts = [(dx, dy) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    groups = np.arange(8) % 2
    scores = alignment.shift_scores(codes, masks, shifts, n_codes=10, margin=4, block=16,
                                    groups=groups)
    counts = alignment.shift_counts(codes, masks, shifts, n_codes=10, margin=4, block=16,
                                    groups=groups)
    assert np.allclose(scores["groups"], alignment.uncertainty_coefficient(counts))
    assert np.allclose(scores["pooled"], alignment.uncertainty_coefficient(counts.sum(axis=0)))
    assert alignment.best_shift(scores["pooled"], shifts) == (2, 0)


def test_warp_without_scaling_is_the_same_as_a_shift():
    codes, masks = _planted(dx_true=0, dy_true=0, n=5)
    shifts = [(-4, 2), (0, 0), (3, -5)]
    a = alignment.shift_counts(codes, masks, shifts, n_codes=10, margin=5, block=16)
    b = alignment.warp_counts(codes, masks, [(1.0, 1.0, dx, dy) for dx, dy in shifts],
                              n_codes=10, margin=5, block=16)
    assert np.array_equal(a, b)


def test_warp_masks_scales_about_the_centre():
    mask = np.zeros((1, 64, 64), bool)
    mask[0, 20:44, 30:34] = True                       # 24 rows tall, centred vertically
    taller = alignment.warp_masks(mask, sx=1.0, sy=1.5)[0]
    rows = np.flatnonzero(taller.any(axis=1))
    assert rows.max() - rows.min() + 1 == pytest.approx(36, abs=1)
    assert (rows.min() + rows.max()) / 2 == pytest.approx(31.5, abs=0.6)


def test_scale_scan_recovers_a_planted_stretch():
    rng = np.random.default_rng(3)
    n, size = 16, 64
    masks = np.zeros((n, size, size), bool)
    for i in range(n):
        x0 = rng.integers(14, 44)
        masks[i, 8:56, x0:x0 + 4] = True
        masks[i, 20:24, 10:54] = True
    image_truth = alignment.warp_masks(masks, sx=1.2, sy=1.0)   # images are 20 % wider than the masks
    codes = rng.integers(2, 10, size=(n, size, size)).astype(np.uint8)
    codes[image_truth & (rng.random((n, size, size)) < 0.8)] = 1
    scales = [round(s, 2) for s in np.arange(0.9, 1.31, 0.1)]
    transforms = [(sx, sy, 0, 0) for sy in scales for sx in scales]
    counts = alignment.warp_counts(codes, masks, transforms, n_codes=10, margin=6, block=16)
    scores = alignment.uncertainty_coefficient(counts[0])
    sx, sy, _, _ = transforms[int(np.argmax(scores))]
    assert (sx, sy) == (1.2, 1.0)


def test_uncertainty_coefficient_limits():
    independent = np.ones((4, 6, 2))
    assert alignment.uncertainty_coefficient(independent) == pytest.approx(0.0, abs=1e-12)
    determined = np.zeros((4, 6, 2))
    determined[:, :3, 0] = 5
    determined[:, 3:, 1] = 1                          # rarer label, still fully determined
    assert alignment.uncertainty_coefficient(determined) == pytest.approx(1.0, rel=1e-12)
    partial = np.array([[[8, 2], [2, 8]]], float)       # one block, two codes, noisy
    assert 0 < alignment.uncertainty_coefficient(partial) < 1
