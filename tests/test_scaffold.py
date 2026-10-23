"""Tests for the Stage 3 pieces: features, scoring and baselines."""
import numpy as np
import pytest

from apple_scaffold import evaluate
from apple_scaffold.baselines import ColourRule, PriorMask
from apple_scaffold.features import FEATURE_NAMES, pixel_features, sample_pixels


def v_mask(size=64, width=3):
    """A thick V shape, like the ground-truth scaffold masks."""
    mask = np.zeros((size, size), bool)
    for row in range(size):
        offset = row // 2
        for centre in (size // 2 - offset, size // 2 + offset):
            lo = max(centre - width, 0)
            mask[size - 1 - row, lo:centre + width] = True
    return mask


# ---------------------------------------------------------------- features
def test_feature_matrix_shape_and_known_values():
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[..., 1] = 200                                   # pure green image
    depth = np.full((32, 48), 100, np.float32)
    depth[:4, :4] = 0                                   # a patch with no depth
    feats = pixel_features(rgb, depth).reshape(32, 48, -1)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert feats.shape[-1] == len(FEATURE_NAMES)
    assert feats[10, 10, col["exg"]] == pytest.approx(2.0, abs=1e-3)       # (2g - r - b)/(r+g+b)
    assert feats[10, 10, col["saturation"]] == pytest.approx(1.0, abs=1e-3)
    assert feats[0, 0, col["depth_ok"]] == 0 and feats[20, 20, col["depth_ok"]] == 1
    assert feats[0, 0, col["x"]] == 0 and feats[-1, -1, col["x"]] == pytest.approx(1.0)
    assert feats[-1, -1, col["y"]] == pytest.approx(1.0)
    assert feats[20, 30, col["depth_grad"]] == pytest.approx(0.0, abs=1e-6)  # flat depth


def test_depth_gradient_matches_a_known_slope():
    """A depth ramp rising 0.5 levels per column has gradient 0.5/255 (depth is scaled by 1/255)."""
    width = 256
    ramp = 50 + 0.5 * np.arange(width, dtype=np.float32)
    depth = np.tile(ramp, (64, 1))
    feats = pixel_features(np.zeros((64, width, 3), np.uint8), depth).reshape(64, width, -1)
    grad = feats[32, 128, FEATURE_NAMES.index("depth_grad")]
    assert grad == pytest.approx(0.5 / 255, rel=1e-3)


def test_balanced_pixel_sampling():
    gt = np.zeros((50, 50), bool)
    gt[:5] = True                                       # 250 scaffold pixels
    idx = sample_pixels(gt, 200, np.random.default_rng(0))
    assert len(idx) == 200 and len(set(idx)) == 200
    assert gt.ravel()[idx].sum() == 100


# ---------------------------------------------------------------- scoring
def test_perfect_prediction_scores_one():
    gt = v_mask()
    scores = evaluate.score_mask(gt, gt)
    assert scores["dice"] == pytest.approx(1.0)
    assert scores["iou"] == pytest.approx(1.0)
    assert scores["cl_f1_t5"] == pytest.approx(1.0)


def test_dice_and_iou_on_a_hand_worked_case():
    gt = np.zeros((10, 10), bool)
    gt[:, :4] = True                  # 40 pixels
    pred = np.zeros((10, 10), bool)
    pred[:, 2:6] = True               # 40 pixels, 20 shared
    assert evaluate.dice(pred, gt) == pytest.approx(0.5)
    assert evaluate.iou(pred, gt) == pytest.approx(20 / 60)


def test_centreline_tolerates_small_shifts_but_not_large_ones():
    gt = v_mask()
    shifted = np.roll(gt, 2, axis=1)
    far = np.roll(gt, 12, axis=1)
    assert evaluate.centreline_scores(shifted, gt, (5,))[5]["f1"] > 0.95
    assert evaluate.centreline_scores(far, gt, (5,))[5]["f1"] < 0.5
    assert evaluate.centreline_scores(np.zeros_like(gt), gt, (5,))[5]["f1"] == 0.0


def test_clean_mask_removes_specks_only():
    mask = v_mask()
    noisy = mask.copy()
    noisy[2, 2] = noisy[60, 5] = True
    assert (evaluate.clean_mask(noisy, min_size=10) == mask).all()


def test_threshold_tuning_finds_the_separating_value():
    gt = v_mask()
    prob = np.where(gt, 0.7, 0.3)
    threshold, best = evaluate.best_threshold([prob], [gt])
    assert 0.3 < threshold <= 0.7 and best == pytest.approx(1.0)


def test_bootstrap_interval_contains_the_mean():
    values = np.random.default_rng(0).normal(0.6, 0.1, 80)
    lo, hi = evaluate.bootstrap_ci(values)
    assert lo < values.mean() < hi and hi - lo < 0.1


# ---------------------------------------------------------------- baselines
def test_prior_is_the_average_mask():
    a, b = np.zeros((4, 4), bool), np.ones((4, 4), bool)
    prior = PriorMask().fit([a, b])
    assert np.allclose(prior.predict_proba(), 0.5)


def test_colour_rule_finds_pale_bark_not_leaves():
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:] = (40, 150, 40)                              # leaves
    rgb[:, 8:12] = (170, 165, 160)                      # pale bark stripe
    gt = np.zeros((20, 20), bool)
    gt[:, 8:12] = True
    rule, score = ColourRule.tune([rgb], [gt], min_size=1)
    assert score == pytest.approx(1.0)
    assert (rule.predict_proba(rgb) == gt).all()
