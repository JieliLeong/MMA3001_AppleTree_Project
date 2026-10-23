"""Two non-learned baselines for locating the scaffold.

:class:`PriorMask` ignores the image entirely and predicts where scaffold
*usually* is: the average of the training masks. Because the trees are
photographed from similar positions, this is a demanding baseline - a model
that cannot beat it has not learned to read the image.

:class:`ColourRule` is a hand-written classical rule: bark is pale and
unsaturated, unlike leaves (green) and apples (red). Its three thresholds are
chosen on validation trees by grid search.
"""
from __future__ import annotations

import itertools

import numpy as np

from apple_scaffold.evaluate import clean_mask, dice


class PriorMask:
    """Predict the per-pixel fraction of training trees whose mask covers it."""

    def fit(self, masks) -> "PriorMask":
        """Average a list of boolean masks of equal shape."""
        self.probability_ = np.mean(np.stack(masks).astype(np.float32), axis=0)
        return self

    def predict_proba(self, rgb=None, depth=None) -> np.ndarray:
        """The same probability map for every image."""
        return self.probability_


class ColourRule:
    """Scaffold where a pixel is unsaturated, not green and bright enough.

    Parameters
    ----------
    max_saturation, max_exg, min_brightness : float
        Thresholds on saturation, excess green and mean brightness (0-1 scale).
    """

    GRID = {"max_saturation": (0.15, 0.2, 0.25, 0.3, 0.4),
            "max_exg": (-0.05, 0.0, 0.05, 0.1),
            "min_brightness": (0.2, 0.3, 0.4, 0.5)}

    def __init__(self, max_saturation=0.25, max_exg=0.05, min_brightness=0.3):
        self.max_saturation = max_saturation
        self.max_exg = max_exg
        self.min_brightness = min_brightness

    @staticmethod
    def _channels(rgb):
        colour = rgb.astype(np.float32) / 255.0
        r, g, b = colour[..., 0], colour[..., 1], colour[..., 2]
        brightest = colour.max(axis=-1)
        saturation = (brightest - colour.min(axis=-1)) / (brightest + 1e-6)
        exg = (2 * g - r - b) / (r + g + b + 1e-6)
        return saturation, exg, colour.mean(axis=-1)

    def predict_proba(self, rgb, depth=None) -> np.ndarray:
        """1.0 where the rule fires, else 0.0."""
        saturation, exg, brightness = self._channels(rgb)
        fires = ((saturation <= self.max_saturation) & (exg <= self.max_exg)
                 & (brightness >= self.min_brightness))
        return fires.astype(np.float32)

    @classmethod
    def tune(cls, images, masks, min_size: int = 30) -> tuple:
        """Grid-search the thresholds for the best mean Dice on ``images``.

        Returns
        -------
        (best ColourRule, its mean Dice)
        """
        channels = [cls._channels(rgb) for rgb in images]
        best, best_score = None, -1.0
        for s, e, v in itertools.product(*cls.GRID.values()):
            score = np.mean([dice(clean_mask((sat <= s) & (ex <= e) & (bri >= v), min_size), m)
                             for (sat, ex, bri), m in zip(channels, masks)])
            if score > best_score:
                best, best_score = cls(s, e, v), float(score)
        return best, best_score
