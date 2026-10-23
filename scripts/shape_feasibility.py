"""Stage 4a: can the scaffold be predicted as a *shape* instead of pixel by pixel?

Usage (from the repository root)::

    python scripts/shape_feasibility.py /content/data --out results/shape_check

Uses training and validation trees only; the test trees stay untouched for the
final evaluation. Four questions:

1. **Representation** - if a tree is described by two centreline curves and
   drawn back, how close is it to the original mask? (upper limit)
2. **Variation** - how much do trees differ (trunk position, fork height,
   spread), and how many SVD modes capture that?
3. **Oracle** - drawing each validation tree from its *true* first k mode
   coefficients: how good could a shape method be if the coefficients were
   predicted perfectly?
4. **Predictability** - a first attempt: ridge regression from a whole-image
   summary to the mode coefficients. Does it beat the mean shape and the
   Stage 3 average-mask baseline on validation trees?
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import data, evaluate, shapes  # noqa: E402
from apple_scaffold.baselines import PriorMask  # noqa: E402
from apple_scaffold.features import load_frame  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE = 256
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
})


def split_frames(folders, seed):
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    frames = frames[(frames["problems"] == "") & frames["path_GT"].notna()].reset_index(drop=True)
    labels = data.grouped_split(frames, seed=seed)
    return frames[labels == "train"].reset_index(drop=True), frames[labels == "val"].reset_index(drop=True)


def scores(masks_pred, masks_true):
    """Mean Dice and centreline F1 (5 px) over trees."""
    rows = [evaluate.score_mask(p, t, tolerances=(5,)) for p, t in zip(masks_pred, masks_true)]
    return float(np.mean([r["dice"] for r in rows])), float(np.mean([r["cl_f1_t5"] for r in rows]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="results/shape_check")
    parser.add_argument("--rows", type=int, default=32, help="sampled rows per centreline")
    parser.add_argument("--modes", type=int, default=5, help="modes predicted by the regression")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = split_frames(args.folders, args.seed)
    print(f"Trees: {len(train)} train, {len(val)} validation (test trees not used)", flush=True)

    # ---------------- describe every mask
    start = time.perf_counter()
    def load_masks(frames):
        return [data.load_gt(p, size=(SIZE, SIZE)) for p in frames.path_GT]
    train_masks, val_masks = load_masks(train), load_masks(val)
    described = [shapes.describe(m, args.rows) for m in train_masks]
    V = np.stack([d[0] for d in described])
    width_profile = np.nanmean(np.stack([d[1] for d in described]), axis=0)
    val_described = [shapes.describe(m, args.rows) for m in val_masks]
    Vv = np.stack([d[0] for d in val_described])
    print(f"Described {len(V) + len(Vv)} masks in {time.perf_counter() - start:.1f} s", flush=True)

    # ---------------- 1. representation quality (validation trees)
    own = scores([shapes.draw(v, d[1]) for v, d in zip(Vv, val_described)], val_masks)
    mean_w = scores([shapes.draw(v, width_profile) for v in Vv], val_masks)

    # ---------------- 2. variation
    n = args.rows
    rows = shapes.sample_rows(SIZE, n)
    trunk_x = V[:, n - 1]                                   # bottom row
    spread_top = V[:, n] - V[:, 0]                          # right - left at the top row
    fork = np.array([d[2] for d in described], float)
    fork[fork < 0] = np.nan
    model = shapes.ShapeModel().fit(V)
    cumulative = np.cumsum(model.explained_variance_ratio_)

    # ---------------- 3. oracle reconstructions
    oracle = {}
    for k in (0, 1, 2, 3, 5, 8, 12):
        coeff = model.encode(Vv, k) if k else np.zeros((len(Vv), 0))
        vecs = model.decode(coeff) if k else np.repeat(model.mean_[None], len(Vv), axis=0)
        oracle[k] = scores([shapes.draw(v, width_profile) for v in vecs], val_masks)

    # ---------------- Stage 3 baseline on the same validation trees
    prior = PriorMask().fit(train_masks)
    best_t, _ = evaluate.best_threshold([prior.predict_proba()] * len(val_masks), val_masks)
    prior_scores = scores([evaluate.clean_mask(prior.predict_proba() >= best_t)] * len(val_masks), val_masks)

    # ---------------- 4. predictability with ridge regression
    start = time.perf_counter()
    def descriptors(frames):
        out_ = []
        for frame in frames.itertuples():
            rgb, depth, _ = load_frame(frame, SIZE)
            out_.append(shapes.image_descriptor(rgb, depth))
        return np.stack(out_)
    X, Xv = descriptors(train), descriptors(val)
    k = args.modes
    Y = model.encode(V, k)
    ridge = Pipeline([("scale", StandardScaler()),
                      ("model", RidgeCV(alphas=np.logspace(-1, 5, 25)))])
    cv_pred = cross_val_predict(ridge, X, Y, cv=KFold(5, shuffle=True, random_state=args.seed))
    r2 = 1 - ((Y - cv_pred) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ridge.fit(X, Y)
    pred_vecs = model.decode(ridge.predict(Xv))
    ridge_masks = [shapes.draw(v, width_profile) for v in pred_vecs]
    ridge_scores = scores(ridge_masks, val_masks)
    print(f"Image descriptors ({X.shape[1]} numbers per tree) and ridge fit in "
          f"{time.perf_counter() - start:.0f} s", flush=True)

    # ---------------- save + print
    table = pd.DataFrame(
        [("Stage 3 baseline: average mask", *prior_scores),
         ("Mean shape (no image)", *oracle[0]),
         (f"Ridge-predicted shape ({k} modes)", *ridge_scores)]
        + [(f"Oracle: true first {kk} modes", *oracle[kk]) for kk in (1, 2, 3, 5, 8, 12)]
        + [("Oracle: full description, mean width", *mean_w),
           ("Oracle: full description, own widths", *own)],
        columns=["method", "dice", "cl_f1_t5"])
    table.to_csv(out / "feasibility.csv", index=False)
    pd.DataFrame({"mode": np.arange(1, len(cumulative) + 1),
                  "explained": model.explained_variance_ratio_,
                  "cumulative": cumulative}).to_csv(out / "modes.csv", index=False)

    print("\n1. REPRESENTATION (validation trees redrawn from their own centrelines)")
    print(f"   own widths:  Dice {own[0]:.3f}, centreline F1 {own[1]:.3f}")
    print(f"   mean widths: Dice {mean_w[0]:.3f}, centreline F1 {mean_w[1]:.3f}")
    print("\n2. VARIATION BETWEEN TREES (256-pixel grid; 1 px = 2.5 px across the original photo)")
    print(f"   trunk position at the bottom: mean {trunk_x.mean():.0f}, std {trunk_x.std():.1f} px")
    print(f"   fork height (row):            mean {np.nanmean(fork):.0f}, std {np.nanstd(fork):.1f} px"
          f"   ({int(np.isnan(fork).sum())} masks never split)")
    print(f"   leader spread at the top:     mean {spread_top.mean():.0f}, std {spread_top.std():.1f} px")
    print("   shape variance captured by the first k SVD modes: "
          + ", ".join(f"k={kk}: {cumulative[kk - 1]:.0%}" for kk in (1, 2, 3, 5, 8)))
    print("\n3/4. VALIDATION SCORES")
    for r in table.itertuples():
        print(f"   {r.method:40s} Dice {r.dice:.3f}   centreline F1@5px {r.cl_f1_t5:.3f}")
    print("\n   ridge: cross-validated R^2 per mode on training trees "
          "(1 = mode predicted perfectly from the photo, 0 = no better than the mean)")
    for i, v in enumerate(r2):
        print(f"     mode {i + 1} ({model.explained_variance_ratio_[i]:.0%} of shape variation): R^2 = {v:.2f}")
    print(f"   ridge regularisation chosen: alpha = {ridge.named_steps['model'].alpha_:.3g}")

    # ---------------- figures
    figure_modes(model, width_profile, n, out)
    figure_variance(cumulative, out)
    figure_scores(table, out)
    figure_examples(val, val_masks, model, pred_vecs, width_profile, out)
    print(f"\nWrote results to {out}")
    return 0


def figure_modes(model, widths, n, out):
    rows = shapes.sample_rows(SIZE, n)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.7), sharey=True)
    for i, ax in enumerate(axes):
        sd = model.singular_values_[i] / np.sqrt(len(model.components_))
        for sign, colour, label in ((-2, PALETTE[0], "-2 std"), (0, INK, "mean"), (2, PALETTE[1], "+2 std")):
            v = model.mean_ + sign * sd * model.components_[i]
            ax.plot(v[:n], rows, color=colour, label=label)
            ax.plot(v[n:], rows, color=colour)
        ax.set(xlim=(0, SIZE), ylim=(SIZE, 0), xlabel="x (pixels)",
               title=f"mode {i + 1}: {model.explained_variance_ratio_[i]:.0%} of variation")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (pixels)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("How trees differ: mean scaffold and the first three SVD modes", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_shape_modes.png")
    plt.close(fig)


def figure_variance(cumulative, out):
    k = np.arange(1, min(len(cumulative), 15) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(k, cumulative[:len(k)], "o-", color=PALETTE[0], markersize=5)
    ax.set(xlabel="number of modes k", ylabel="share of shape variation captured",
           ylim=(0, 1.02), title="Shape variation captured by k modes", xticks=k)
    fig.tight_layout()
    fig.savefig(out / "fig_shape_variance.png")
    plt.close(fig)


def figure_scores(table, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(table))[::-1]
    colours = [GREY if ("baseline" in m or "Mean shape" in m) else
               (PALETTE[0] if "Ridge" in m else PALETTE[2]) for m in table.method]
    ax.scatter(table.dice, y, color=colours, s=30, zorder=3, label="Dice")
    ax.scatter(table.cl_f1_t5, y, color=colours, s=30, marker="D", facecolors="none",
               zorder=3, label="centreline F1 (5 px)")
    ax.set_yticks(y)
    ax.set_yticklabels(table.method, fontsize=7)
    ax.set(xlim=(0, 1.04), xlabel="mean score on validation trees",
           title="Grey: baselines. Blue: predicted from the photo. Green: oracle limits")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_feasibility.png")
    plt.close(fig)


def figure_examples(val, val_masks, model, pred_vecs, widths, out, n_show=4):
    fig, axes = plt.subplots(1, n_show, figsize=(2.1 * n_show, 2.4))
    for ax, i in zip(axes, np.linspace(0, len(val) - 1, n_show).astype(int)):
        truth = val_masks[i]
        mean_shape = shapes.draw(model.mean_, widths)
        predicted = shapes.draw(pred_vecs[i], widths)
        image = np.full(truth.shape + (3,), 255, np.uint8)
        image[truth] = (205, 205, 200)
        image[mean_shape & ~predicted] = (138, 137, 132)
        image[predicted] = (42, 120, 214)
        ax.imshow(image)
        ax.set_title(val.tree_id[i], fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Validation trees: truth (light grey), mean shape (dark grey), "
                 "ridge-predicted shape (blue)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_shape_examples.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
