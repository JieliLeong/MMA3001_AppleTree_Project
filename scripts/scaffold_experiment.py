"""Stage 3: compare methods for finding the scaffold (trunk + main leaders).

Usage (from the repository root)::

    python scripts/scaffold_experiment.py /content/data --out results/scaffold

Methods compared
----------------
``prior``           average training mask - where scaffold usually is (baseline)
``colour_rule``     hand-tuned "pale, unsaturated, not green" rule (baseline)
``logistic``        logistic regression on the pixel features (linear ML baseline)
``random_forest``   random forest on the pixel features
``rf_no_position``  random forest without the x, y features (ablation: how much
                    is learned from the image rather than from position?)
``mlp``             neural network (multi-layer perceptron) on the pixel features

Protocol (no test-set leakage)
------------------------------
* models learn from pixels sampled from the **training** trees;
* probability thresholds and the colour rule's settings are chosen on
  **validation** trees;
* scores are reported once, on the **test** trees.

Writes ``metrics.csv`` (one row per test tree and method), ``summary.csv``,
``importance.csv``, ``mlp_curve.csv`` and ``fig_*.png`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import data, evaluate  # noqa: E402
from apple_scaffold.baselines import ColourRule, PriorMask  # noqa: E402
from apple_scaffold.features import (FEATURE_NAMES, POSITION_FEATURES,  # noqa: E402
                                     load_frame, pixel_features, sample_pixels)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL_SIZE = 256                       # every method is scored on the masks' native grid
BASELINES = ("prior", "colour_rule")
LEARNED = ("logistic", "random_forest", "rf_no_position", "mlp")
METHODS = BASELINES + LEARNED
LABELS = {"prior": "Prior (average mask)", "colour_rule": "Colour rule",
          "logistic": "Logistic regression", "random_forest": "Random forest",
          "rf_no_position": "Random forest, no x/y", "mlp": "Neural network (MLP)"}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, BASELINE_GREY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
})


# --------------------------------------------------------------------------
# Set-up
# --------------------------------------------------------------------------
def make_models(seed: int) -> dict:
    """The learned models, each with the feature columns it uses."""
    all_cols = list(range(len(FEATURE_NAMES)))
    no_pos = [i for i, n in enumerate(FEATURE_NAMES) if n not in POSITION_FEATURES]
    forest = dict(n_estimators=60, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    return {
        "logistic": (Pipeline([("scale", StandardScaler()),
                               ("model", LogisticRegression(max_iter=1000))]), all_cols),
        "random_forest": (RandomForestClassifier(**forest), all_cols),
        "rf_no_position": (RandomForestClassifier(**forest), no_pos),
        "mlp": (Pipeline([("scale", StandardScaler()),
                          ("model", MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024,
                                                  learning_rate_init=1e-3, max_iter=100,
                                                  early_stopping=True, n_iter_no_change=8,
                                                  random_state=seed))]), all_cols),
    }


def split_frames(folders, seed: int):
    """Clean labelled summer frames, split by tree exactly as in the audit."""
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    frames = frames[(frames["problems"] == "") & frames["path_GT"].notna()].reset_index(drop=True)
    labels = data.grouped_split(frames, seed=seed)
    return {name: frames[labels == name].reset_index(drop=True) for name in ("train", "val", "test")}


def to_eval_grid(prob: np.ndarray) -> np.ndarray:
    """Resample a probability map to the 256 x 256 evaluation grid (bilinear)."""
    if prob.shape[0] == EVAL_SIZE:
        return prob
    zoom = EVAL_SIZE / prob.shape[0]
    return np.clip(ndimage.zoom(prob, zoom, order=1), 0, 1)


def predict_map(method, models, prior, rule, rgb, depth, feats, size):
    """Probability map for one image from one method."""
    if method == "prior":
        return prior.predict_proba()
    if method == "colour_rule":
        return rule.predict_proba(rgb)
    model, cols = models[method]
    return model.predict_proba(feats[:, cols])[:, 1].reshape(size, size)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_scores(metrics, summary, out):
    order = summary.sort_values("dice_mean")["method"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, (col, title) in zip(axes, [("dice", "Dice (area overlap)"),
                                       ("cl_f1_t5", "Centreline F1 (within 5 px)")]):
        for i, method in enumerate(order):
            values = metrics.loc[metrics.method == method, col].to_numpy()
            colour = BASELINE_GREY if method in BASELINES else PALETTE[0]
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(values))
            ax.scatter(values, i + jitter, s=6, color=colour, alpha=0.25, linewidths=0)
            row = summary.set_index("method").loc[method]
            ax.errorbar(row[f"{col}_mean"], i,
                        xerr=[[row[f"{col}_mean"] - row[f"{col}_lo"]], [row[f"{col}_hi"] - row[f"{col}_mean"]]],
                        fmt="o", color=colour, markersize=6, capsize=3, linewidth=1.5)
        ax.set(title=title, xlim=(0, 1), xlabel="score per test tree (dot = mean, bar = 95% CI)")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABELS[m] for m in order])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=BASELINE_GREY, label="baseline"),
               plt.Line2D([], [], marker="o", linestyle="", color=PALETTE[0], label="learned")]
    axes[1].legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_scores.png")
    plt.close(fig)


def figure_examples(examples, thresholds, out):
    shown = ["prior", "colour_rule", "random_forest", "mlp"]
    fig, axes = plt.subplots(len(examples), len(shown) + 1,
                             figsize=(2.0 * (len(shown) + 1), 2.1 * len(examples)), squeeze=False)
    for row, (tree_id, rgb, gt, probs) in enumerate(examples):
        base = (rgb.astype(float) * 0.55).astype(np.uint8)
        outline = gt ^ ndimage.binary_erosion(gt, iterations=2)
        panels = [("photo + ground truth", None)] + [(LABELS[m], m) for m in shown]
        for col, (title, method) in enumerate(panels):
            ax = axes[row, col]
            image = base.copy()
            if method is None:
                image[gt] = (0.45 * image[gt] + 0.55 * np.array([232, 123, 164])).astype(np.uint8)
            else:
                pred = evaluate.clean_mask(probs[method] >= thresholds[method])
                image[pred] = (0.4 * image[pred] + 0.6 * np.array([42, 120, 214])).astype(np.uint8)
                image[outline] = (255, 255, 255)
            ax.imshow(image)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(tree_id, fontsize=8)
    fig.suptitle("Test trees: prediction in blue, ground-truth outline in white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_examples.png")
    plt.close(fig)


def figure_importance(importance, out):
    imp = importance.sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.barh(imp.feature, imp["mean"], xerr=imp["std"], color=PALETTE[0], height=0.6,
            error_kw={"ecolor": MUTED, "linewidth": 1})
    ax.set(xlabel="drop in ROC AUC when the feature is shuffled",
           title="Random forest: which features matter (validation pixels)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_importance.png")
    plt.close(fig)


def figure_mlp(curve, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    axes[0].plot(curve.epoch, curve.training_loss, color=PALETTE[0])
    axes[0].set(xlabel="epoch", ylabel="training loss (log-loss)", title="MLP training loss")
    axes[1].plot(curve.epoch, curve.validation_accuracy, color=PALETTE[0])
    best = curve.loc[curve.validation_accuracy.idxmax()]
    axes[1].axvline(best.epoch, color=MUTED, linestyle="--", linewidth=1)
    axes[1].annotate(f"best epoch {int(best.epoch)}", (best.epoch, curve.validation_accuracy.min()),
                     xytext=(4, 0), textcoords="offset points", color=MUTED)
    axes[1].set(xlabel="epoch", ylabel="held-out pixel accuracy",
                title="Early-stopping monitor (10% of training pixels held out)")
    fig.tight_layout()
    fig.savefig(out / "fig_mlp_convergence.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="results/scaffold")
    parser.add_argument("--size", type=int, default=256, help="grid side for features (256 = native)")
    parser.add_argument("--pixels-per-image", type=int, default=1500)
    parser.add_argument("--val-images", type=int, default=30, help="validation trees used for tuning")
    parser.add_argument("--max-train-images", type=int, default=None, help="limit for quick runs")
    parser.add_argument("--max-test-images", type=int, default=None, help="limit for quick runs")
    parser.add_argument("--fill-depth", action="store_true", help="fill depth holes first (Stage 2)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    clock = {}

    # ---------------- data
    splits = split_frames(args.folders, args.seed)
    train, val, test = splits["train"], splits["val"], splits["test"]
    if args.max_train_images:
        train = train.sample(min(args.max_train_images, len(train)), random_state=args.seed)
    val = val.sample(min(args.val_images, len(val)), random_state=args.seed)
    if args.max_test_images:
        test = test.sample(min(args.max_test_images, len(test)), random_state=args.seed)
    print(f"Trees: {len(train)} train, {len(val)} validation (tuning), {len(test)} test | "
          f"grid {args.size}x{args.size} | fill depth: {args.fill_depth}", flush=True)

    start = time.perf_counter()
    X, y, train_masks = [], [], []
    for frame in train.itertuples():
        rgb, depth, gt = load_frame(frame, args.size, args.fill_depth)
        feats = pixel_features(rgb, depth)
        idx = sample_pixels(gt, args.pixels_per_image, rng)
        X.append(feats[idx])
        y.append(gt.ravel()[idx])
        train_masks.append(gt)
    X, y = np.concatenate(X), np.concatenate(y)
    clock["features_train_s"] = time.perf_counter() - start
    print(f"Training pixels: {len(y):,} ({y.mean():.0%} scaffold), {X.shape[1]} features, "
          f"{clock['features_train_s']:.0f} s", flush=True)

    # ---------------- fit
    prior = PriorMask().fit(train_masks)
    models = make_models(args.seed)
    for name, (model, cols) in models.items():
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X[:, cols], y)
        clock[f"fit_{name}_s"] = time.perf_counter() - start
        note = ("  (hit the iteration limit before its stopping test was met)"
                if any(issubclass(w.category, ConvergenceWarning) for w in caught) else "")
        print(f"  fitted {name:15s} in {clock[f'fit_{name}_s']:.1f} s{note}", flush=True)
    mlp = models["mlp"][0].named_steps["model"]
    curve = pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1),
                          "training_loss": mlp.loss_curve_,
                          "validation_accuracy": mlp.validation_scores_})
    curve.to_csv(out / "mlp_curve.csv", index=False)

    # ---------------- tune on validation trees
    val_items = []
    for frame in val.itertuples():
        rgb, depth, _ = load_frame(frame, args.size, args.fill_depth)
        gt256 = data.load_gt(frame.path_GT, size=(EVAL_SIZE, EVAL_SIZE))
        val_items.append((rgb, depth, pixel_features(rgb, depth), gt256))
    val_gts = [item[3] for item in val_items]
    rule, rule_dice = ColourRule.tune([item[0] for item in val_items],
                                      [to_eval_grid_mask(g, args.size) for g in val_gts])
    thresholds = {"colour_rule": 0.5}
    for method in ("prior",) + LEARNED:
        probs = [to_eval_grid(predict_map(method, models, prior, rule, rgb, depth, feats, args.size))
                 for rgb, depth, feats, _ in val_items]
        thresholds[method], _ = evaluate.best_threshold(probs, val_gts)
    print("Thresholds chosen on validation trees:", json.dumps(thresholds), flush=True)
    print(f"Colour rule chosen: saturation <= {rule.max_saturation}, excess green <= {rule.max_exg}, "
          f"brightness >= {rule.min_brightness} (validation Dice {rule_dice:.3f}; marking every pixel "
          f"as scaffold would score {np.mean([evaluate.dice(np.ones_like(g), g) for g in val_gts]):.3f})",
          flush=True)
    edges = [name for name, grid in ColourRule.GRID.items()
             if getattr(rule, name) in (min(grid), max(grid))]
    if edges:
        print(f"  note: these settings sit at the end of their search range: {', '.join(edges)}",
              flush=True)

    # feature importance (random forest) on balanced validation pixels
    Xv, yv = [], []
    for rgb, depth, feats, gt256 in val_items:
        gt = to_eval_grid_mask(gt256, args.size)
        idx = sample_pixels(gt, 1000, rng)
        Xv.append(feats[idx])
        yv.append(gt.ravel()[idx])
    Xv, yv = np.concatenate(Xv), np.concatenate(yv)
    perm = permutation_importance(models["random_forest"][0], Xv, yv, scoring="roc_auc",
                                  n_repeats=3, random_state=args.seed)
    importance = pd.DataFrame({"feature": FEATURE_NAMES, "mean": perm.importances_mean,
                               "std": perm.importances_std})
    importance.to_csv(out / "importance.csv", index=False)
    del val_items, Xv, yv

    # ---------------- test (scored once)
    rows, examples, predict_ms = [], [], {m: [] for m in METHODS}
    example_ids = set(test.sample(min(3, len(test)), random_state=args.seed).tree_id)
    feature_ms = []
    for k, frame in enumerate(test.itertuples(), start=1):
        rgb, depth, _ = load_frame(frame, args.size, args.fill_depth)
        gt = data.load_gt(frame.path_GT, size=(EVAL_SIZE, EVAL_SIZE))
        start = time.perf_counter()
        feats = pixel_features(rgb, depth)
        feature_ms.append(1000 * (time.perf_counter() - start))
        probs = {}
        for method in METHODS:
            start = time.perf_counter()
            probs[method] = to_eval_grid(predict_map(method, models, prior, rule, rgb, depth,
                                                     feats, args.size))
            predict_ms[method].append(1000 * (time.perf_counter() - start))
            pred = evaluate.clean_mask(probs[method] >= thresholds[method])
            rows.append({"tree_id": frame.tree_id, "method": method, **evaluate.score_mask(pred, gt)})
        if frame.tree_id in example_ids:
            rgb256 = np.asarray(Image.fromarray(rgb).resize((EVAL_SIZE, EVAL_SIZE)))
            examples.append((frame.tree_id, rgb256, gt, probs))
        if k % 10 == 0 or k == len(test):
            print(f"  scored {k}/{len(test)} test trees", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    # ---------------- summary
    summary = []
    for method in METHODS:
        m = metrics[metrics.method == method]
        row = {"method": method, "threshold": thresholds[method],
               "predict_ms_per_image": float(np.mean(predict_ms[method])),
               "fit_s": clock.get(f"fit_{method}_s", 0.0)}
        for col in ("dice", "iou", "cl_f1_t3", "cl_f1_t5", "cl_f1_t10",
                    "cl_precision_t5", "cl_recall_t5"):
            row[f"{col}_mean"] = float(m[col].mean())
            row[f"{col}_lo"], row[f"{col}_hi"] = evaluate.bootstrap_ci(m[col], seed=args.seed)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({**vars(args), **clock,
                                                   "feature_ms_per_image": float(np.mean(feature_ms)),
                                                   "n_train_pixels": int(len(y)),
                                                   "colour_rule": vars(rule)}, indent=1, default=str))

    figure_scores(metrics, summary, out)
    figure_examples(examples, thresholds, out)
    figure_importance(importance, out)
    figure_mlp(curve, out)

    # ---------------- print
    pd.set_option("display.width", 140)
    print("\nTEST RESULTS (mean over test trees, [95% bootstrap CI])")
    for r in summary.sort_values("dice_mean", ascending=False).itertuples():
        print(f"  {LABELS[r.method]:24s} Dice {r.dice_mean:.3f} [{r.dice_lo:.3f}-{r.dice_hi:.3f}]  "
              f"IoU {r.iou_mean:.3f}  centreline F1@5px {r.cl_f1_t5_mean:.3f} "
              f"[{r.cl_f1_t5_lo:.3f}-{r.cl_f1_t5_hi:.3f}]  (@3px {r.cl_f1_t3_mean:.3f}, "
              f"@10px {r.cl_f1_t10_mean:.3f})")
    print("\nCOST")
    print(f"  features: {np.mean(feature_ms):.0f} ms per image; training-pixel extraction "
          f"{clock['features_train_s']:.0f} s")
    for r in summary.itertuples():
        print(f"  {LABELS[r.method]:24s} fit {r.fit_s:6.1f} s   predict {r.predict_ms_per_image:7.1f} ms/image")
    stopped = "early stopping" if mlp.n_iter_ < mlp.max_iter else f"the {mlp.max_iter}-epoch limit"
    print(f"\nMLP: {mlp.n_iter_} epochs (ended by {stopped}), best held-out pixel accuracy "
          f"{max(mlp.validation_scores_):.3f}")
    best_ml = summary[summary.method.isin(LEARNED)].sort_values("dice_mean").iloc[-1].method
    paired = (metrics[metrics.method == best_ml].set_index("tree_id").dice
              - metrics[metrics.method == "prior"].set_index("tree_id").dice)
    lo, hi = evaluate.bootstrap_ci(paired, seed=args.seed)
    print(f"\nPAIRED: {LABELS[best_ml]} minus prior baseline, Dice per tree: mean {paired.mean():+.3f} "
          f"[{lo:+.3f} to {hi:+.3f}], better on {int((paired > 0).sum())}/{len(paired)} trees")
    print("\nRANDOM FOREST FEATURE IMPORTANCE (drop in ROC AUC when shuffled):")
    for r in importance.sort_values("mean", ascending=False).itertuples():
        print(f"  {r.feature:11s} {r.mean:.3f} +/- {r.std:.3f}")
    print(f"\nWrote results to {out}")
    return 0


def to_eval_grid_mask(mask256: np.ndarray, size: int) -> np.ndarray:
    """Bring a 256-grid mask to the feature grid (nearest) for pixel sampling/tuning."""
    if size == EVAL_SIZE:
        return mask256
    return ndimage.zoom(mask256.astype(np.uint8), size / EVAL_SIZE, order=0).astype(bool)


if __name__ == "__main__":
    raise SystemExit(main())
