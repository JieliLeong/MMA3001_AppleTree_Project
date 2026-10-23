"""Stage 4b: predict the scaffold as a shape from the photo, drawn as a fuzzy band.

Usage (from the repository root)::

    python scripts/shape_method.py /content/data --out results/shape_method

Development uses training and validation trees only; the test trees are kept
for the single final evaluation (Stage 4c).

Pipeline
--------
1. **Shape targets.** Each training mask becomes two centreline curves; the SVD
   shape model (Week 2.3) turns them into ``K`` mode coefficients per tree.
2. **Inputs per tree.**
   * ``coarse`` - whole-image summary of brightness, greenness, saturation and
     depth (Stage 4a).
   * ``bark`` - summary of a bark-probability map from an image-only pixel
     classifier (random forest without x/y). Training trees get *out-of-fold*
     maps: the classifier that drew a tree's map never saw that tree, so the
     regression below learns from maps as noisy as the ones it will meet.
3. **Regressors** for the ``K`` coefficients: ridge, KNN (Week 3.6), random
   forest and MLP (Week 5), plus their average. 5-fold cross-validation on the
   training trees gives out-of-fold predictions, from which come R^2 per mode
   and the error covariance of each model.
4. **Fuzzy band.** Shapes are sampled around the prediction with that error
   covariance (Monte Carlo) and averaged into a probability map. Its threshold
   is chosen on training trees (out-of-fold), then everything is scored on the
   validation trees.

Writes ``ladder.csv``, ``mode_r2.csv``, ``selected.json`` and ``fig_*.png``.
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
from scipy import ndimage
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import data, evaluate, shapes  # noqa: E402
from apple_scaffold.baselines import PriorMask  # noqa: E402
from apple_scaffold.features import (FEATURE_NAMES, POSITION_FEATURES, load_frame,  # noqa: E402
                                     pixel_features, sample_pixels)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE = 256
IMAGE_ONLY = [i for i, n in enumerate(FEATURE_NAMES) if n not in POSITION_FEATURES]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GREY, LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#9ec3ef"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
})
MODEL_LABELS = {"ridge_coarse": "Ridge, coarse summary", "ridge_bark": "Ridge, + bark map",
                "knn": "KNN, + bark map", "forest": "Random forest, + bark map",
                "mlp": "Neural network, + bark map", "average": "Average of the four models"}


# --------------------------------------------------------------------------
# Data and inputs
# --------------------------------------------------------------------------
def split_frames(folders, seed):
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    frames = frames[(frames["problems"] == "") & frames["path_GT"].notna()].reset_index(drop=True)
    labels = data.grouped_split(frames, seed=seed)
    return frames[labels == "train"].reset_index(drop=True), frames[labels == "val"].reset_index(drop=True)


def first_pass(frames, rng, pixels_per_tree):
    """Masks, coarse descriptors and a balanced pixel sample (image-only features) per tree."""
    masks, coarse, Xpix, ypix = [], [], [], []
    for frame in frames.itertuples():
        rgb, depth, gt = load_frame(frame, SIZE)
        feats = pixel_features(rgb, depth)
        idx = sample_pixels(gt, pixels_per_tree, rng)
        masks.append(gt)
        coarse.append(shapes.image_descriptor(rgb, depth))
        Xpix.append(feats[idx][:, IMAGE_ONLY])
        ypix.append(gt.ravel()[idx])
    return masks, np.stack(coarse), Xpix, ypix


def bark_classifier(seed):
    return RandomForestClassifier(n_estimators=40, max_depth=14, min_samples_leaf=10,
                                  n_jobs=-1, random_state=seed)


def bark_maps(frames, indices, model):
    """Bark-probability maps (image-only classifier) for the given trees."""
    maps = {}
    for i in indices:
        frame = frames.iloc[i]
        rgb, depth, _ = load_frame(frame, SIZE)
        prob = model.predict_proba(pixel_features(rgb, depth)[:, IMAGE_ONLY])[:, 1]
        maps[i] = prob.reshape(SIZE, SIZE).astype(np.float32)
    return maps


# --------------------------------------------------------------------------
# Regressors
# --------------------------------------------------------------------------
def regressors(seed, n_train):
    """Name -> (input set, estimator). Scaling and PCA sit inside each pipeline,
    so cross-validation refits them on every fold (no leakage)."""
    smallest_fold = int(n_train * 0.8 * 0.8)          # inner training size in nested CV
    n_components = min(30, smallest_fold - 1)

    def compact(model):
        return Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components, random_state=seed)), ("model", model)])
    knn = GridSearchCV(compact(KNeighborsRegressor(weights="distance")),
                       {"model__n_neighbors": [k for k in (5, 10, 20, 40) if k < smallest_fold]},
                       cv=5)
    mlp = TransformedTargetRegressor(
        regressor=compact(MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000,
                                       learning_rate_init=1e-3, random_state=seed)),
        transformer=StandardScaler())
    return {
        "ridge_coarse": ("coarse", Pipeline([("scale", StandardScaler()),
                                             ("model", RidgeCV(alphas=np.logspace(-1, 5, 25)))])),
        "ridge_bark": ("both", Pipeline([("scale", StandardScaler()),
                                         ("model", RidgeCV(alphas=np.logspace(-1, 5, 25)))])),
        "knn": ("both", knn),
        "forest": ("both", compact(RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                         max_features=0.5, n_jobs=-1,
                                                         random_state=seed))),
        "mlp": ("both", mlp),
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_set(pred_masks, masks, seed):
    rows = [evaluate.score_mask(p, t, tolerances=(5,)) for p, t in zip(pred_masks, masks)]
    dice = np.array([r["dice"] for r in rows])
    cl = np.array([r["cl_f1_t5"] for r in rows])
    return {"dice": dice.mean(), "dice_lo": evaluate.bootstrap_ci(dice, seed=seed)[0],
            "dice_hi": evaluate.bootstrap_ci(dice, seed=seed)[1], "cl_f1": cl.mean(),
            "cl_lo": evaluate.bootstrap_ci(cl, seed=seed)[0],
            "cl_hi": evaluate.bootstrap_ci(cl, seed=seed)[1], "per_tree_dice": dice}


def band_masks(coeffs, cov, model, widths, threshold, n_samples, seed):
    return [evaluate.clean_mask(shapes.probability_map(c, cov, model, widths, n_samples,
                                                       seed + i) >= threshold)
            for i, c in enumerate(coeffs)]


def tune_band_threshold(coeffs, cov, model, widths, masks, n_samples, seed):
    maps = [shapes.probability_map(c, cov, model, widths, n_samples, seed + i)
            for i, c in enumerate(coeffs)]
    return evaluate.best_threshold(maps, masks)[0]


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="results/shape_method")
    parser.add_argument("--modes", type=int, default=8, help="shape modes predicted (K)")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--samples", type=int, default=200, help="Monte Carlo shapes per band")
    parser.add_argument("--pixels-per-tree", type=int, default=500)
    parser.add_argument("--tune-trees", type=int, default=100,
                        help="training trees used to choose band thresholds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    clock = {}
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    train, val = split_frames(args.folders, args.seed)
    print(f"Trees: {len(train)} train, {len(val)} validation (test trees not used)", flush=True)

    # ---------------- 1. shape targets
    start = time.perf_counter()
    train_masks, coarse_tr, Xpix, ypix = first_pass(train, rng, args.pixels_per_tree)
    val_masks, coarse_va, _, _ = first_pass(val, rng, 10)
    described = [shapes.describe(m, args.rows) for m in train_masks]
    V = np.stack([d[0] for d in described])
    widths = np.nanmean(np.stack([d[1] for d in described]), axis=0)
    shape_model = shapes.ShapeModel().fit(V)
    K = args.modes
    Y = shape_model.encode(V, K)
    clock["load_s"] = time.perf_counter() - start
    print(f"Loaded and described trees in {clock['load_s']:.0f} s; {K} modes capture "
          f"{shape_model.explained_variance_ratio_[:K].sum():.0%} of shape variation", flush=True)

    # ---------------- 2. bark maps: out-of-fold for training trees
    start = time.perf_counter()
    folds = KFold(5, shuffle=True, random_state=args.seed)
    maps_tr = {}
    for f, (fit_idx, held_idx) in enumerate(folds.split(np.arange(len(train))), start=1):
        clf = bark_classifier(args.seed).fit(np.concatenate([Xpix[i] for i in fit_idx]),
                                             np.concatenate([ypix[i] for i in fit_idx]))
        maps_tr.update(bark_maps(train, held_idx, clf))
        print(f"  bark maps: fold {f}/5 done", flush=True)
    clf = bark_classifier(args.seed).fit(np.concatenate(Xpix), np.concatenate(ypix))
    maps_va = bark_maps(val, range(len(val)), clf)
    bark_tr = np.stack([shapes.map_descriptor(maps_tr[i]) for i in range(len(train))])
    bark_va = np.stack([shapes.map_descriptor(maps_va[i]) for i in range(len(val))])
    inputs = {"coarse": (coarse_tr, coarse_va),
              "both": (np.hstack([coarse_tr, bark_tr]), np.hstack([coarse_va, bark_va]))}
    clock["bark_s"] = time.perf_counter() - start
    print(f"Bark maps made in {clock['bark_s']:.0f} s", flush=True)

    # ---------------- 3. regressors
    tune_idx = np.sort(rng.choice(len(train), min(args.tune_trees, len(train)), replace=False))
    tune_masks = [train_masks[i] for i in tune_idx]
    results, predictions = {}, {}
    for name, (input_set, estimator) in regressors(args.seed, len(train)).items():
        start = time.perf_counter()
        X_tr, X_va = inputs[input_set]
        oof = cross_val_predict(estimator, X_tr, Y, cv=folds)
        estimator.fit(X_tr, Y)
        predictions[name] = (oof, estimator.predict(X_va))
        clock[f"fit_{name}_s"] = time.perf_counter() - start
        print(f"  {name:13s} cross-validated in {clock[f'fit_{name}_s']:.1f} s", flush=True)
    names = list(predictions)
    predictions["average"] = (np.mean([predictions[n][0] for n in names if n != "ridge_coarse"], axis=0),
                              np.mean([predictions[n][1] for n in names if n != "ridge_coarse"], axis=0))

    # ---------------- 4. ladder on validation trees
    ladder, per_tree = [], {}

    def add(label, kind, pred_masks, extra=None):
        s = score_set(pred_masks, val_masks, args.seed)
        per_tree[label] = s.pop("per_tree_dice")
        ladder.append({"method": label, "kind": kind, **(extra or {}), **s})
        print(f"  {label:44s} Dice {s['dice']:.3f} [{s['dice_lo']:.3f}-{s['dice_hi']:.3f}]   "
              f"centreline F1 {s['cl_f1']:.3f}", flush=True)

    print("\nVALIDATION LADDER", flush=True)
    prior = PriorMask().fit(train_masks)
    t_prior = evaluate.best_threshold([prior.predict_proba()] * len(tune_masks), tune_masks)[0]
    add("Baseline: average mask (Stage 3)", "baseline",
        [evaluate.clean_mask(prior.predict_proba() >= t_prior)] * len(val_masks), {"threshold": t_prior})
    add("Mean shape, thin line (no photo)", "baseline",
        [shapes.draw(shape_model.mean_, widths)] * len(val_masks))
    prior_cov = np.cov(Y.T)
    t_np = tune_band_threshold(np.zeros((len(tune_idx), K)), prior_cov, shape_model, widths,
                               tune_masks, args.samples // 2, args.seed)
    add("Band with no photo information (check)", "baseline",
        band_masks(np.zeros((len(val), K)), prior_cov, shape_model, widths, t_np,
                   args.samples, args.seed), {"threshold": t_np})

    mode_r2 = []
    for name in ["ridge_coarse", "ridge_bark", "knn", "forest", "mlp", "average"]:
        oof, pred_va = predictions[name]
        r2 = 1 - ((Y - oof) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        wr2 = shapes.weighted_r2(Y, oof)
        mode_r2.append({"model": name, "weighted_r2": wr2, **{f"mode_{i + 1}": v for i, v in enumerate(r2)}})
        cov = np.cov((Y - oof).T)
        if name == "ridge_coarse":
            add(f"{MODEL_LABELS[name]}: thin line", "thin",
                [shapes.draw(v, widths) for v in shape_model.decode(pred_va)], {"weighted_r2": wr2})
        threshold = tune_band_threshold(oof[tune_idx], cov, shape_model, widths, tune_masks,
                                        args.samples // 2, args.seed)
        add(f"{MODEL_LABELS[name]}: band", "band",
            band_masks(pred_va, cov, shape_model, widths, threshold, args.samples, args.seed),
            {"weighted_r2": wr2, "threshold": threshold})

    ladder = pd.DataFrame(ladder)
    ladder.to_csv(out / "ladder.csv", index=False)
    mode_r2 = pd.DataFrame(mode_r2)
    mode_r2.to_csv(out / "mode_r2.csv", index=False)

    bands = ladder[ladder.kind == "band"]
    best = bands.sort_values("dice").iloc[-1]
    best_key = next(k for k, v in MODEL_LABELS.items() if best.method.startswith(v + ":"))
    baseline_dice = per_tree["Baseline: average mask (Stage 3)"]
    paired = per_tree[best.method] - baseline_dice
    lo, hi = evaluate.bootstrap_ci(paired, seed=args.seed)
    (out / "selected.json").write_text(json.dumps({
        "model": best_key, "threshold": float(best.threshold), "modes": K, "rows": args.rows,
        "samples": args.samples, "pixels_per_tree": args.pixels_per_tree, "seed": args.seed,
        "validation_dice": float(best.dice), "validation_cl_f1": float(best.cl_f1),
        **{k: round(v, 2) for k, v in clock.items()}}, indent=1))

    print("\nSHAPE MODES PREDICTED FROM THE PHOTO (cross-validated R^2 on training trees)")
    show = [f"mode_{i}" for i in range(1, 6)]
    print("  " + "model".ljust(30) + "  pooled " + "  ".join(m.replace("mode_", "mode ") for m in show))
    for r in mode_r2.itertuples():
        print(f"  {MODEL_LABELS[r.model]:30s}  {r.weighted_r2:6.2f} "
              + "  ".join(f"{getattr(r, m):6.2f}" for m in show))
    print(f"\nSELECTED on validation Dice: {best.method}")
    print(f"PAIRED vs average-mask baseline (validation trees): Dice {paired.mean():+.3f} "
          f"[{lo:+.3f} to {hi:+.3f}], better on {int((paired > 0).sum())}/{len(paired)} trees")
    print("COST: " + ", ".join(f"{k} {v:.0f} s" for k, v in clock.items()))

    figures(ladder, mode_r2, val, val_masks, maps_va, prior, t_prior, predictions[best_key][1],
            np.cov((Y - predictions[best_key][0]).T), float(best.threshold), shape_model,
            widths, MODEL_LABELS[best_key], args, out)
    print(f"\nWrote results to {out}")
    return 0


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figures(ladder, mode_r2, val, val_masks, maps_va, prior, t_prior, pred_va, cov, threshold,
            shape_model, widths, best_label, args, out):
    # ladder
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.38 * len(ladder) + 1.2), sharey=True)
    y = np.arange(len(ladder))[::-1]
    colour = {"baseline": GREY, "thin": LIGHT, "band": PALETTE[0]}
    for ax, (col, lo, hi, title) in zip(axes, [("dice", "dice_lo", "dice_hi", "Dice"),
                                               ("cl_f1", "cl_lo", "cl_hi", "Centreline F1 (5 px)")]):
        for yy, r in zip(y, ladder.itertuples()):
            ax.errorbar(getattr(r, col), yy, xerr=[[getattr(r, col) - getattr(r, lo)],
                                                   [getattr(r, hi) - getattr(r, col)]],
                        fmt="o", color=colour[r.kind], capsize=3, markersize=5)
        ax.axvline(ladder.iloc[0][col], color=GREY, linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="validation trees (dot = mean, bar = 95% CI)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ladder.method, fontsize=7)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=l)
               for l, c in (("baseline", GREY), ("thin predicted line", LIGHT), ("fuzzy band", PALETTE[0]))]
    fig.suptitle("Improvement ladder (dashed line: average-mask baseline)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
    fig.savefig(out / "fig_ladder.png")
    plt.close(fig)

    # R^2 per mode
    models = ["ridge_bark", "knn", "forest", "mlp"]
    modes = [f"mode_{i}" for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.2
    for j, m in enumerate(models):
        row = mode_r2.set_index("model").loc[m]
        ax.bar(np.arange(5) + (j - 1.5) * width, [row[c] for c in modes], width * 0.9,
               color=PALETTE[j], label=MODEL_LABELS[m])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ev = shape_model.explained_variance_ratio_
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"mode {i + 1}\n({ev[i]:.0%} of variation)" for i in range(5)], fontsize=7)
    ax.set(ylabel="cross-validated R^2", title="How well each shape mode is predicted from the photo")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_mode_r2.png")
    plt.close(fig)

    # bands on example validation trees
    show = np.linspace(0, len(val) - 1, 4).astype(int)
    fig, axes = plt.subplots(len(show), 4, figsize=(8.2, 2.1 * len(show)), squeeze=False)
    titles = ["photo + truth", "bark-probability map", "average mask (baseline)", f"band: {best_label}"]
    for row, i in enumerate(show):
        rgb, _, _ = load_frame(val.iloc[i], SIZE)
        truth = val_masks[i]
        outline = truth ^ ndimage.binary_erosion(truth, iterations=2)
        band = shapes.probability_map(pred_va[i], cov, shape_model, widths, args.samples, args.seed + i)
        panels = [rgb, maps_va[i], prior.predict_proba() >= t_prior, band]
        for col, image in enumerate(panels):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(rgb)
            elif col == 1:
                ax.imshow(image, cmap="Blues", vmin=0, vmax=1)
            elif col == 2:
                ax.imshow(np.where(image, 0.6, 0.0), cmap="Greys", vmin=0, vmax=1)
            else:
                ax.imshow(image, cmap="Blues", vmin=0, vmax=1)
            ys, xs = np.nonzero(outline)
            ax.scatter(xs, ys, s=0.15, color=PALETTE[1] if col else "#ffffff", linewidths=0)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(val.tree_id[i], fontsize=7)
    fig.suptitle("Validation trees; ground-truth outline in orange", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_bands.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
