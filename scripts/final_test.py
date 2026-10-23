"""Stage 4d: the final test, run once on the untouched test trees.

Usage (from the repository root)::

    python scripts/final_test.py /content/data --out results/final

Protocol (Week 5.5.10): the configuration fixed in Stage 4c
(``results/shape_settings/final_config.json``) is retrained on all training
and validation trees together, then scored once on the 85 test trees.

Compared on the same test trees, tree by tree:

* **Shape method (final)** - the configuration chosen by the Stage 4c rules
  (primary result);
* **Shape method, fast variant** - no bark map, M = 200 Monte Carlo shapes,
  declared in advance as a secondary result for the speed-accuracy trade-off;
* **Average mask (train+val)** - the baseline, refitted on the same 476 trees;
* every **Stage 3** method, from the per-tree test scores Stage 3 saved
  (those models were trained on the 392 training trees only).

Also reports the time per photo of both shape pipelines (each step timed on
its own), the best / median / worst test trees, and how scores change with an
occlusion proxy (share of the scaffold whose colour looks like leaves or fruit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import evaluate, shapes  # noqa: E402
from apple_scaffold.baselines import PriorMask  # noqa: E402
from apple_scaffold.features import load_frame, pixel_features  # noqa: E402
from apple_scaffold.shape_pipeline import (IMAGE_ONLY, SIZE, band_probabilities,  # noqa: E402
                                           bark_classifier, bark_maps, describe_masks,
                                           first_pass, out_of_fold_bark_maps, per_tree_scores,
                                           ridge, split_frames, tune_band_threshold)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GREY, LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#9ec3ef"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
})
LABELS = {"shape_final": "Shape method (final)", "shape_fast": "Shape method, fast variant",
          "avg_mask": "Average mask (train+val)", "s3_prior": "Stage 3: average mask",
          "s3_colour_rule": "Stage 3: colour rule", "s3_logistic": "Stage 3: logistic regression",
          "s3_random_forest": "Stage 3: random forest",
          "s3_rf_no_position": "Stage 3: random forest, no x/y", "s3_mlp": "Stage 3: neural network"}
GROUP = {"shape_final": "shape", "shape_fast": "shape", "avg_mask": "baseline",
         "s3_prior": "baseline", "s3_colour_rule": "baseline"}
COLOUR = {"shape": PALETTE[0], "baseline": GREY, "stage3": LIGHT}


def log(message: str) -> None:
    print(message, flush=True)


def shape_pipeline(X_dev, X_test, Y, model, widths, M, folds, tune_idx, dev_masks, seed):
    """Fit on all development trees; tune the band threshold on development trees
    (out-of-fold); return test probability maps and the fitted pieces."""
    oof = cross_val_predict(ridge(), X_dev, Y, cv=folds)
    estimator = ridge().fit(X_dev, Y)
    cov = np.atleast_2d(np.cov((Y - oof).T))
    threshold = tune_band_threshold(oof[tune_idx], cov, model, widths,
                                    [dev_masks[i] for i in tune_idx], max(M // 2, 10), seed)
    pred = estimator.predict(X_test)
    probs = band_probabilities(pred, cov, model, widths, M, seed)
    masks = [evaluate.clean_mask(p >= threshold) for p in probs]
    return {"estimator": estimator, "cov": cov, "threshold": threshold, "probs": probs,
            "masks": masks, "alpha": float(estimator.named_steps["model"].alpha_),
            "weighted_r2": shapes.weighted_r2(Y, oof)}


def paired(metrics, a, b, column, seed):
    x = metrics[metrics.method == a].set_index("tree_id")[column]
    y = metrics[metrics.method == b].set_index("tree_id")[column]
    diff = (x - y).dropna()
    lo, hi = evaluate.bootstrap_ci(diff, seed=seed)
    return {"comparison": f"{LABELS[a]} minus {LABELS[b]}", "metric": column, "mean": diff.mean(),
            "lo": lo, "hi": hi, "better": int((diff > 0).sum()), "trees": len(diff)}


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--config", default="results/shape_settings/final_config.json")
    parser.add_argument("--stage3", default="results/scaffold/metrics.csv")
    parser.add_argument("--out", default="results/final")
    args = parser.parse_args(argv)
    if not Path(args.config).exists():
        raise SystemExit(f"{args.config} not found: run Stage 4c first.")
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    K, M, grid, rows = cfg["modes_K"], cfg["monte_carlo_M"], cfg["summary_grid"], cfg["centreline_rows"]
    log(f"Configuration from Stage 4c: bark map {'yes' if cfg['use_bark_map'] else 'no'}, K={K}, "
        f"grid {grid}x{grid}, {rows} rows, M={M}")

    splits = split_frames(args.folders, seed)
    dev = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    log(f"Development trees (train + validation): {len(dev)}; test trees: {len(test)}")

    start = time.perf_counter()
    dev_masks, coarse_dev, Xpix, ypix = first_pass(dev, rng, cfg["pixels_per_tree"], grid=grid)
    test_masks, coarse_test, _, _ = first_pass(test, rng, 10, grid=grid)
    log(f"Loaded trees in {time.perf_counter() - start:.0f} s")
    folds = KFold(5, shuffle=True, random_state=seed)

    clf = None
    if cfg["use_bark_map"]:
        start = time.perf_counter()
        maps_dev = out_of_fold_bark_maps(dev, Xpix, ypix, folds, seed, log=log)
        clf = bark_classifier(seed).fit(np.concatenate(Xpix), np.concatenate(ypix))
        maps_test = bark_maps(test, range(len(test)), clf)
        bark_dev = np.stack([shapes.map_descriptor(maps_dev[i]) for i in range(len(dev))])
        bark_test = np.stack([shapes.map_descriptor(maps_test[i]) for i in range(len(test))])
        X_dev, X_test = np.hstack([coarse_dev, bark_dev]), np.hstack([coarse_test, bark_test])
        log(f"Bark maps made in {time.perf_counter() - start:.0f} s")
    else:
        X_dev, X_test = coarse_dev, coarse_test

    V, widths = describe_masks(dev_masks, rows)
    model = shapes.ShapeModel().fit(V)
    Y = model.encode(V, K)
    tune_idx = np.sort(rng.choice(len(dev), min(cfg["tune_trees"], len(dev)), replace=False))

    # ---------------- the methods, scored once on the test trees
    final = shape_pipeline(X_dev, X_test, Y, model, widths, M, folds, tune_idx, dev_masks, seed)
    fast = shape_pipeline(coarse_dev, coarse_test, Y, model, widths, 200, folds, tune_idx,
                          dev_masks, seed)
    prior = PriorMask().fit(dev_masks)
    t_prior = evaluate.best_threshold([prior.predict_proba()] * len(tune_idx),
                                      [dev_masks[i] for i in tune_idx])[0]
    prior_masks = [evaluate.clean_mask(prior.predict_proba() >= t_prior)] * len(test)

    frames = []
    for key, masks in (("shape_final", final["masks"]), ("shape_fast", fast["masks"]),
                       ("avg_mask", prior_masks)):
        scores = per_tree_scores(masks, test_masks)
        frames.append(scores.assign(tree_id=test.tree_id.to_numpy(), method=key))
    stage3_found = Path(args.stage3).exists()
    if stage3_found:
        s3 = pd.read_csv(args.stage3)
        if set(s3.tree_id) != set(test.tree_id):
            log("WARNING: Stage 3 test trees differ from these test trees; Stage 3 rows skipped")
            stage3_found = False
        else:
            s3 = s3.assign(method="s3_" + s3.method)
            frames.append(s3[frames[0].columns])
    else:
        log(f"Note: {args.stage3} not found; Stage 3 methods are not compared")
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(out / "final_metrics.csv", index=False)

    summary = []
    for method, m in metrics.groupby("method", sort=False):
        row = {"method": method, "label": LABELS[method]}
        for col in ("dice", "iou", "cl_f1_t3", "cl_f1_t5", "cl_f1_t10"):
            row[col] = m[col].mean()
        row["dice_lo"], row["dice_hi"] = evaluate.bootstrap_ci(m.dice, seed=seed)
        row["cl_lo"], row["cl_hi"] = evaluate.bootstrap_ci(m.cl_f1_t5, seed=seed)
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values("dice", ascending=False)
    summary.to_csv(out / "final_summary.csv", index=False)

    comparisons = [("shape_final", "avg_mask"), ("shape_fast", "avg_mask"),
                   ("shape_final", "shape_fast")]
    if stage3_found:
        comparisons += [("shape_final", "s3_prior"), ("shape_final", "s3_mlp"),
                        ("shape_final", "s3_random_forest")]
    pairs = pd.DataFrame([paired(metrics, a, b, col, seed) for a, b in comparisons
                          for col in ("dice", "cl_f1_t5")])
    pairs.to_csv(out / "paired.csv", index=False)

    # ---------------- time per photo, each step timed on its own
    steps = {name: {"load + resize photo": [], "photo summary": [], "bark map": [],
                    "ridge prediction": [], "Monte Carlo band": [], "threshold + clean": []}
             for name in ("shape_final", "shape_fast")}
    covered, rgbs = [], []
    for i in range(len(test)):
        t = time.perf_counter()
        rgb, depth, _ = load_frame(test.iloc[i], SIZE)
        load_s = time.perf_counter() - t
        rgbs.append(rgb)
        covered.append(evaluate.covered_fraction(rgb, test_masks[i]))
        if i >= 20:
            continue
        for name, run, use_bark, m_samples in (("shape_final", final, cfg["use_bark_map"], M),
                                               ("shape_fast", fast, False, 200)):
            s = steps[name]
            s["load + resize photo"].append(load_s)
            t = time.perf_counter()
            x = shapes.image_descriptor(rgb, depth, grid=grid)
            s["photo summary"].append(time.perf_counter() - t)
            if use_bark:
                t = time.perf_counter()
                prob = clf.predict_proba(pixel_features(rgb, depth)[:, IMAGE_ONLY])[:, 1]
                x = np.concatenate([x, shapes.map_descriptor(prob.reshape(SIZE, SIZE))])
                s["bark map"].append(time.perf_counter() - t)
            t = time.perf_counter()
            c = run["estimator"].predict(x[None])[0]
            s["ridge prediction"].append(time.perf_counter() - t)
            t = time.perf_counter()
            p = shapes.probability_map(c, run["cov"], model, widths, m_samples, seed)
            s["Monte Carlo band"].append(time.perf_counter() - t)
            t = time.perf_counter()
            evaluate.clean_mask(p >= run["threshold"])
            s["threshold + clean"].append(time.perf_counter() - t)
    timing = pd.DataFrame([{"pipeline": name, "step": step,
                            "ms_per_photo": 1000 * np.mean(v) if v else 0.0}
                           for name, s in steps.items() for step, v in s.items()])
    timing.to_csv(out / "timing.csv", index=False)

    # ---------------- occlusion
    occ = pd.DataFrame({"tree_id": test.tree_id, "covered": covered})
    per = metrics.pivot(index="tree_id", columns="method", values="dice").loc[test.tree_id]
    rho_final = spearmanr(occ.covered, per["shape_final"].to_numpy())
    rho_base = spearmanr(occ.covered, per["avg_mask"].to_numpy())
    rho_gain = spearmanr(occ.covered, (per["shape_final"] - per["avg_mask"]).to_numpy())
    occ.assign(dice_final=per["shape_final"].to_numpy(),
               dice_baseline=per["avg_mask"].to_numpy()).to_csv(out / "occlusion.csv", index=False)

    (out / "run_info.json").write_text(json.dumps({
        "config": cfg, "n_dev": len(dev), "n_test": len(test),
        "threshold_final": final["threshold"], "threshold_fast": fast["threshold"],
        "threshold_avg_mask": t_prior, "ridge_alpha_final": final["alpha"],
        "ridge_alpha_fast": fast["alpha"], "dev_cv_weighted_r2_final": final["weighted_r2"],
        "stage3_compared": stage3_found}, indent=1, default=float))

    # ---------------- print
    log(f"\nFINAL TEST RESULTS ({len(test)} untouched test trees; mean [95% bootstrap CI])")
    for r in summary.itertuples():
        log(f"  {r.label:34s} Dice {r.dice:.3f} [{r.dice_lo:.3f}-{r.dice_hi:.3f}]  IoU {r.iou:.3f}  "
            f"centreline F1@5px {r.cl_f1_t5:.3f} [{r.cl_lo:.3f}-{r.cl_hi:.3f}]  "
            f"(@3px {r.cl_f1_t3:.3f}, @10px {r.cl_f1_t10:.3f})")
    log("\nPAIRED, tree by tree (positive = first method better)")
    for r in pairs.itertuples():
        name = "Dice" if r.metric == "dice" else "centreline F1"
        log(f"  {r.comparison:72s} {name:13s} {r.mean:+.3f} [{r.lo:+.3f} to {r.hi:+.3f}], "
            f"better on {r.better}/{r.trees}")
    log(f"\nTIME PER PHOTO (ms, mean over {min(20, len(test))} test photos)")
    for name in ("shape_final", "shape_fast"):
        t = timing[timing.pipeline == name]
        parts = ", ".join(f"{r.step} {r.ms_per_photo:.0f}" for r in t.itertuples() if r.ms_per_photo > 0)
        log(f"  {LABELS[name]:30s} total {t.ms_per_photo.sum():6.0f}   ({parts})")
    log("\nOCCLUSION (share of scaffold coloured like leaves or fruit; Spearman correlation)")
    log(f"  covered fraction: median {np.median(covered):.2f}, range {np.min(covered):.2f}-{np.max(covered):.2f}")
    log(f"  final Dice vs covered:      rho = {rho_final.statistic:+.2f} (p = {rho_final.pvalue:.3f})")
    log(f"  baseline Dice vs covered:   rho = {rho_base.statistic:+.2f} (p = {rho_base.pvalue:.3f})")
    log(f"  gain over baseline vs covered: rho = {rho_gain.statistic:+.2f} (p = {rho_gain.pvalue:.3f})")

    figures(metrics, summary, test, rgbs, test_masks, final, prior_masks, occ, per,
            rho_final, rho_base, out)
    log(f"\nWrote results to {out}")
    return 0


# --------------------------------------------------------------------------
def figures(metrics, summary, test, rgbs, test_masks, final, prior_masks, occ, per,
            rho_final, rho_base, out):
    # scores of every method on the test trees
    order = summary.sort_values("dice")["method"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 0.42 * len(order) + 1.3), sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, [("dice", "dice_lo", "dice_hi", "Dice"),
                                               ("cl_f1_t5", "cl_lo", "cl_hi", "Centreline F1 (5 px)")]):
        for i, method in enumerate(order):
            values = metrics.loc[metrics.method == method, col].to_numpy()
            colour = COLOUR[GROUP.get(method, "stage3")]
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(values))
            ax.scatter(values, i + jitter, s=5, color=colour, alpha=0.25, linewidths=0)
            r = summary.set_index("method").loc[method]
            ax.errorbar(r[col], i, xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt="o",
                        color=colour, markersize=6, capsize=3)
        ax.set(title=title, xlim=(0, 1), xlabel="test trees (dot = mean, bar = 95% CI)")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABELS[m] for m in order], fontsize=7)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=l)
               for l, c in (("baseline", GREY), ("Stage 3 pixel models", LIGHT), ("shape method", PALETTE[0]))]
    fig.suptitle(f"Final test: every method on the same {len(test)} untouched trees", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
    fig.savefig(out / "fig_final_scores.png")
    plt.close(fig)

    # best, median and worst test trees for the final method
    dice = per["shape_final"].to_numpy()
    ranked = np.argsort(dice)
    picks = [("best", ranked[-1]), ("median", ranked[len(ranked) // 2]), ("worst", ranked[0])]
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.6))
    for row, (name, i) in enumerate(picks):
        truth = test_masks[i]
        outline = truth ^ ndimage.binary_erosion(truth, iterations=2)
        ys, xs = np.nonzero(outline)
        panels = [("photo + truth", rgbs[i], None),
                  (f"average mask (Dice {per['avg_mask'].iloc[i]:.2f})", prior_masks[i], "mask"),
                  ("fuzzy band (probability)", final["probs"][i], "prob"),
                  (f"final mask (Dice {dice[i]:.2f})", final["masks"][i], "mask")]
        for col, (title, image, kind) in enumerate(panels):
            ax = axes[row, col]
            if kind is None:
                ax.imshow(image)
                ax.scatter(xs, ys, s=0.15, color="#ffffff", linewidths=0)
            elif kind == "prob":
                ax.imshow(image, cmap="Blues", vmin=0, vmax=1)
                ax.scatter(xs, ys, s=0.15, color=PALETTE[1], linewidths=0)
            else:
                ax.imshow(np.where(image, 0.75, 0.0), cmap="Blues", vmin=0, vmax=1)
                ax.scatter(xs, ys, s=0.15, color=PALETTE[1], linewidths=0)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{name}: {test.tree_id.iloc[i]}", fontsize=8)
    fig.suptitle("Final method on the best, median and worst test trees (truth outline in orange)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_best_median_worst.png")
    plt.close(fig)

    # occlusion
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(occ.covered, per["avg_mask"], s=14, color=GREY, alpha=0.8,
               label=f"average mask (rho {rho_base.statistic:+.2f})")
    ax.scatter(occ.covered, per["shape_final"], s=14, color=PALETTE[0], alpha=0.8,
               label=f"shape method (rho {rho_final.statistic:+.2f})")
    ax.set(xlabel="share of the scaffold coloured like leaves or fruit (occlusion proxy)",
           ylabel="Dice per test tree", title="Does hiding by leaves hurt?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_occlusion.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
