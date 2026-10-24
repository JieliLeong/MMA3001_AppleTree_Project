"""Stage 4c: choose the shape method's settings and measure its performance.

Usage (from the repository root)::

    python scripts/shape_settings.py /content/data --out results/shape_settings

Everything here uses training and validation trees only, so the shape method
is scored on the test trees only once (Stage 4d). Settings are varied one at
a time around a base configuration (ridge, coarse summary on a 16 x 16 grid,
K = 8 modes, 32 centreline rows, M = 200 Monte Carlo shapes), and each is
chosen by a rule stated *before* looking at the numbers:

* **bark map** - kept only if its paired gain in validation Dice has a 95 %
  confidence interval entirely above zero;
* **K, summary grid, centreline rows** - the simplest value whose mean
  validation Dice is within 0.005 of the best;
* **M (Monte Carlo shapes)** - the smallest M whose band error (RMS, against a
  6400-shape reference) is at most 0.01 - one percentage point of probability -
  and whose validation Dice is within 0.002 of the largest M tested.

Performance (brief section 6): Monte Carlo convergence (error ~ M^-1/2),
learning curve (data requirements), repeatability over seeds, run time per
pipeline step, original (loop) vs vectorised band drawing, FLOP / arithmetic-
intensity estimates and model memory.

Writes CSVs, ``final_config.json`` and ``fig_*.png`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import evaluate, shapes  # noqa: E402
from apple_scaffold.baselines import PriorMask  # noqa: E402
from apple_scaffold.features import load_frame, pixel_features  # noqa: E402
from apple_scaffold.shape_pipeline import (IMAGE_ONLY, SIZE, band_masks,  # noqa: E402
                                           bark_classifier, bark_maps, coarse_descriptors,
                                           describe_masks, first_pass, out_of_fold_bark_maps,
                                           per_tree_scores, ridge, split_frames,
                                           tune_band_threshold)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
})


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
# One configuration, scored on the validation trees
# --------------------------------------------------------------------------
def run_config(X_tr, X_va, V, widths, K, M, seed, tune_idx, train_masks, val_masks):
    """Fit ridge on training trees, tune the band threshold on training trees
    (out-of-fold), and score the band on validation trees."""
    model = shapes.ShapeModel().fit(V)
    Y = model.encode(V, K)
    folds = KFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(ridge(), X_tr, Y, cv=folds)
    estimator = ridge().fit(X_tr, Y)
    pred = estimator.predict(X_va)
    cov = np.atleast_2d(np.cov((Y - oof).T))
    tune_masks = [train_masks[i] for i in tune_idx]
    threshold = tune_band_threshold(oof[tune_idx], cov, model, widths, tune_masks,
                                    max(M // 2, 10), seed)
    masks = band_masks(pred, cov, model, widths, threshold, M, seed)
    scores = per_tree_scores(masks, val_masks, tolerances=(5,))
    return {"dice": scores["dice"].to_numpy(), "cl_f1": scores["cl_f1_t5"].to_numpy(),
            "threshold": threshold, "weighted_r2": shapes.weighted_r2(Y, oof),
            "alpha": float(estimator.named_steps["model"].alpha_), "pred": pred, "cov": cov,
            "model": model, "estimator": estimator}


def summarise(name, value, result, seed):
    lo, hi = evaluate.bootstrap_ci(result["dice"], seed=seed)
    return {"setting": name, "value": value, "dice": result["dice"].mean(), "dice_lo": lo,
            "dice_hi": hi, "cl_f1": result["cl_f1"].mean(), "weighted_r2": result["weighted_r2"],
            "alpha": result["alpha"], "threshold": result["threshold"]}


def simplest_within(rows: pd.DataFrame, tolerance: float, order) -> object:
    """Simplest value (first in ``order``) whose Dice is within ``tolerance`` of the best."""
    best = rows["dice"].max()
    for value in order:
        if rows.loc[rows["value"] == value, "dice"].iloc[0] >= best - tolerance:
            return value
    return order[-1]


def coverage_loop(vectors, widths):
    """Original implementation: draw every sampled shape separately, then average."""
    return np.mean([shapes.draw(v, widths) for v in vectors], axis=0)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="results/shape_settings")
    parser.add_argument("--samples", type=int, default=200, help="base Monte Carlo shapes M")
    parser.add_argument("--pixels-per-tree", type=int, default=500)
    parser.add_argument("--tune-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed, M0, K0, GRID0, ROWS0 = args.seed, args.samples, 8, 16, 32
    rng = np.random.default_rng(seed)
    timing = {}

    splits = split_frames(args.folders, seed)
    train, val = splits["train"], splits["val"]
    log(f"Trees: {len(train)} train, {len(val)} validation (test trees not used)")

    start = time.perf_counter()
    train_masks, coarse_tr, Xpix, ypix = first_pass(train, rng, args.pixels_per_tree)
    val_masks, coarse_va, _, _ = first_pass(val, rng, 10)
    V, widths = describe_masks(train_masks, ROWS0)
    tune_idx = np.sort(rng.choice(len(train), min(args.tune_trees, len(train)), replace=False))
    log(f"Loaded {len(train) + len(val)} trees in {time.perf_counter() - start:.0f} s")

    # baseline on the same validation trees, threshold tuned on the same training trees
    prior = PriorMask().fit(train_masks)
    t_prior = evaluate.best_threshold([prior.predict_proba()] * len(tune_idx),
                                      [train_masks[i] for i in tune_idx])[0]
    baseline = per_tree_scores([evaluate.clean_mask(prior.predict_proba() >= t_prior)] * len(val),
                               val_masks, tolerances=(5,))
    baseline_dice = float(baseline["dice"].mean())
    log(f"Baseline (average mask) validation Dice {baseline_dice:.3f}")
    settings = []

    def run(X_tr, X_va, V_=V, widths_=widths, K=K0, M=M0, seed_=seed, tune=tune_idx,
            masks_tr=train_masks):
        return run_config(X_tr, X_va, V_, widths_, K, M, seed_, tune, masks_tr, val_masks)

    # ---------------- 1. bark map: keep or drop (pre-stated paired rule)
    log("\n1. BARK MAP DECISION")
    start = time.perf_counter()
    folds = KFold(5, shuffle=True, random_state=seed)
    maps_tr = out_of_fold_bark_maps(train, Xpix, ypix, folds, seed, log=log)
    clf = bark_classifier(seed).fit(np.concatenate(Xpix), np.concatenate(ypix))
    maps_va = bark_maps(val, range(len(val)), clf)
    timing["bark_maps_total_s"] = time.perf_counter() - start
    bark_tr = np.stack([shapes.map_descriptor(maps_tr[i]) for i in range(len(train))])
    bark_va = np.stack([shapes.map_descriptor(maps_va[i]) for i in range(len(val))])
    without = run(coarse_tr, coarse_va)
    with_bark = run(np.hstack([coarse_tr, bark_tr]), np.hstack([coarse_va, bark_va]))
    gain = with_bark["dice"] - without["dice"]
    g_lo, g_hi = evaluate.bootstrap_ci(gain, seed=seed)
    use_bark = g_lo > 0
    settings += [summarise("bark map", "no", without, seed), summarise("bark map", "yes", with_bark, seed)]
    log(f"   without bark map: Dice {without['dice'].mean():.3f}   with: {with_bark['dice'].mean():.3f}")
    log(f"   paired gain {gain.mean():+.4f} [{g_lo:+.4f} to {g_hi:+.4f}], better on "
        f"{int((gain > 0).sum())}/{len(gain)} trees -> {'KEEP' if use_bark else 'DROP'} the bark map "
        f"(it cost {timing['bark_maps_total_s']:.0f} s to make)")
    X_tr0 = np.hstack([coarse_tr, bark_tr]) if use_bark else coarse_tr
    X_va0 = np.hstack([coarse_va, bark_va]) if use_bark else coarse_va
    base = with_bark if use_bark else without

    def inputs_with_grid(grid):
        if grid == GRID0:
            return coarse_tr, coarse_va
        return coarse_descriptors(train, grid), coarse_descriptors(val, grid)

    def stack(c_tr, c_va):
        return (np.hstack([c_tr, bark_tr]), np.hstack([c_va, bark_va])) if use_bark else (c_tr, c_va)

    # ---------------- 2. number of shape modes K
    log("\n2. NUMBER OF SHAPE MODES K")
    k_rows = []
    for K in (2, 3, 5, 8, 12, 16):
        r = base if K == K0 else run(X_tr0, X_va0, K=K)
        k_rows.append(summarise("modes K", K, r, seed))
        log(f"   K={K:2d}: Dice {r['dice'].mean():.3f}, centreline F1 {r['cl_f1'].mean():.3f}, "
            f"pooled R^2 {r['weighted_r2']:.2f}")
    k_rows = pd.DataFrame(k_rows)
    K_best = simplest_within(k_rows, 0.005, [2, 3, 5, 8, 12, 16])
    log(f"   -> K = {K_best}")
    settings += k_rows.to_dict("records")

    # ---------------- 3. summary grid (input resolution)
    log("\n3. PHOTO SUMMARY GRID (input resolution)")
    g_rows, grid_inputs = [], {}
    for grid in (8, 16, 32):
        start = time.perf_counter()
        grid_inputs[grid] = stack(*inputs_with_grid(grid))
        r = run(*grid_inputs[grid], K=K_best)
        g_rows.append({**summarise("summary grid", grid, r, seed),
                       "n_inputs": grid_inputs[grid][0].shape[1]})
        log(f"   {grid}x{grid}: {grid_inputs[grid][0].shape[1]} inputs, Dice {r['dice'].mean():.3f} "
            f"({time.perf_counter() - start:.0f} s)")
    g_rows = pd.DataFrame(g_rows)
    GRID_best = simplest_within(g_rows, 0.005, [8, 16, 32])
    X_tr1, X_va1 = grid_inputs[GRID_best]
    log(f"   -> grid {GRID_best}x{GRID_best}")
    settings += g_rows.to_dict("records")

    # ---------------- 4. centreline rows
    log("\n4. CENTRELINE ROWS (shape resolution)")
    r_rows, described = [], {}
    for rows in (16, 32, 64):
        described[rows] = describe_masks(train_masks, rows)
        V_r, w_r = described[rows]
        val_vectors = [shapes.describe(m, rows)[0] for m in val_masks]
        rep = np.mean([evaluate.dice(shapes.draw(v, w_r), m) for v, m in zip(val_vectors, val_masks)])
        r = run(X_tr1, X_va1, V_=V_r, widths_=w_r, K=K_best)
        r_rows.append({**summarise("centreline rows", rows, r, seed), "representation_dice": rep})
        log(f"   {rows} rows: representation Dice {rep:.3f}, method Dice {r['dice'].mean():.3f}")
    r_rows = pd.DataFrame(r_rows)
    ROWS_best = simplest_within(r_rows, 0.005, [16, 32, 64])
    V1, widths1 = described[ROWS_best]
    log(f"   -> {ROWS_best} rows")
    settings += r_rows.drop(columns="representation_dice").to_dict("records")
    chosen = run(X_tr1, X_va1, V_=V1, widths_=widths1, K=K_best)

    # ---------------- 5. Monte Carlo convergence
    log("\n5. MONTE CARLO CONVERGENCE")
    n_check = min(20, len(val))
    references = [shapes.probability_map(chosen["pred"][i], chosen["cov"], chosen["model"], widths1,
                                         6400, seed + 10_000 + i) for i in range(n_check)]
    mc = []
    for M in (10, 25, 50, 100, 200, 400, 800, 1600):
        errors, seconds = [], []
        for repeat in range(3):
            for i in range(n_check):
                t0 = time.perf_counter()
                p = shapes.probability_map(chosen["pred"][i], chosen["cov"], chosen["model"],
                                           widths1, M, seed + 100 * repeat + i)
                seconds.append(time.perf_counter() - t0)
                errors.append(np.sqrt(np.mean((p - references[i]) ** 2)))
        mc.append({"M": M, "rms_error": float(np.mean(errors)), "ms_per_band": 1000 * np.mean(seconds)})
    mc = pd.DataFrame(mc)
    slope = float(np.polyfit(np.log(mc.M), np.log(mc.rms_error), 1)[0])
    log("   RMS difference from a 6400-shape reference: " +
        ", ".join(f"M={r.M}: {r.rms_error:.4f}" for r in mc.itertuples()))
    log(f"   observed order (log-log slope) {slope:.2f}; theory -0.50")
    m_rows = []
    for M in (25, 50, 100, 200, 400, 800):
        masks = band_masks(chosen["pred"], chosen["cov"], chosen["model"], widths1,
                           chosen["threshold"], M, seed)
        m_rows.append({"M": M, "dice": per_tree_scores(masks, val_masks, (5,))["dice"].mean()})
    m_rows = pd.DataFrame(m_rows).merge(mc[["M", "rms_error"]], on="M")
    ok = m_rows[(m_rows.rms_error <= 0.01) & (m_rows.dice >= m_rows.dice.iloc[-1] - 0.002)]
    M_best = int(ok.M.iloc[0]) if len(ok) else int(m_rows.M.iloc[-1])
    log("   validation Dice: " + ", ".join(f"M={r.M}: {r.dice:.4f}" for r in m_rows.itertuples())
        + f" -> M = {M_best}")
    mc = mc.merge(m_rows[["M", "dice"]], on="M", how="left")
    mc.to_csv(out / "mc_convergence.csv", index=False)

    # ---------------- 6. learning curve (data requirements)
    log("\n6. LEARNING CURVE (how many training trees are needed)")
    curve = []
    for n in [n for n in (25, 50, 100, 200) if n < len(train)] + [len(train)]:
        for repeat in range(3 if n < len(train) else 1):
            sub = np.sort(np.random.default_rng(seed + repeat).choice(len(train), n, replace=False))
            sub_masks = [train_masks[i] for i in sub]
            V_s, w_s = describe_masks(sub_masks, ROWS_best)
            tune = np.arange(n) if n <= args.tune_trees else np.sort(
                np.random.default_rng(seed).choice(n, args.tune_trees, replace=False))
            r = run_config(X_tr1[sub], X_va1, V_s, w_s, min(K_best, n - 2), M_best, seed, tune,
                           sub_masks, val_masks)
            curve.append({"n_train": n, "repeat": repeat, "dice": r["dice"].mean(),
                          "weighted_r2": r["weighted_r2"]})
        d = pd.DataFrame(curve)
        d = d[d.n_train == n]
        log(f"   {n:3d} trees: Dice {d.dice.mean():.3f} +/- {d.dice.std(ddof=0):.3f}")
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "learning_curve.csv", index=False)

    # ---------------- 7. repeatability over seeds
    log("\n7. REPEATABILITY (5 seeds: CV folds, Monte Carlo draws, tuning trees)")
    seeds = []
    for s in range(5):
        tune = np.sort(np.random.default_rng(seed + s).choice(len(train), len(tune_idx), replace=False))
        r = run_config(X_tr1, X_va1, V1, widths1, K_best, M_best, seed + s, tune, train_masks, val_masks)
        seeds.append({"seed": seed + s, "dice": r["dice"].mean(), "cl_f1": r["cl_f1"].mean()})
    seeds = pd.DataFrame(seeds)
    seeds.to_csv(out / "seeds.csv", index=False)
    log(f"   Dice {seeds.dice.mean():.3f} +/- {seeds.dice.std(ddof=1):.4f} (std over seeds); "
        f"range {seeds.dice.min():.3f}-{seeds.dice.max():.3f}")

    # ---------------- 8. speed: original vs vectorised, and time per step
    log("\n8. SPEED")
    final = run_config(X_tr1, X_va1, V1, widths1, K_best, M_best, seed, tune_idx, train_masks, val_masks)
    draws = np.random.default_rng(seed).multivariate_normal(final["pred"][0], final["cov"], size=M_best)
    vectors = final["model"].decode(draws)
    t0 = time.perf_counter(); slow = coverage_loop(vectors, widths1); t_loop = time.perf_counter() - t0
    t0 = time.perf_counter(); fast = shapes.coverage(vectors, widths1); t_vec = time.perf_counter() - t0
    log(f"   band drawing, {M_best} shapes: loop {1000 * t_loop:.1f} ms, vectorised {1000 * t_vec:.1f} ms "
        f"({t_loop / t_vec:.0f}x faster); identical result: {np.array_equal(slow, fast)}")
    steps = {"load + resize photo": [], "photo summary": [], "ridge prediction": [],
             "Monte Carlo band": [], "threshold + clean": []}
    if use_bark:
        steps["bark map (pixel classifier)"] = []
    for i in range(min(20, len(val))):
        frame = val.iloc[i]
        # each step is timed on its own, so no step is counted twice
        t = time.perf_counter()
        rgb, depth, _ = load_frame(frame, SIZE)
        steps["load + resize photo"].append(time.perf_counter() - t)
        t = time.perf_counter()
        x = shapes.image_descriptor(rgb, depth, grid=GRID_best)
        steps["photo summary"].append(time.perf_counter() - t)
        if use_bark:
            t = time.perf_counter()
            prob = clf.predict_proba(pixel_features(rgb, depth)[:, IMAGE_ONLY])[:, 1]
            x = np.concatenate([x, shapes.map_descriptor(prob.reshape(SIZE, SIZE))])
            steps["bark map (pixel classifier)"].append(time.perf_counter() - t)
        t = time.perf_counter()
        c = final["estimator"].predict(x[None])[0]
        steps["ridge prediction"].append(time.perf_counter() - t)
        t = time.perf_counter()
        p = shapes.probability_map(c, final["cov"], final["model"], widths1, M_best, seed)
        steps["Monte Carlo band"].append(time.perf_counter() - t)
        t = time.perf_counter()
        evaluate.clean_mask(p >= final["threshold"])
        steps["threshold + clean"].append(time.perf_counter() - t)
    timing_rows = pd.DataFrame([{"step": k, "ms_per_image": 1000 * np.mean(v)} for k, v in steps.items()])
    timing_rows.to_csv(out / "timing.csv", index=False)
    for r in timing_rows.itertuples():
        log(f"   {r.step:30s} {r.ms_per_image:8.2f} ms")
    total_ms = timing_rows.ms_per_image.sum()
    log(f"   {'total per image':30s} {total_ms:8.2f} ms")

    # ---------------- 9. FLOPs, arithmetic intensity, memory (estimates)
    log("\n9. FLOPS, ARITHMETIC INTENSITY AND MEMORY (estimates; see docstring of this script)")
    d = X_tr1.shape[1]
    ridge_flops = 2 * d * K_best + 2 * d                       # scale + matrix-vector product
    ridge_bytes = 8 * (d * K_best + 3 * d)                     # weights, means, scales, input
    log(f"   ridge prediction: {ridge_flops:,} FLOPs, ~{ridge_bytes / 1e3:.0f} kB read -> "
        f"arithmetic intensity {ridge_flops / ridge_bytes:.2f} FLOP/byte (memory-bound)")
    H, Wd = SIZE, SIZE
    band_flops = M_best * H * (2 * 12 + 8) + H * Wd           # PCHIP evaluations + interval maths + prefix sum
    band_bytes = M_best * H * 4 * 8 + H * Wd * 8               # scattered counter updates + prefix sum
    log(f"   Monte Carlo band: ~{band_flops / 1e6:.1f} MFLOP, ~{band_bytes / 1e6:.1f} MB moved -> "
        f"~{band_flops / band_bytes:.2f} FLOP/byte (memory-bound; scattered writes dominate)")
    mlp_flops = 2 * (13 * 64 + 64 * 32 + 32 * 1) * SIZE * SIZE
    stage3 = Path("results/scaffold/summary.csv")
    if stage3.exists():
        s3 = pd.read_csv(stage3).set_index("method")
        mlp_ms = float(s3.loc["mlp", "predict_ms_per_image"])
        rf_ms = float(s3.loc["random_forest", "predict_ms_per_image"])
        log(f"   Stage 3 MLP: ~{mlp_flops / 1e9:.2f} GFLOP per image in {mlp_ms:.0f} ms -> "
            f"~{mlp_flops / (mlp_ms / 1000) / 1e9:.1f} GFLOP/s achieved")
        log(f"   Stage 3 random forest: tree look-ups, almost no FLOPs; {rf_ms:.0f} ms per image")
        log(f"   shape method end to end: {total_ms:.0f} ms per image")
    sizes = {"ridge coefficients": final["estimator"].named_steps["model"].coef_.nbytes,
             "shape model (mean + K modes)": final["model"].mean_.nbytes
             + final["model"].components_[:K_best].nbytes,
             "bark classifier (pickled)": len(pickle.dumps(clf))}
    for k, v in sizes.items():
        log(f"   memory, {k:30s} {v / 1e3:10.1f} kB")

    # ---------------- save
    pd.DataFrame(settings).to_csv(out / "settings.csv", index=False)
    config = {"use_bark_map": bool(use_bark), "bark_gain": float(gain.mean()),
              "bark_gain_ci": [g_lo, g_hi], "modes_K": int(K_best), "summary_grid": int(GRID_best),
              "centreline_rows": int(ROWS_best), "monte_carlo_M": int(M_best),
              "pixels_per_tree": args.pixels_per_tree, "tune_trees": args.tune_trees, "seed": seed,
              "validation_dice": float(final["dice"].mean()),
              "validation_cl_f1": float(final["cl_f1"].mean()),
              "baseline_validation_dice": baseline_dice, "ms_per_image": float(total_ms),
              "rules": {"bark map": "keep only if the paired 95% CI of the Dice gain is above 0",
                        "K, grid, rows": "simplest value within 0.005 Dice of the best",
                        "M": "smallest M with band RMS error <= 0.01 and Dice within 0.002 "
                             "of the largest M"}}
    (out / "final_config.json").write_text(json.dumps(config, indent=1))
    paired = final["dice"] - baseline["dice"].to_numpy()
    lo, hi = evaluate.bootstrap_ci(paired, seed=seed)
    log(f"\nCHOSEN: bark map {'yes' if use_bark else 'no'}, K={K_best}, grid {GRID_best}x{GRID_best}, "
        f"{ROWS_best} rows, M={M_best}")
    log(f"Validation Dice {final['dice'].mean():.3f} vs baseline {baseline_dice:.3f}: paired "
        f"{paired.mean():+.3f} [{lo:+.3f} to {hi:+.3f}], better on {int((paired > 0).sum())}/{len(paired)} trees")

    figures(pd.DataFrame(settings), k_rows, g_rows, r_rows, mc, slope, curve, baseline_dice,
            timing_rows, t_loop, t_vec, M_best, K_best, GRID_best, ROWS_best, out)
    log(f"\nWrote results to {out}")
    return 0


# --------------------------------------------------------------------------
def figures(settings, k_rows, g_rows, r_rows, mc, slope, curve, baseline_dice, timing_rows,
            t_loop, t_vec, M_best, K_best, GRID_best, ROWS_best, out):
    # sensitivity to the three settings
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, rows, label, chosen in zip(axes, [k_rows, g_rows, r_rows],
                                       ["shape modes K", "photo summary grid", "centreline rows"],
                                       [K_best, GRID_best, ROWS_best]):
        x = rows["value"].astype(float)
        ax.errorbar(x, rows["dice"], yerr=[rows["dice"] - rows["dice_lo"], rows["dice_hi"] - rows["dice"]],
                    fmt="o-", color=PALETTE[0], capsize=3, markersize=5)
        ax.axhline(baseline_dice, color=GREY, linestyle="--", linewidth=1)
        pick = rows[rows["value"] == chosen].iloc[0]
        ax.scatter([float(chosen)], [pick["dice"]], s=90, facecolors="none", edgecolors=PALETTE[1],
                   linewidths=1.5, zorder=4)
        ax.set(xlabel=label, xticks=list(x))
    axes[0].set_ylabel("validation Dice (95% CI)")
    axes[0].annotate("baseline", (axes[0].get_xlim()[0], baseline_dice), xytext=(4, 3),
                     textcoords="offset points", color=GREY, fontsize=7)
    fig.suptitle("Sensitivity to settings (orange ring: chosen by the simplicity rule)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_settings.png")
    plt.close(fig)

    # Monte Carlo convergence
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    ax.loglog(mc.M, mc.rms_error, "o-", color=PALETTE[0], label=f"measured (slope {slope:.2f})")
    ref = mc.rms_error.iloc[0] * (mc.M / mc.M.iloc[0]) ** -0.5
    ax.loglog(mc.M, ref, "--", color=GREY, label="theory: slope -0.5")
    ax.set(xlabel="Monte Carlo shapes M (log scale)", ylabel="RMS error of the band (log scale)",
           title="Band error falls like $M^{-1/2}$")
    ax.legend(fontsize=7)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax = axes[1]
    ax.plot(mc.M, mc.ms_per_band, "o-", color=PALETTE[0])
    ax.axvline(M_best, color=PALETTE[1], linestyle="--", linewidth=1)
    ax.annotate(f"chosen M = {M_best}", (M_best, mc.ms_per_band.max()), xytext=(4, -10),
                textcoords="offset points", color=PALETTE[1], fontsize=7)
    ax.set(xlabel="Monte Carlo shapes M", ylabel="time per band (ms)", title="Cost grows roughly linearly with M")
    fig.tight_layout()
    fig.savefig(out / "fig_mc_convergence.png")
    plt.close(fig)

    # learning curve
    agg = curve.groupby("n_train").dice.agg(["mean", "std"]).fillna(0).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.errorbar(agg.n_train, agg["mean"], yerr=agg["std"], fmt="o-", color=PALETTE[0], capsize=3)
    ax.axhline(baseline_dice, color=GREY, linestyle="--", linewidth=1)
    ax.annotate("baseline (average mask)", (agg.n_train.min(), baseline_dice), xytext=(2, 4),
                textcoords="offset points", color=GREY, fontsize=7)
    ax.set(xscale="log", xlabel="training trees (log scale)", ylabel="validation Dice",
           title="Learning curve: how much data is needed")
    ax.set_xticks(agg.n_train)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curve.png")
    plt.close(fig)

    # time per step and the vectorisation speed-up
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    t = timing_rows.iloc[::-1]
    axes[0].barh(t.step, t.ms_per_image, color=PALETTE[0], height=0.6)
    axes[0].set(xlabel="ms per image", title="Where the time goes (one photo)")
    axes[0].grid(axis="y", visible=False)
    axes[1].bar(["loop\n(original)", "vectorised"], [1000 * t_loop, 1000 * t_vec],
                color=[GREY, PALETTE[0]], width=0.6)
    axes[1].set(ylabel="ms per band", title=f"Band drawing, M = {M_best}: {t_loop / t_vec:.0f}x faster")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_timing.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
