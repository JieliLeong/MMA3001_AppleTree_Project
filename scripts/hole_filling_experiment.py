"""Depth-hole filling experiment: accuracy, cost and convergence of each method.

Usage (from the repository root)::

    python scripts/hole_filling_experiment.py /content/data --out results/holes --frames 20

Validation design: disc-shaped holes of known radius are punched into depth
that *is* known, every method fills them, and the filled values are compared
with the true ones (errors in 8-bit depth levels). Only trees from the
training split are used, so later stages can still be tested on unseen trees.

Writes to ``--out``:

* ``real_holes.csv``   - size of every real hole in the sampled frames
* ``accuracy.csv``     - one row per (frame, radius, method): RMSE, MAE, max
  error, run time, iterations and final residual
* ``omega_sweep.csv``  - SOR iterations against the relaxation factor
* ``fig_*.png``        - figures for the report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import data, holes  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical colours in a fixed order (validated colour-blind-safe palette)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5,
    "lines.markersize": 5,
})


# --------------------------------------------------------------------------
def training_frames(folders, seed: int) -> pd.DataFrame:
    """All clean frames (both seasons) of the trees in the training split."""
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    frames = frames[frames["problems"] == ""]
    labelled = frames[frames["path_GT"].notna()]
    split = data.grouped_split(labelled, seed=seed)
    train_trees = set(labelled.loc[split == "train", "tree_id"])
    return frames[frames["tree_id"].isin(train_trees)].reset_index(drop=True)


def score(filled: np.ndarray, truth: np.ndarray) -> dict:
    """Error statistics over the punched pixels, in depth levels."""
    err = filled - truth
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "max_abs": float(np.max(np.abs(err)))}


def run_accuracy(frames, radii, per_frame, tol, rng):
    """Punch holes, fill with every method, record error and cost."""
    rows, hole_rows, example = [], [], None
    for f, frame in frames.iterrows():
        depth = data.load_depth(frame.path_D).astype(float)
        for area in np.bincount(holes.ndimage.label(holes.find_holes(depth))[0].ravel())[1:]:
            hole_rows.append({"tree_id": frame.tree_id, "season": frame.season, "area": int(area)})
        valid = depth > 0
        for radius in radii:
            mask = holes.punch_discs(valid, radius, per_frame, rng, margin=4)
            if not mask.any():
                continue
            corrupted = depth.copy()
            corrupted[mask] = 0.0
            for method in holes.METHODS:
                options = {"tol": tol} if method in ("jacobi", "gauss_seidel", "sor") else {}
                start = time.perf_counter()
                result = holes.fill_holes(corrupted, mask, method, **options)
                seconds = time.perf_counter() - start
                rows.append({"tree_id": frame.tree_id, "season": frame.season, "radius": radius,
                             "method": method, "n_pixels": int(mask.sum()),
                             "time_ms": 1000 * seconds, "iterations": result.iterations,
                             "residual": result.residual, "converged": result.converged,
                             **score(result.depth[mask], depth[mask])})
            if example is None and radius == max(radii):
                example = (frame, depth, mask)
        print(f"  frame {f + 1}/{len(frames)} done ({frame.tree_id} {frame.season})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(hole_rows), example


def run_omega_sweep(frames, radius, rng, tol, n_frames=3):
    """SOR iterations as a function of the relaxation factor (sensitivity)."""
    rows = []
    for _, frame in frames.head(n_frames).iterrows():
        depth = data.load_depth(frame.path_D).astype(float)
        mask = holes.punch_discs(depth > 0, radius, 1, rng, margin=4)
        if not mask.any():
            continue
        corrupted = depth.copy()
        corrupted[mask] = 0.0
        for omega in np.round(np.arange(1.0, 1.96, 0.05), 2):
            result = holes.solve_sor(corrupted, mask, omega=omega, tol=tol)
            rows.append({"tree_id": frame.tree_id, "omega": omega,
                         "iterations": result.iterations,
                         "omega_estimate": holes.optimal_omega(mask)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def slope(x, y) -> float:
    """Least-squares slope of log(y) against log(x) (observed order)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def last_slope(x, y) -> float:
    """Log-log slope between the two largest x values (closest to asymptotic)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.log(y[-1] / y[-2]) / np.log(x[-1] / x[-2]))


def plain_log_axis(axis) -> None:
    """Label a log axis with plain numbers at 1-2-5 steps and no minor labels."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def figures(acc, real, sweep, example, out: Path) -> None:
    labels = {"nearest": "griddata nearest", "linear": "griddata linear",
              "cubic": "griddata cubic", "rbf": "RBF thin-plate", "direct": "Laplace (direct LU)",
              "jacobi": "Laplace Jacobi", "gauss_seidel": "Laplace Gauss-Seidel",
              "sor": "Laplace SOR"}
    colour = dict(zip(holes.METHODS, PALETTE))
    mean = acc.groupby(["method", "radius"]).mean(numeric_only=True).reset_index()

    # accuracy - the four Laplace solvers give the same answer, so show one
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in ("nearest", "linear", "cubic", "rbf", "direct"):
        d = mean[mean.method == method]
        ax.plot(d.radius, d.rmse, "o-", color=colour[method], label=labels[method])
    ax.set(xlabel="hole radius (pixels)", ylabel="RMSE (depth levels)",
           title="Fill accuracy on punched holes", xticks=sorted(acc.radius.unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy.png")
    plt.close(fig)

    # cost
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in holes.METHODS:
        d = mean[mean.method == method]
        ax.plot(d.radius, d.time_ms, "o-", color=colour[method], label=labels[method])
    ax.set(xlabel="hole radius (pixels)", ylabel="time per frame (ms, log scale)",
           yscale="log", title="Run time per frame", xticks=sorted(acc.radius.unique()))
    plain_log_axis(ax.yaxis)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_runtime.png")
    plt.close(fig)

    # convergence - iterations against radius on log-log axes
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in ("jacobi", "gauss_seidel", "sor"):
        d = mean[mean.method == method]
        k = last_slope(d.radius, d.iterations)
        ax.plot(d.radius, d.iterations, "o-", color=colour[method],
                label=f"{labels[method]} (slope {k:.2f} at largest R)")
    ax.set(xscale="log", yscale="log", xlabel="hole radius R (pixels, log scale)",
           ylabel="iterations (log scale)",
           title="Iterations: theory $R^2$ for Jacobi and Gauss-Seidel, $R$ for SOR")
    plain_log_axis(ax.xaxis)
    plain_log_axis(ax.yaxis)
    ax.set_xticks(sorted(acc.radius.unique()))       # label exactly the radii tested
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_iterations.png")
    plt.close(fig)

    # SOR sensitivity to omega
    if len(sweep):
        s = sweep.groupby("omega").iterations.mean()
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(s.index, s.values, "o-", color=PALETTE[0])
        estimate = sweep.omega_estimate.mean()
        ax.axvline(estimate, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"estimated optimum {estimate:.2f}", (estimate, s.max()),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="top", color=MUTED)
        ax.set(xlabel="relaxation factor omega (1 = Gauss-Seidel)", ylabel="mean iterations",
               title="SOR sensitivity to omega")
        fig.tight_layout()
        fig.savefig(out / "fig_omega.png")
        plt.close(fig)

    # real hole sizes
    if len(real):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        bins = np.logspace(0, np.log10(max(real.area.max(), 10)), 30)
        for i, season in enumerate(sorted(real.season.unique())):
            ax.hist(real.area[real.season == season], bins=bins, histtype="step",
                    linewidth=1.5, color=PALETTE[i], label=season)
        ax.set(xscale="log", xlabel="hole area (pixels, log scale)", ylabel="number of holes",
               title="Sizes of real holes (enclosed, no depth return)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_real_holes.png")
        plt.close(fig)

    # worked example: one punched hole filled two ways
    if example is not None:
        frame, depth, mask = example
        labels_, _ = holes.ndimage.label(mask)
        box = holes.ndimage.find_objects(labels_)[0]
        pad = 12
        box = tuple(slice(max(s.start - pad, 0), s.stop + pad) for s in box)
        corrupted = depth.copy()
        corrupted[mask] = 0.0
        laplace = holes.fill_holes(corrupted, mask, "direct").depth
        nearest = holes.fill_holes(corrupted, mask, "nearest").depth
        lo, hi = depth[box][depth[box] > 0].min(), depth[box].max()
        panels = [("true depth", depth), ("hole punched", corrupted),
                  ("Laplace fill", laplace), ("nearest fill", nearest)]
        fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
        for ax, (title, image) in zip(axes, panels):
            ax.imshow(image[box], cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{frame.tree_id} {frame.season}: same hole, two fills", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_example.png")
        plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="dataset folders")
    parser.add_argument("--out", default="results/holes")
    parser.add_argument("--frames", type=int, default=20, help="frames to sample")
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--per-frame", type=int, default=3, help="holes per frame and radius")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="iterative solvers stop when max residual < tol (depth levels)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = training_frames(args.folders, args.seed)
    frames = frames.sample(min(args.frames, len(frames)), random_state=args.seed)
    frames = frames.reset_index(drop=True)
    print(f"Sampled {len(frames)} training frames "
          f"({(frames.season == 'Summer').sum()} summer, {(frames.season == 'Winter').sum()} winter)")

    acc, real, example = run_accuracy(frames, args.radii, args.per_frame, args.tol, rng)
    sweep = run_omega_sweep(frames, max(args.radii), rng, args.tol)
    acc.to_csv(out / "accuracy.csv", index=False)
    real.to_csv(out / "real_holes.csv", index=False)
    sweep.to_csv(out / "omega_sweep.csv", index=False)
    figures(acc, real, sweep, example, out)

    pd.set_option("display.width", 120)
    print("\nReal holes per frame (enclosed, <= 2000 px):")
    if len(real):
        per_frame = real.groupby(["tree_id", "season"]).size()
        print(f"  mean {per_frame.mean():.1f} holes per frame; area percentiles (px): "
              f"50% {real.area.quantile(0.5):.0f}, 90% {real.area.quantile(0.9):.0f}, "
              f"99% {real.area.quantile(0.99):.0f}")
    print("\nRMSE (depth levels), mean over frames:")
    print(acc.pivot_table(index="method", columns="radius", values="rmse").round(2).to_string())
    print("\nMax absolute error (depth levels), worst frame:")
    print(acc.pivot_table(index="method", columns="radius", values="max_abs", aggfunc="max")
          .round(1).to_string())
    print("\nTime per frame (ms), mean:")
    print(acc.pivot_table(index="method", columns="radius", values="time_ms").round(1).to_string())
    its = acc[acc.method.isin(["jacobi", "gauss_seidel", "sor"])]
    print("\nIterations, mean:")
    print(its.pivot_table(index="method", columns="radius", values="iterations").round(0).to_string())
    print("  observed order = log-log slope of iterations against radius "
          "(theory: 2 for Jacobi and Gauss-Seidel, 1 for optimal SOR)")
    for method in ("jacobi", "gauss_seidel", "sor"):
        d = its[its.method == method].groupby("radius").iterations.mean()
        print(f"  {method:13s} fitted over all radii {slope(d.index, d.values):.2f}, "
              f"between the two largest radii {last_slope(d.index, d.values):.2f}")
    print(f"  all iterative runs converged: {bool(its.converged.all())}")
    if len(sweep):
        best = sweep.groupby("omega").iterations.mean().idxmin()
        print(f"\nSOR: fewest iterations at omega = {best:.2f}; "
              f"estimate from hole size = {sweep.omega_estimate.mean():.2f}")
    print(f"\nWrote CSVs and figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
