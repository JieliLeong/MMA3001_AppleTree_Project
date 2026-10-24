"""Stage 1b: do the supplied masks line up with the photos?

Usage (from the repository root)::

    python scripts/alignment_check.py /content/data --out results/alignment

On some trees the truth outline visibly misses the real trunk. Two explanations
need different responses:

* the masks are **rough sketches**, so the error changes from tree to tree
  (a limitation of the ground truth, to be reported); or
* the masks sit in a **different frame** from the photo (depth camera, crop,
  winter photo), so the truth is offset the same way on every tree (a fixable
  registration bug).

For every labelled tree the script moves its mask (shifts up to +/-24 px on the
256 x 256 grid, then stretches of 0.80-1.25 about the centre) and measures how
much of the moved label the image explains within the same 32 x 32 block (the
uncertainty coefficient, see :mod:`apple_scaffold.alignment`). It does this for
four images of each tree: summer colour, summer depth, winter colour and winter
depth. It reports

* the best shift, then the best stretch and shift, pooled over all trees and
  separately in each of 5 groups of trees (does the answer repeat?);
* how much better the best transform agrees than the truth as drawn;
* a control with each mask paired with the *next* tree's images (what
  position alone gives);
* overlays of the truth on summer and winter photos for 6 random trees.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apple_scaffold import alignment, data  # noqa: E402
from apple_scaffold.features import load_frame  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})
SOURCES = ["summer colour", "summer depth", "winter colour", "winter depth"]
SIZE, BLOCK = 256, 32
SCALES = [round(float(v), 2) for v in np.arange(0.80, 1.251, 0.05)]


def log(message: str) -> None:
    print(message, flush=True)


def load_trees(folders):
    """Labelled trees with complete summer and winter frames."""
    files, _ = data.index_files(folders)
    frames = data.build_frames(files)
    clean = frames[frames["problems"] == ""]
    summer = clean[(clean.season == "Summer") & clean.path_GT.notna()].set_index("tree_id")
    winter = clean[clean.season == "Winter"].set_index("tree_id")
    trees = sorted(set(summer.index) & set(winter.index))
    return summer.loc[trees], winter.loc[trees], trees


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="results/alignment")
    parser.add_argument("--max-shift", type=int, default=24)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summer, winter, trees = load_trees(args.folders)
    n = len(trees)
    log(f"{n} labelled trees with clean summer and winter frames")
    if n == 0:
        raise SystemExit("No labelled trees with both summer and winter frames were found.")
    masks = np.zeros((n, SIZE, SIZE), bool)
    codes = {name: np.zeros((n, SIZE, SIZE), np.uint8) for name in SOURCES}
    for i, tree in enumerate(trees):
        rgb, depth, gt = load_frame(summer.loc[tree], SIZE)
        masks[i] = gt
        codes["summer colour"][i] = alignment.colour_codes(rgb)
        codes["summer depth"][i] = alignment.depth_codes(depth)
        rgb, depth, _ = load_frame(winter.loc[tree], SIZE)
        codes["winter colour"][i] = alignment.colour_codes(rgb)
        codes["winter depth"][i] = alignment.depth_codes(depth)
        if (i + 1) % 100 == 0 or i + 1 == n:
            log(f"  loaded {i + 1}/{n}")
    n_codes = {"summer colour": 64, "winter colour": 64, "summer depth": 16, "winter depth": 16}

    span = range(-args.max_shift, args.max_shift + 1, args.step)
    shifts = [(dx, dy) for dy in span for dx in span]
    groups = np.random.default_rng(args.seed).permutation(n) % args.folds
    zero = shifts.index((0, 0))
    mismatched = np.roll(masks, 1, axis=0)

    rows, scale_rows, summary, maps, scale_maps = [], [], [], {}, {}
    for name in SOURCES:
        log(f"\n{name}:")
        scores = alignment.shift_scores(codes[name], masks, shifts, n_codes[name],
                                        args.max_shift, BLOCK, groups)
        as_drawn = scores["pooled"][zero]
        shift_only = scores["pooled"].max()
        shift = alignment.best_shift(scores["pooled"], shifts)
        log(f"  best shift alone (right, down): {shift} px")

        # alternate: best stretch at the current shift, then best shift at that stretch
        # (whole grids in round 1, then neighbourhoods of the current answer)
        scale = (1.0, 1.0)
        for round_ in range(1, 9):
            if round_ == 1:
                scales = [(sx, sy) for sy in SCALES for sx in SCALES]
            else:
                near = [v for v in SCALES if abs(v - scale[0]) <= 0.1 + 1e-9]
                near_y = [v for v in SCALES if abs(v - scale[1]) <= 0.1 + 1e-9]
                scales = [(sx, sy) for sy in near_y for sx in near]
            counts = alignment.warp_counts(codes[name], masks, [(*v, *shift) for v in scales],
                                           n_codes[name], args.max_shift, BLOCK)
            new_scale = scales[int(np.argmax(alignment.uncertainty_coefficient(counts[0])))]
            local = [(dx, dy) for dy in range(shift[1] - 6, shift[1] + 7, args.step)
                     for dx in range(shift[0] - 6, shift[0] + 7, args.step)
                     if max(abs(dx), abs(dy)) <= args.max_shift]
            local_scores = alignment.shift_scores(codes[name], alignment.warp_masks(masks, *new_scale),
                                                  local, n_codes[name], args.max_shift, BLOCK)
            new_shift = alignment.best_shift(local_scores["pooled"], local)
            log(f"  round {round_}: stretch {new_scale}, shift {new_shift}, "
                f"{100 * local_scores['pooled'].max():.2f} % explained")
            if new_scale == scale and new_shift == shift:
                break
            scale, shift = new_scale, new_shift

        # whole grids around the answer: for the figures and the five groups of trees
        grid = [(sx, sy, shift[0], shift[1]) for sy in SCALES for sx in SCALES]
        counts = alignment.warp_counts(codes[name], masks, grid, n_codes[name], args.max_shift,
                                       BLOCK, groups)
        on_grid = alignment.uncertainty_coefficient(counts.sum(axis=0))
        scale_groups = alignment.uncertainty_coefficient(counts)
        scores = alignment.shift_scores(codes[name], alignment.warp_masks(masks, *scale), shifts,
                                        n_codes[name], args.max_shift, BLOCK, groups)
        best = scores["pooled"].max()
        maps[name], scale_maps[name] = scores["pooled"], on_grid
        group_scale = [grid[int(np.argmax(g))][:2] for g in scale_groups]
        group_shift = [alignment.best_shift(g, shifts) for g in scores["groups"]]
        agree = (all(abs(a - scale[0]) <= 0.05 + 1e-9 and abs(b - scale[1]) <= 0.05 + 1e-9
                     for a, b in group_scale)
                 and all(abs(a - shift[0]) <= 2 and abs(b - shift[1]) <= 2 for a, b in group_shift))
        for k, (dx, dy) in enumerate(shifts):
            rows.append({"source": name, "sx": scale[0], "sy": scale[1], "dx": dx, "dy": dy,
                         "explained": scores["pooled"][k],
                         **{f"explained_group{g}": scores["groups"][g, k]
                            for g in range(len(scores["groups"]))}})
        for k, (a, b, c, d) in enumerate(grid):
            scale_rows.append({"source": name, "sx": a, "sy": b, "dx": c, "dy": d,
                               "explained": on_grid[k]})
        control = alignment.warp_counts(codes[name], mismatched, [(1, 1, 0, 0), (*scale, *shift)],
                                        n_codes[name], args.max_shift, BLOCK)
        control = alignment.uncertainty_coefficient(control[0])
        record = {
            "source": name, "scale_x": scale[0], "scale_y": scale[1], "dx": shift[0], "dy": shift[1],
            "groups_agree": agree,
            "group_scales": "; ".join(f"({a:.2f},{b:.2f})" for a, b in group_scale),
            "group_shifts": "; ".join(f"({a},{b})" for a, b in group_shift),
            "explained_as_drawn": as_drawn, "explained_best_shift_alone": shift_only,
            "explained_best": best, "gain_pct": 100 * (best / as_drawn - 1),
            "control_as_drawn": control[0], "control_best": control[1],
        }
        summary.append(record)
        log(f"  final: stretch {scale}, shift {shift}; groups agree: {agree}")
        log(f"    group stretches {group_scale}")
        log(f"    group shifts    {group_shift}")
        log(f"  share of the truth explained by the image: as drawn {100 * as_drawn:.2f} %, "
            f"best shift alone {100 * shift_only:.2f} %, best stretch + shift {100 * best:.2f} % "
            f"(relative gain {record['gain_pct']:+.0f} %); trees mismatched on purpose "
            f"{100 * control[0]:.2f} %")

    pd.DataFrame(rows).to_csv(out / "shift_scan.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out / "scale_scan.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "trees": n, "max_shift": args.max_shift, "step": args.step, "block": BLOCK,
        "scales": SCALES, "folds": args.folds, "seed": args.seed,
        "convention": "the truth is scaled by (sx, sy) about the image centre, then moved dx px "
                      "right and dy px down (256 grid)"},
        indent=2))

    # heat maps of information against scale
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.4))
    fig.subplots_adjust(hspace=0.55, wspace=0.4)
    edges = np.array(SCALES)
    half = (edges[1] - edges[0]) / 2
    for ax, name in zip(axes.ravel(), SOURCES):
        grid = scale_maps[name].reshape(len(SCALES), len(SCALES))   # rows sy, columns sx
        low, high = grid.min(), grid.max()
        image = ax.imshow((grid - low) / (high - low), cmap="Blues", vmin=0, vmax=1, origin="lower",
                          extent=(edges[0] - half, edges[-1] + half, edges[0] - half, edges[-1] + half))
        r = table.set_index("source").loc[name]
        ax.plot(1, 1, marker="+", color=INK, markersize=12, mew=1.5)
        best_sx, best_sy = SCALES[int(np.argmax(scale_maps[name])) % len(SCALES)], \
            SCALES[int(np.argmax(scale_maps[name])) // len(SCALES)]
        ax.plot(best_sx, best_sy, marker="o", mfc="none", mec=PALETTE[1], markersize=11, mew=2)
        ax.set_title(f"{name}\nbest ({best_sx:.2f}, {best_sy:.2f}) at shift "
                     f"({int(r.dx)}, {int(r.dy)})\n"
                     f"range {100 * low:.2f}-{100 * high:.2f} % explained", fontsize=8)
        ax.set_xlabel("stretch truth sideways (x scale)")
        ax.set_ylabel("stretch truth up-down (y scale)")
    fig.colorbar(image, ax=axes, shrink=0.6, label="agreement (0 = panel minimum, 1 = maximum)")
    fig.suptitle("Does the truth need stretching to agree with each image? (+ = as drawn, o = best)",
                 fontsize=9)
    fig.savefig(out / "fig_scale_scan.png", bbox_inches="tight")
    plt.close(fig)

    # heat maps of information against shift
    span = np.array(list(span))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.4))
    fig.subplots_adjust(hspace=0.55, wspace=0.4)
    for ax, name in zip(axes.ravel(), SOURCES):
        grid = maps[name].reshape(len(span), len(span))           # rows dy, columns dx
        low, high = grid.min(), grid.max()
        image = ax.imshow((grid - low) / (high - low), cmap="Blues", vmin=0, vmax=1, origin="upper",
                          extent=(span[0] - args.step / 2, span[-1] + args.step / 2,
                                  span[-1] + args.step / 2, span[0] - args.step / 2))
        r = table.set_index("source").loc[name]
        ax.plot(0, 0, marker="+", color=INK, markersize=12, mew=1.5)
        ax.plot(r.dx, r.dy, marker="o", mfc="none", mec=PALETTE[1], markersize=11, mew=2)
        ax.set_title(f"{name}\nbest ({int(r.dx)}, {int(r.dy)}) at stretch "
                     f"({r.scale_x:.2f}, {r.scale_y:.2f})\n"
                     f"range {100 * low:.2f}-{100 * high:.2f} % explained", fontsize=8)
        ax.set_xlabel("move truth right (px)")
        ax.set_ylabel("move truth down (px)")
    fig.colorbar(image, ax=axes, shrink=0.6, label="agreement (0 = panel minimum, 1 = maximum)")
    fig.suptitle("Where does the truth agree best with each image? (+ = no shift, o = best)",
                 fontsize=9)
    fig.savefig(out / "fig_shift_scan.png", bbox_inches="tight")
    plt.close(fig)

    # overlays on full-size photos
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(n, size=min(6, n), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.4))
    for k, i in enumerate(picks):
        tree = trees[i]
        gt = data.load_gt(summer.loc[tree].path_GT, size=(640, 480))
        outline = gt ^ ndimage.binary_erosion(gt, iterations=3)
        for j, (season, frame) in enumerate((("summer", summer.loc[tree]), ("winter", winter.loc[tree]))):
            photo = np.asarray(Image.open(frame.path_RGB).convert("RGB").resize((640, 480))).copy()
            photo[outline] = (235, 104, 52)
            ax = axes[k // 2, 2 * (k % 2) + j]
            ax.imshow(photo)
            ax.set_title(f"{tree} {season}", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Truth outline (orange) on the summer and winter photos of 6 random trees", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_truth_overlays.png", dpi=110)
    plt.close(fig)

    log("\nSummary: how the truth must be stretched (scale) and moved (px, right / down) "
        "to agree best with each image")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        log(table[["source", "scale_x", "scale_y", "dx", "dy", "explained_as_drawn",
                   "explained_best", "gain_pct", "groups_agree"]].round(4).to_string(index=False))
    log(f"\nWrote CSVs and figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
