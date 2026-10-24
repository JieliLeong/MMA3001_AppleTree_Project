"""Every method side by side on the same test trees (best, median and worst).

Usage (from the repository root)::

    python scripts/final_test.py /content/data --out /content/final_rerun --save-masks
    python scripts/method_gallery.py /content/data --final-masks /content/final_rerun/test_masks.npz

The Stage 3 pixel models are retrained exactly as in ``scripts/scaffold_experiment.py``
(same seed, same training pixels, thresholds tuned on the same 30 validation
trees); the shape method and the average-mask baseline come from the masks the
final test saved. Nothing is tuned here and no new scores are reported: the
script checks that every score it shows matches the score saved by Stage 3 or
Stage 4d for that tree, and says so.

Writes ``fig_all_methods.png`` (rows: the shape method's best, median and worst
test tree; columns: photo, depth, truth, then every method's predicted mask
with its Dice and IoU) and ``gallery_scores.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apple_scaffold import data, evaluate  # noqa: E402
from apple_scaffold.baselines import ColourRule, PriorMask  # noqa: E402
from apple_scaffold.features import load_frame, pixel_features, sample_pixels  # noqa: E402
import scaffold_experiment as s3  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORANGE = "#eb6834"
COLUMNS = [("colour_rule", "Colour rule"), ("logistic", "Logistic regression"),
           ("random_forest", "Random forest"), ("mlp", "Neural network (MLP)"),
           ("avg_mask", "Average mask\n(baseline)"), ("shape_final", "Shape method\n(final)")]


def log(message: str) -> None:
    print(message, flush=True)


def stage3_models(train, val, seed, size=256, pixels_per_image=1500, val_images=30):
    """Refit the Stage 3 pixel models and thresholds exactly as Stage 3 did."""
    rng = np.random.default_rng(seed)
    X, y, train_masks = [], [], []
    for frame in train.itertuples():
        rgb, depth, gt = load_frame(frame, size)
        feats = pixel_features(rgb, depth)
        idx = sample_pixels(gt, pixels_per_image, rng)
        X.append(feats[idx])
        y.append(gt.ravel()[idx])
        train_masks.append(gt)
    X, y = np.concatenate(X), np.concatenate(y)
    prior = PriorMask().fit(train_masks)
    models = s3.make_models(seed)
    models.pop("rf_no_position")                       # not shown in the figure
    for name, (model, cols) in models.items():
        model.fit(X[:, cols], y)
        log(f"  refitted {name}")
    val = val.sample(min(val_images, len(val)), random_state=seed)
    items = []
    for frame in val.itertuples():
        rgb, depth, _ = load_frame(frame, size)
        items.append((rgb, depth, pixel_features(rgb, depth),
                      data.load_gt(frame.path_GT, size=(s3.EVAL_SIZE, s3.EVAL_SIZE))))
    rule, _ = ColourRule.tune([it[0] for it in items],
                              [s3.to_eval_grid_mask(it[3], size) for it in items])
    thresholds = {"colour_rule": 0.5}
    for method in ("logistic", "random_forest", "mlp"):
        probs = [s3.to_eval_grid(s3.predict_map(method, models, prior, rule, r, d, f, size))
                 for r, d, f, _ in items]
        thresholds[method], _ = evaluate.best_threshold(probs, [it[3] for it in items])
    return models, prior, rule, thresholds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--final-masks", default="results/final/test_masks.npz")
    parser.add_argument("--stage3", default="results/scaffold/metrics.csv")
    parser.add_argument("--final-metrics", default="results/final/final_metrics.csv")
    parser.add_argument("--out", default="results/gallery")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    saved = np.load(args.final_masks)
    ids = list(saved["tree_id"])
    final_dice = np.array([evaluate.dice(p, t) for p, t in zip(saved["shape_final"], saved["truth"])])
    order = np.argsort(final_dice)
    picks = [("best", order[-1]), ("median", order[len(order) // 2]), ("worst", order[0])]
    log("Trees shown: " + ", ".join(f"{name} {ids[i]} (Dice {final_dice[i]:.2f})" for name, i in picks))

    splits = s3.split_frames(args.folders, args.seed)
    test = splits["test"].set_index("tree_id")
    log(f"Refitting the Stage 3 models on {len(splits['train'])} training trees (a few minutes) ...")
    models, prior, rule, thresholds = stage3_models(splits["train"], splits["val"], args.seed)

    rows, panels = [], []
    for name, i in picks:
        tree = ids[i]
        frame = test.loc[tree]
        rgb, depth, _ = load_frame(frame, 256)
        truth = saved["truth"][i]
        feats = pixel_features(rgb, depth)
        masks = {}
        for method in ("colour_rule", "logistic", "random_forest", "mlp"):
            prob = s3.to_eval_grid(s3.predict_map(method, models, prior, rule, rgb, depth, feats, 256))
            masks[method] = evaluate.clean_mask(prob >= thresholds[method])
        for method in ("avg_mask", "shape_final"):
            masks[method] = saved[method][i]
        for method, mask in masks.items():
            rows.append({"tree_id": tree, "row": name, "method": method,
                         "dice": evaluate.dice(mask, truth), "iou": evaluate.iou(mask, truth)})
        panels.append((name, tree, rgb, depth, truth, masks))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "gallery_scores.csv", index=False)

    # check: every score shown equals the score saved by Stage 3 / Stage 4d for that tree
    checks, missing = [], 0
    for path, methods in ((args.stage3, ["colour_rule", "logistic", "random_forest", "mlp"]),
                          (args.final_metrics, ["avg_mask", "shape_final"])):
        if not Path(path).exists():
            missing += int(scores.method.isin(methods).sum())
            continue
        saved_scores = pd.read_csv(path).set_index(["tree_id", "method"]).dice.to_dict()
        for r in scores[scores.method.isin(methods)].itertuples():
            if (r.tree_id, r.method) in saved_scores:
                checks.append(abs(r.dice - saved_scores[(r.tree_id, r.method)]))
            else:
                missing += 1
    if checks:
        worst = max(checks)
        log(f"Check against the saved Stage 3 / Stage 4d scores: largest difference {worst:.4f} "
            f"over {len(checks)} scores -> {'MATCH' if worst < 1e-6 else 'DIFFERENT - tell Claude'}")
    if missing:
        log(f"  ({missing} scores had no saved value to compare with)")

    # figure
    n_cols = 3 + len(COLUMNS)
    fig, axes = plt.subplots(len(panels), n_cols, figsize=(1.55 * n_cols, 1.95 * len(panels)))
    titles = ["Photo", "Depth", "Truth"] + [label for _, label in COLUMNS]
    for row, (name, tree, rgb, depth, truth, masks) in enumerate(panels):
        outline = truth ^ ndimage.binary_erosion(truth, iterations=1)
        depth_img = np.where(depth > 0, depth, 0)
        images = [np.asarray(Image.fromarray(rgb).resize((256, 256))), depth_img, truth]
        for col in range(n_cols):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(images[0])
            elif col == 1:
                ax.imshow(images[1], cmap="gray", vmin=0, vmax=max(1, depth_img.max()))
            elif col == 2:
                ax.imshow(images[2], cmap="gray", vmin=0, vmax=1)
            else:
                method = COLUMNS[col - 3][0]
                pred = np.dstack([masks[method]] * 3).astype(float)
                pred[outline] = matplotlib.colors.to_rgb(ORANGE)
                ax.imshow(pred)
                r = scores[(scores.tree_id == tree) & (scores.method == method)].iloc[0]
                ax.set_xlabel(f"Dice {r.dice:.2f} | IoU {r.iou:.2f}", fontsize=7)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"{name}\n{tree}", fontsize=8)
    fig.suptitle("Every method on the same three test trees (the shape method's best, median and "
                 "worst); predicted masks in white, truth outline in orange", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig_all_methods.png", dpi=160)
    plt.close(fig)
    log(f"\nWrote {out / 'fig_all_methods.png'} and gallery_scores.csv")
    with pd.option_context("display.width", 160):
        log(scores.pivot(index="method", columns="row", values="dice")
            .loc[[m for m, _ in COLUMNS], ["best", "median", "worst"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
