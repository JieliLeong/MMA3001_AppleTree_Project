# Apple-tree scaffold recovery (MMA3001 individual project)

Recover the main scaffold (trunk and leaders) of summer apple trees, whose
branches are partly hidden by foliage, from RGB-D images. The work uses the
MMA3001 RGB-D apple-tree dataset.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Index, pair and audit the data; leakage-safe split by tree | done |
| 2 | Fill holes in depth (Laplace solvers vs interpolation) | done |
| 3 | Find scaffold pixels: baselines (average mask, colour rule) vs logistic regression, random forest, neural network | done: best model (MLP) ties the average-mask baseline, paired Dice -0.007 [-0.015, +0.002]; position features dominate |
| 4a | Feasibility of predicting the scaffold as a shape (centreline curves + SVD modes) | done: representation Dice 0.91-0.96; 5 modes = 95 % of variation; oracle 5-mode Dice 0.76; ridge on a coarse summary 0.36 (baseline 0.37) |
| 4b | Shape method: bark maps (out-of-fold), ridge / KNN / random forest / MLP regressors, Monte Carlo fuzzy band, validation ladder | done: ridge + band, validation Dice 0.421 vs baseline 0.370, paired +0.051 [+0.033, +0.069], better on 59/84 trees |
| 4c | Settings (pre-stated rules) and performance: bark map, K, input grid, centreline rows, Monte Carlo M, learning curve, seeds, timing, FLOPs, memory | done: bark map yes, K=3, grid 8x8, 16 rows, M=800; validation Dice 0.420 vs 0.370; MC error slope -0.48 (theory -0.5); seeds 0.418 +/- 0.001 |
| 4d | Final test, once, on the 85 untouched test trees; paired comparisons with the baseline and every Stage 3 model | built |

## Data facts that affect the method

Checked on tree R01N01, 23 Oct 2026; run the audit to confirm across the dataset.

| Item | Finding | Consequence |
|---|---|---|
| Depth PNG | 8-bit greyscale stored as 3 identical channels; `0` = no return; larger = farther | Depth is a rescaled encoding, not mm, so metric 3D needs the scale and camera intrinsics |
| GT PNG | 256 x 256 binary (0/255) | Resize to 640 x 480 with nearest-neighbour, or evaluate at 256 x 256 |
| GT content | Main trunk + two leaders only, drawn as a thick line | Compare centrelines with a pixel tolerance, not just mask IoU |
| File times | `h-m-s`, not zero-padded (`12-41-4`) | Names are parsed with a regular expression |
| Re-matched winter files | Some winter files carry a second ID: `R04N15_Winter_RGB-R04N42_RGB_12-19-45.png` | Parser accepts both forms; the second ID is kept as `source_id` |

## Layout

```
apple_scaffold/data.py               filename parsing, frame pairing, loaders, grouped split
apple_scaffold/holes.py              depth-hole filling: Laplace (direct, Jacobi, Gauss-Seidel, SOR)
                                     and griddata / RBF interpolation
apple_scaffold/features.py           per-pixel features (colour, texture, depth, position)
apple_scaffold/baselines.py          average-mask and colour-rule baselines
apple_scaffold/evaluate.py           Dice, IoU, centreline F1, threshold tuning, bootstrap CIs
apple_scaffold/shapes.py             scaffold as two centreline curves; PCHIP drawing; SVD shape model;
                                     Monte Carlo probability band
apple_scaffold/shape_pipeline.py     shared steps of the shape method (inputs, ridge, band, scoring)
scripts/audit_dataset.py             dataset audit: completeness, image properties, split, overlays
scripts/hole_filling_experiment.py   accuracy, cost and convergence of every fill method
scripts/scaffold_experiment.py       Stage 3 comparison: train, tune on validation, score on test
scripts/shape_feasibility.py         Stage 4a: representation quality, shape modes, oracle limits
scripts/shape_method.py              Stage 4b: shape regressors and fuzzy band, validation ladder
scripts/shape_settings.py            Stage 4c: settings by pre-stated rules; performance studies
scripts/final_test.py                Stage 4d: retrain on train+val, score once on the test trees
tests/                               pytest tests (manufactured-solution checks for the solvers)
results/                             experiment CSVs and figures, one folder per stage
```

## Running in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<your-user>/<this-repo>.git
%cd <this-repo>
!pip install --quiet -r requirements.txt

# audit both dataset folders (edit the paths to where they sit in your Drive)
!python scripts/audit_dataset.py "/content/drive/MyDrive/AppleTree/Dataset (With Summer GT)" \
                                 "/content/drive/MyDrive/AppleTree/Dataset" --out audit

# depth-hole filling experiment (training trees only)
!python scripts/hole_filling_experiment.py /content/data --out results/holes --frames 20

# Stage 3 method comparison (add --fill-depth to use hole-filled depth)
!python scripts/scaffold_experiment.py /content/data --out results/scaffold

# Stage 4a shape feasibility check (training and validation trees only)
!python scripts/shape_feasibility.py /content/data --out results/shape_check

# Stage 4b shape method, validation ladder (training and validation trees only)
!python scripts/shape_method.py /content/data --out results/shape_method

# Stage 4c settings and performance (training and validation trees only)
!python scripts/shape_settings.py /content/data --out results/shape_settings

# Stage 4d final test (uses results/shape_settings/final_config.json and results/scaffold/metrics.csv)
!python scripts/final_test.py /content/data --out results/final

!pytest -v                                   # run the tests
!pdoc --math ./apple_scaffold -o ./docs      # HTML documentation
```

The audit writes `audit/frames.csv` (one row per tree and season, with any
problems), `audit/split.csv` (the saved train/val/test record, grouped by
tree; use `--group-by row` for the stricter row-wise split) and GT overlay
images in `audit/overlays/`.
