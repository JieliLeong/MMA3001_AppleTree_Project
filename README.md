# Apple-tree scaffold recovery from summer RGB-D photos

MMA3001 Numerical Methods and Machine Learning, individual project (Monash, 2026).

In summer, most of an apple tree's main scaffold (the trunk and the two leaders a
pruning or harvesting robot must avoid) is hidden behind leaves and fruit. This
project predicts the scaffold from one summer colour photo and depth image, and
validates the prediction against the supplied ground-truth masks on trees the
method has never seen.

**Main finding.** Pixel-by-pixel machine-learning models (logistic regression,
random forest, neural network) could not beat a simple baseline, the average
training mask. They mostly learned *where* scaffolds usually are, because the
scaffold itself is hidden. Treating the scaffold as one **shape**, described by a
few SVD modes, predicting those modes from the photo with ridge regression, and
drawing the prediction as a Monte Carlo probability band, did beat the baseline
on the 85 test trees (see *Evaluation sequence* under
*Validation and verification* for how these trees were used).

## Result on the 85 test trees

| Method | Dice [95% CI] | Centreline F1, 5 px [95% CI] | Time per photo |
|---|---|---|---|
| **Shape method (final)** | **0.434** [0.409, 0.459] | **0.406** [0.365, 0.446] | 522 ms |
| Shape method, fast variant (no bark map, M = 200) | 0.418 [0.393, 0.442] | 0.394 [0.363, 0.425] | 93 ms |
| Baseline: average mask (train + validation) | 0.386 [0.361, 0.411] | 0.344 [0.308, 0.379] | < 1 ms |
| Stage 3: neural network (per pixel) | 0.381 [0.358, 0.403] | 0.387 [0.362, 0.410] | 57 ms |
| Stage 3: random forest (per pixel) | 0.362 [0.342, 0.381] | 0.333 [0.316, 0.348] | 499 ms |

Paired, tree by tree, final shape method minus baseline: **Dice +0.048
[+0.028, +0.068], better on 57 of 85 trees**; centreline F1 +0.062 [+0.027,
+0.098]. The validation estimate (0.420, gain +0.050) was close to the test
result (0.434, gain +0.048). That is consistent with little overfitting to the
validation trees, but with 84-85 trees, and with the test trees already used in
Stage 3, it does not rule overfitting out. On centreline F1 the shape
method ties the Stage 3 neural network (+0.019 [-0.013, +0.052]).

Figures: [`results/final/`](results/final) (scores, best / median / worst test
trees, occlusion check). Full numbers: `results/final/final_summary.csv` and
`paired.csv`.

## Inputs and outputs

| | Description |
|---|---|
| Input | Summer colour photo (640 x 480, 8-bit RGB) and depth image (640 x 480, 8-bit encoded, `0` taken to mean no sensor return). Both are resized to the 256 x 256 grid of the ground-truth masks |
| Output | Probability map (256 x 256, values 0-1) that each pixel is scaffold, a thresholded mask, and the two predicted centreline curves |
| Ground truth | Supplied 256 x 256 binary masks of trunk + two leaders (561 labelled trees); how they were made is not documented |
| Intended use | Perception step for an orchard robot: where the main branches run, in image coordinates |
| Limits | 2-D only; depth is a relative encoding (not millimetres) and camera intrinsics are unknown, so no metric 3-D lengths |

## Method, stage by stage

| Stage | What | Course link | Outcome |
|---|---|---|---|
| 1 | Index, pair and audit the data; split **by tree** into 392 train / 84 validation / 85 test | Weeks 1, 5.6 | 4,981 files; 561 labelled + 544 unlabelled trees; every tree complete |
| 1b | Check that the supplied masks line up with the images: shift and stretch every mask over its summer and winter photo and depth image, in 5 separate groups of trees | Week 5.5 | No reliable global correction: the groups disagree and the best positions sit at the edge of the search range. The cause of the discrepancies is undetermined (see *Validation*) |
| 2 | Fill depth holes: Laplace equation (direct sparse LU, Jacobi, Gauss-Seidel, SOR) vs griddata / RBF interpolation, on punched holes with known answers | Weeks 2.3, 3.5, 10.1, 11.3 | See `results/holes/`. On test fields the direct solve reproduces an exactly harmonic field to ~1e-11; Jacobi and Gauss-Seidel need ~R^2 iterations, SOR ~R |
| 3 | Per-pixel classifiers: logistic regression, random forest, MLP vs average-mask and colour-rule baselines | Weeks 5.1-5.6 | Best (MLP) ties the baseline: paired Dice -0.007 [-0.015, +0.002]; x and y position dominate |
| 4a | Feasibility of a shape representation (two centreline curves, SVD modes) | Weeks 2.3, 3.4 | Redraw Dice 0.91-0.96; 5 modes = 95 % of variation; oracle with 5 true modes 0.76 |
| 4b | Shape regressors (ridge, KNN, random forest, MLP) + Monte Carlo probability band; validation ladder | Weeks 3.6, 5, 6 | Ridge + band best; simple model wins on 392 trees |
| 4c | Settings chosen by rules fixed in advance; performance studies | Weeks 2.1, 2.2, 5.5 | Bark map yes, K = 3, 8 x 8 summary, 16 rows, M = 800; Monte Carlo error slope -0.48 (theory -0.5) |
| 4d | Final test: retrain on train + validation, score the shape method once on the 85 test trees | Week 5.5.10 | Table above |

## Validation and verification

* **Evaluation against supplied annotations:** predictions are compared with the
  supplied masks on trees excluded from model fitting. Scores measure agreement with
  these annotations, whose alignment with physical branches remains uncertain.
* **Evaluation sequence:** the dataset was split by tree into 392 training, 84
  validation and 85 test trees. Stage 3 evaluated the pixel-based models on the test
  partition. Later stages developed the shape-based approach using training and
  validation data, then evaluated it on the same 85 test trees. Because Stage 3 test
  results were already available during subsequent development, this partition cannot
  be described as an untouched final test set for the complete project.
* **Baselines and paired comparisons:** methods are compared tree by tree against the
  average-mask baseline. Paired differences and 95% bootstrap confidence intervals,
  calculated using 2,000 resamples, quantify differences in agreement with the
  supplied annotations.
* **Controls against data leakage:** splitting by tree keeps observations of the same
  tree together. Bark maps for development trees are generated out of fold. Prediction
  thresholds and model settings are selected using development data rather than final
  test scores.
* **Verification against known solutions:** the Laplace solvers reproduce an exactly
  harmonic field. A probability band without image information approximates the
  average-mask baseline. Monte Carlo error decreases approximately at the theoretical
  rate, and the vectorised and loop implementations produce identical results.
* **Repeatability:** five random seeds produced validation Dice of approximately 0.418
  ± 0.001, where the variation is the standard deviation across seeds.
* **Ground-truth alignment:** visual inspection revealed discrepancies between the
  supplied masks and visible trunk and branch locations. A registration analysis
  explored shifts and stretches across summer and winter observations but did not
  establish a reliable global correction. This does not determine whether the
  discrepancies arise from annotation approximation, image correspondence or
  coordinate mapping. Consequently, Dice, IoU and centreline F1 quantify agreement
  with the supplied annotations. They do not independently establish accurate
  localisation of the physical branches. Comparisons use the same reference masks, but
  annotation errors may affect different methods differently.
* **Limitations of validation:** the evaluation does not establish metric 3D accuracy,
  performance in other orchards, seasons or camera configurations, or the correctness
  of the supplied annotations. The depth conversion and camera calibration remain
  unverified.

## Data facts that shape the method

Confirmed on the full dataset by the audit (`results/audit/`).

| Item | Finding | Consequence |
|---|---|---|
| Depth PNG | 8-bit, three identical channels; max values 152-174. `0` appears where nothing is in range (sky, distant background), so it is taken to mean no return. The direction of the scale is undocumented: on R01N01 values fall along the ground away from the camera, which suggests larger = nearer (not checked across the dataset) | Relative depth only; no metric 3-D. No method or result depends on the direction |
| GT PNG | 256 x 256 binary (0/255) | All scoring on the 256 x 256 grid; photos are resized (squashed) to match. How the masks were registered to the photos is not documented |
| GT content | Main trunk + two leaders, drawn as a thick line | Centreline F1 reported alongside Dice |
| GT accuracy | Smooth Y-shaped masks with visible discrepancies from the trunk on some trees (e.g. R01N01); no reliable global correction found over 561 trees (Stage 1b); annotation accuracy and coordinate mapping unverified | Scores measure agreement with the supplied annotations; reported as a limitation |
| File times | `h-m-s`, not zero-padded (`12-41-4`) | Names parsed with a regular expression |
| Re-matched winter files | 104 files named like `R04N15_Winter_RGB-R04N42_RGB_12-19-45.png` | Parser accepts both forms; second ID kept as `source_id` |

The dataset belongs to its providers and is **not** included in this repository.

## Reproducing the results

1. Put the two dataset zip files in a Google Drive folder `MyDrive/MMA3001_data`.
2. Open [`notebooks/run_all.ipynb`](notebooks/run_all.ipynb) in Google Colab and
   choose **Runtime > Run all**. It clones this repository, unpacks the data,
   runs the tests and every stage in order (about 65 minutes in total), and
   rebuilds the documentation.

Everything is seeded, so a rerun gives the same scores; only run times vary.
Individual stages can be run from the repository root, for example:

```bash
pip install -r requirements.txt
python scripts/scaffold_experiment.py /content/data --out results/scaffold
python scripts/final_test.py /content/data --out results/final
```

## Repository layout

```
apple_scaffold/            the package
  data.py                  file-name parsing, frame pairing, loaders, grouped split
  alignment.py             do the masks line up with the images? (shift / stretch registration)
  holes.py                 depth-hole filling: Laplace solvers and interpolators
  features.py              per-pixel features (colour, texture, depth, position)
  baselines.py             average-mask and colour-rule baselines
  evaluate.py              Dice, IoU, centreline F1, thresholds, bootstrap CIs, occlusion proxy
  shapes.py                centreline description, PCHIP drawing, SVD shape model, Monte Carlo band
  shape_pipeline.py        shared steps of the shape method
scripts/                   one script per stage (audit, alignment_check, holes, scaffold, shape_*, final_test)
tests/                     pytest tests, including manufactured-solution checks
results/                   CSVs and figures, one folder per stage
notebooks/run_all.ipynb    reproduces everything in Colab
docs/                      HTML documentation (pdoc); open docs/apple_scaffold.html
reports/                   pytest report (test_report.html, test_report.xml)
AI_USE.md                  how AI was used, checked and corrected
LICENSE                    MIT licence (code only)
```

## Documentation and tests

* HTML documentation is generated from the docstrings with
  `pdoc --math ./apple_scaffold -o ./docs`. Download the `docs` folder and open
  `apple_scaffold.html` in a browser.
* Tests: `python -m pytest` (67 tests). The latest report is
  `reports/test_report.html`.

## AI use

See [`AI_USE.md`](AI_USE.md).

## Licence

Code: MIT (see `LICENSE`). The dataset is not covered by this licence.
