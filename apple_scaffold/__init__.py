"""Scaffold recovery for occluded apple trees from summer RGB-D photos (MMA3001 project).

Modules, in the order the project uses them:

* `apple_scaffold.data` - parse file names, pair each tree's files, load images,
  split trees into train / validation / test without leakage.
* `apple_scaffold.holes` - fill missing depth by solving Laplace's equation
  (direct sparse LU, Jacobi, Gauss-Seidel, SOR) or by scattered-data interpolation.
* `apple_scaffold.features` - per-pixel features for the Stage 3 classifiers.
* `apple_scaffold.baselines` - the average-mask and colour-rule baselines.
* `apple_scaffold.evaluate` - Dice, IoU, centreline F1, threshold tuning,
  bootstrap confidence intervals, occlusion proxy.
* `apple_scaffold.shapes` - describe a scaffold as two centreline curves, draw it
  back (PCHIP), SVD shape model, Monte Carlo probability band.
* `apple_scaffold.shape_pipeline` - the shared steps of the final shape method.

The experiments that use these modules are in ``scripts/``; results are in ``results/``.
"""
