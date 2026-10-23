"""Filling missing depth values ("holes") in the apple-tree depth images.

A *hole* is a group of pixels with no sensor return (``depth == 0``) that is
enclosed by valid pixels. Large zero regions touching the image border are
usually out-of-range background (sky, distant rows), so they are not holes.

Two families of method are compared:

**Laplace (harmonic) filling** - MMA3001 Weeks 2.3, 10.1, 11.3.
    Inside each hole, solve Laplace's equation
    ``d2u/dx2 + d2u/dy2 = 0`` with the surrounding valid pixels as Dirichlet
    boundary values. With the five-point stencil (grid spacing 1 pixel) every
    hole pixel satisfies::

        4 u[i, j] - u[i-1, j] - u[i+1, j] - u[i, j-1] - u[i, j+1] = 0

    i.e. each filled value is the average of its four neighbours. Writing this
    for all hole pixels gives a sparse linear system ``A u = b``. ``A`` has 4 on
    the diagonal and -1 for each neighbouring hole pixel; ``b`` collects the
    known boundary values. It is solved

    * directly (:func:`solve_direct`, sparse LU via ``scipy.sparse.linalg.spsolve``),
    * by Jacobi iteration (:func:`solve_jacobi`),
    * by Gauss-Seidel / successive over-relaxation (:func:`solve_sor`).

    ``A`` is only *weakly* diagonally dominant (``|4| = 1+1+1+1`` for pixels
    whose four neighbours are all in the hole), strictly so in rows next to the
    boundary, and the hole is connected, so Jacobi still converges (irreducible
    diagonal dominance) - but slowly for large holes.

**Scattered-data interpolation** - MMA3001 Week 3.5.
    ``scipy.interpolate.griddata`` (``nearest``, ``linear``, ``cubic``) and
    ``RBFInterpolator`` (thin-plate spline), fed with the valid pixels in a
    ring around each hole (:func:`fill_interpolate`).

Depth values are 8-bit encoded levels (not millimetres), so errors are in
**levels**. Hole pixels in the input may hold any value; they are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import RBFInterpolator, griddata
from scipy.sparse.linalg import spsolve

LAPLACE_METHODS = ("direct", "jacobi", "gauss_seidel", "sor")
INTERP_METHODS = ("nearest", "linear", "cubic", "rbf")
METHODS = INTERP_METHODS + LAPLACE_METHODS
"""All fill methods accepted by :func:`fill_holes`."""

_CROSS = ndimage.generate_binary_structure(2, 1)   # 4-connected neighbourhood


@dataclass
class FillResult:
    """Output of a fill method.

    Attributes
    ----------
    depth : numpy.ndarray
        Float copy of the input with the hole pixels filled.
    iterations : int
        Iterations used (0 for non-iterative methods).
    residual : float
        Largest absolute residual ``|4u - sum of neighbours|`` of the discrete
        Laplace equation over the hole pixels after filling (``nan`` for the
        interpolation methods, which do not solve that equation).
    converged : bool
        False if an iterative solver stopped at ``max_iter``.
    """

    depth: np.ndarray
    iterations: int = 0
    residual: float = float("nan")
    converged: bool = True


# --------------------------------------------------------------------------
# Finding holes
# --------------------------------------------------------------------------
def find_holes(depth: np.ndarray, max_area: int = 2000,
               exclude_border: bool = True) -> np.ndarray:
    """Return a mask of the hole pixels worth filling.

    Parameters
    ----------
    depth : numpy.ndarray
        2-D depth image; ``0`` means no return.
    max_area : int
        Zero regions larger than this (pixels) are treated as background.
    exclude_border : bool
        Drop zero regions that touch the image border.

    Returns
    -------
    numpy.ndarray of bool
        True for pixels in 4-connected zero regions that pass both tests.
    """
    labels, n = ndimage.label(depth == 0, structure=_CROSS)
    if n == 0:
        return np.zeros(depth.shape, bool)
    area = np.bincount(labels.ravel(), minlength=n + 1)
    keep = area <= max_area
    keep[0] = False
    if exclude_border:
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        keep[np.unique(edge)] = False
    return keep[labels]


# --------------------------------------------------------------------------
# Laplace filling
# --------------------------------------------------------------------------
def _setup(depth: np.ndarray, hole: np.ndarray):
    """Flattened working copy, hole indices and their four neighbour indices.

    The initial guess in each hole is the mean of its boundary values.
    """
    hole = np.asarray(hole, bool)
    if hole.shape != depth.shape:
        raise ValueError("hole mask and depth image must have the same shape")
    if hole[0].any() or hole[-1].any() or hole[:, 0].any() or hole[:, -1].any():
        raise ValueError("holes must not touch the image border")
    width = depth.shape[1]
    u = depth.astype(np.float64).ravel().copy()
    p = np.flatnonzero(hole)
    nb = np.stack([p - width, p + width, p - 1, p + 1])      # up, down, left, right
    boundary = ~hole.ravel()[nb]
    if np.any(boundary & (u[nb] <= 0)):
        raise ValueError("a hole borders a pixel with no depth; enlarge the hole mask")

    labels, n = ndimage.label(hole, structure=_CROSS)
    grown = ndimage.grey_dilation(labels, footprint=_CROSS)
    ring = (grown > 0) & ~hole
    sums = np.bincount(grown[ring], weights=depth[ring].astype(float), minlength=n + 1)
    counts = np.bincount(grown[ring], minlength=n + 1)
    u[p] = (sums / np.maximum(counts, 1))[labels.ravel()[p]]
    return u, p, nb


def _residual(u: np.ndarray, p: np.ndarray, nb: np.ndarray) -> float:
    """Largest absolute residual of the five-point Laplace equation."""
    if p.size == 0:
        return 0.0
    return float(np.max(np.abs(4.0 * u[p] - u[nb].sum(axis=0))))


def solve_direct(depth: np.ndarray, hole: np.ndarray) -> FillResult:
    """Fill holes by assembling ``A u = b`` and solving it with sparse LU."""
    u, p, nb = _setup(depth, hole)
    n = p.size
    if n == 0:
        return FillResult(u.reshape(depth.shape), residual=0.0)
    position = np.full(u.size, -1)
    position[p] = np.arange(n)
    rows, cols, b = [np.arange(n)], [np.arange(n)], np.zeros(n)
    for k in range(4):
        j = position[nb[k]]
        inside = j >= 0
        rows.append(np.flatnonzero(inside))
        cols.append(j[inside])
        b += np.where(inside, 0.0, u[nb[k]])               # known boundary values
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    values = np.where(rows == cols, 4.0, -1.0)
    A = sparse.csc_matrix((values, (rows, cols)), shape=(n, n))
    u[p] = spsolve(A, b)
    return FillResult(u.reshape(depth.shape), residual=_residual(u, p, nb))


def solve_jacobi(depth: np.ndarray, hole: np.ndarray, tol: float = 1e-3,
                 max_iter: int = 200_000) -> FillResult:
    """Fill holes by Jacobi iteration on the five-point Laplace equation.

    Every hole pixel is replaced by the average of its four neighbours from the
    *previous* iterate. Stops when the largest residual falls below ``tol``
    (depth levels). The residual of the previous iterate equals 4 x the update,
    so no extra pass is needed to test convergence.
    """
    u, p, nb = _setup(depth, hole)
    for k in range(1, max_iter + 1):
        new = 0.25 * u[nb].sum(axis=0)
        change = np.max(np.abs(new - u[p])) if p.size else 0.0
        u[p] = new
        if 4.0 * change < tol:
            return FillResult(u.reshape(depth.shape), k, _residual(u, p, nb), True)
    return FillResult(u.reshape(depth.shape), max_iter, _residual(u, p, nb), False)


def optimal_omega(hole: np.ndarray) -> float:
    """Over-relaxation factor estimated from the size of the largest hole.

    Uses the model-problem result for an ``n x n`` square,
    ``omega = 2 / (1 + sin(pi / (n + 1)))``, with ``n = sqrt(area)``.
    """
    labels, n = ndimage.label(hole, structure=_CROSS)
    if n == 0:
        return 1.0
    side = np.sqrt(np.bincount(labels.ravel())[1:].max())
    return float(2.0 / (1.0 + np.sin(np.pi / (side + 1.0))))


def solve_sor(depth: np.ndarray, hole: np.ndarray, omega: float | None = None,
              tol: float = 1e-3, max_iter: int = 200_000) -> FillResult:
    """Fill holes by red-black Gauss-Seidel with optional over-relaxation.

    Hole pixels are split like a chessboard. All "red" pixels are updated
    first; the "black" pixels then use the new red values, which is what makes
    this Gauss-Seidel rather than Jacobi (and lets NumPy update each colour in
    one vectorised step). ``omega = 1`` is plain Gauss-Seidel; ``1 < omega < 2``
    is successive over-relaxation (SOR). ``omega=None`` uses
    :func:`optimal_omega`.
    """
    u, p, nb = _setup(depth, hole)
    if omega is None:
        omega = optimal_omega(hole)
    if not 0.0 < omega < 2.0:
        raise ValueError("omega must lie in (0, 2) for convergence")
    width = depth.shape[1]
    red = ((p // width) + (p % width)) % 2 == 0
    colours = [(p[red], nb[:, red]), (p[~red], nb[:, ~red])]
    for k in range(1, max_iter + 1):
        largest = 0.0
        for pc, nbc in colours:
            if pc.size == 0:
                continue
            delta = 0.25 * u[nbc].sum(axis=0) - u[pc]
            u[pc] += omega * delta
            largest = max(largest, float(np.max(np.abs(delta))))
        if 4.0 * largest < tol:
            return FillResult(u.reshape(depth.shape), k, _residual(u, p, nb), True)
    return FillResult(u.reshape(depth.shape), max_iter, _residual(u, p, nb), False)


# --------------------------------------------------------------------------
# Interpolation baselines
# --------------------------------------------------------------------------
def fill_interpolate(depth: np.ndarray, hole: np.ndarray, method: str = "linear",
                     ring: int = 3) -> FillResult:
    """Fill each hole by scattered-data interpolation from a ring of valid pixels.

    Parameters
    ----------
    method : {"nearest", "linear", "cubic", "rbf"}
        ``griddata`` method, or ``"rbf"`` for a thin-plate-spline
        ``RBFInterpolator``.
    ring : int
        Width (pixels) of the band of valid pixels used around each hole.
    """
    if method not in INTERP_METHODS:
        raise ValueError(f"unknown interpolation method {method!r}")
    hole = np.asarray(hole, bool)
    out = depth.astype(np.float64).copy()
    labels, _ = ndimage.label(hole, structure=_CROSS)
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        box = tuple(slice(max(s.start - ring, 0), min(s.stop + ring, size))
                    for s, size in zip(box, depth.shape))
        target = labels[box] == k
        local = depth[box]
        band = (ndimage.binary_dilation(target, structure=_CROSS, iterations=ring)
                & ~target & ~hole[box] & (local > 0))
        points = np.argwhere(band).astype(float)
        values = local[band].astype(float)
        query = np.argwhere(target).astype(float)
        if method == "rbf":
            filled = RBFInterpolator(points, values, kernel="thin_plate_spline")(query)
        else:
            filled = griddata(points, values, query, method=method)
            missing = np.isnan(filled)
            if missing.any():                    # outside the convex hull of the ring
                filled[missing] = griddata(points, values, query[missing], method="nearest")
        view = out[box]
        view[target] = filled
    return FillResult(out)


# --------------------------------------------------------------------------
# One entry point
# --------------------------------------------------------------------------
def fill_holes(depth: np.ndarray, hole: np.ndarray, method: str = "direct",
               **options) -> FillResult:
    """Fill ``hole`` pixels of ``depth`` with any method in :data:`METHODS`.

    ``options`` are passed on (e.g. ``tol``, ``omega``, ``ring``).
    ``"gauss_seidel"`` is :func:`solve_sor` with ``omega=1``.
    """
    if method in INTERP_METHODS:
        return fill_interpolate(depth, hole, method, **options)
    if method == "direct":
        return solve_direct(depth, hole)
    if method == "jacobi":
        return solve_jacobi(depth, hole, **options)
    if method == "gauss_seidel":
        return solve_sor(depth, hole, omega=1.0, **options)
    if method == "sor":
        return solve_sor(depth, hole, **options)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


# --------------------------------------------------------------------------
# Validation helper: holes with known answers
# --------------------------------------------------------------------------
def disc(radius: int) -> np.ndarray:
    """Boolean disc of the given radius (shape ``2r+1`` square)."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def punch_discs(valid: np.ndarray, radius: int, count: int,
                rng: np.random.Generator, margin: int = 4) -> np.ndarray:
    """Place up to ``count`` non-overlapping disc-shaped holes in valid depth.

    Each disc, grown by ``margin`` pixels, lies wholly inside ``valid`` and away
    from the image border, so every boundary and ring pixel has a known value.
    The true depth under the discs is then the reference for scoring a fill.
    """
    clear = ndimage.binary_erosion(valid, structure=disc(radius + margin), border_value=0)
    mask = np.zeros(valid.shape, bool)
    shape, reach = disc(radius), 2 * radius + margin + 1
    for _ in range(count):
        centres = np.argwhere(clear)
        if len(centres) == 0:
            break
        cy, cx = centres[rng.integers(len(centres))]
        mask[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] |= shape
        clear[max(cy - reach, 0):cy + reach + 1, max(cx - reach, 0):cx + reach + 1] = False
    return mask
