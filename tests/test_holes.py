"""Tests for apple_scaffold.holes.

The Laplace solvers are verified against *manufactured solutions*: fields with
a known answer inside the hole. ``x**2 - y**2`` satisfies the five-point
Laplace equation exactly (the stencil differentiates quadratics without
truncation error), so a correct solver must reproduce it to round-off.
"""
import numpy as np
import pytest

from apple_scaffold import holes


# ---------------------------------------------------------------- helpers
def grid(shape=(41, 51)):
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    return y, x


def harmonic_field():
    """Discretely harmonic field, kept positive so every pixel is 'valid'."""
    y, x = grid()
    return (x - 25) ** 2 - (y - 20) ** 2 + 2 * x + 3 * y + 600.0


def centre_hole(shape=(41, 51), radius=6):
    mask = np.zeros(shape, bool)
    cy, cx = shape[0] // 2, shape[1] // 2
    mask[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = holes.disc(radius)
    return mask


def with_hole(field, mask):
    corrupted = field.copy()
    corrupted[mask] = 0.0
    return corrupted


# ---------------------------------------------------------------- finding holes
def test_find_holes_keeps_only_small_enclosed_zero_regions():
    depth = np.full((40, 40), 100, np.uint8)
    depth[10:13, 10:13] = 0          # small enclosed hole       -> kept
    depth[0:5, 20:30] = 0            # touches the border        -> dropped
    depth[20:35, 15:35] = 0          # enclosed but large (300)  -> dropped
    mask = holes.find_holes(depth, max_area=100)
    assert mask.sum() == 9
    assert mask[10:13, 10:13].all()


# ---------------------------------------------------------------- verification
def test_direct_solver_reproduces_discrete_harmonic_field():
    field, mask = harmonic_field(), centre_hole()
    result = holes.solve_direct(with_hole(field, mask), mask)
    assert np.max(np.abs(result.depth[mask] - field[mask])) < 1e-8
    assert result.residual < 1e-8


@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel", "sor"])
def test_iterative_solvers_converge_to_the_direct_solution(method):
    field, mask = harmonic_field(), centre_hole()
    corrupted = with_hole(field, mask)
    reference = holes.solve_direct(corrupted, mask).depth
    result = holes.fill_holes(corrupted, mask, method, tol=1e-9)
    assert result.converged
    assert result.residual < 1e-8
    assert np.max(np.abs(result.depth[mask] - reference[mask])) == pytest.approx(0, abs=1e-6)


def test_gauss_seidel_beats_jacobi_and_sor_beats_both():
    field, mask = harmonic_field(), centre_hole(radius=10)
    corrupted = with_hole(field, mask)
    its = {m: holes.fill_holes(corrupted, mask, m, tol=1e-6).iterations
           for m in ("jacobi", "gauss_seidel", "sor")}
    assert its["sor"] < its["gauss_seidel"] < its["jacobi"]
    # theory: Gauss-Seidel needs about half as many iterations as Jacobi
    assert its["gauss_seidel"] / its["jacobi"] == pytest.approx(0.5, abs=0.1)


def test_laplace_fill_obeys_the_maximum_principle():
    """A harmonic fill can never exceed the range of its boundary values."""
    from scipy import ndimage
    rng = np.random.default_rng(0)
    depth = rng.uniform(50, 150, size=(30, 30))
    mask = centre_hole((30, 30), radius=5)
    boundary = ndimage.binary_dilation(mask) & ~mask
    filled = holes.solve_direct(with_hole(depth, mask), mask).depth[mask]
    assert depth[boundary].min() <= filled.min()
    assert filled.max() <= depth[boundary].max()


@pytest.mark.parametrize("method", ["linear", "cubic", "rbf"])
def test_interpolators_reproduce_a_plane(method):
    y, x = grid()
    plane = 2.0 * x + 3.0 * y + 40.0
    mask = centre_hole()
    result = holes.fill_holes(with_hole(plane, mask), mask, method)
    assert np.max(np.abs(result.depth[mask] - plane[mask])) < 1e-6


def test_every_method_fills_every_hole_pixel():
    field, mask = harmonic_field(), centre_hole()
    for method in holes.METHODS:
        filled = holes.fill_holes(with_hole(field, mask), mask, method).depth
        assert np.isfinite(filled).all(), method
        assert (filled[mask] > 0).all(), method


# ---------------------------------------------------------------- guards
def test_hole_touching_the_border_is_rejected():
    depth = np.full((20, 20), 100.0)
    mask = np.zeros((20, 20), bool)
    mask[0:3, 5:8] = True
    with pytest.raises(ValueError):
        holes.solve_direct(depth, mask)


def test_hole_next_to_missing_depth_is_rejected():
    depth = np.full((20, 20), 100.0)
    depth[10, 12] = 0                      # an unfilled zero beside the hole
    mask = np.zeros((20, 20), bool)
    mask[9:12, 9:12] = True
    with pytest.raises(ValueError):
        holes.solve_jacobi(depth, mask)


def test_punched_discs_lie_inside_valid_depth_and_do_not_overlap():
    valid = np.ones((120, 160), bool)
    valid[:, :40] = False
    mask = holes.punch_discs(valid, radius=5, count=6, rng=np.random.default_rng(1), margin=3)
    assert not (mask & ~valid).any()
    from scipy import ndimage
    _, n = ndimage.label(mask)
    assert n == 6
    assert mask.sum() == 6 * holes.disc(5).sum()
