import numpy as np
import pytest

from luminet import fast, spin

pytestmark = pytest.mark.skipif(not fast.available, reason="numba is not installed")


@pytest.fixture(scope="module")
def disk():
    settings = {"mass": 1.0, "incl": 1.45, "acc": 1.0, "outer_edge": 60.0}
    mapping = spin.lensing_map(settings, n_rings=30, n_angles=90)
    rx, ry = spin.reach(mapping, max_radius=30)
    rates = spin.true_rates(mapping["radii"], 1.0)
    return mapping, (rx * 0.8, ry * 0.95), rates


@pytest.mark.parametrize("width,height", [(120, 64), (60, 120), (200, 40)])
def test_compiled_dots_match_numpy(disk, width, height):
    """
    The compiled path has to light exactly the dots the numpy path does.

    This compares the whole of dots() in both modes rather than the kernel
    against a copied formula, because a copied formula can share the kernel's
    mistake: one did, and passed a kernel that drew the picture upside down.
    """
    mapping, extent, rates = disk
    parcels = spin.Parcels(mapping["radii"], count=width * height * 2, seed=3,
                           clumps=0, spread="log")
    projector = fast.Projector(mapping, (0, 1))

    for t in (0.0, 2.5, 31.0):
        lit_np, bright_np = spin.dots(mapping, parcels, t, width, height, extent, rates)
        lit_nb, bright_nb = spin.dots(mapping, parcels, t, width, height, extent, rates,
                                      projector=projector)
        assert lit_np.any()
        assert np.array_equal(lit_np, lit_nb)
        assert np.allclose(bright_np, bright_nb, rtol=0, atol=1e-12)


def test_compiled_dots_are_right_way_up(disk):
    """The shadow sits above the ghost image, not below it."""
    mapping, extent, rates = disk
    width, height = 120, 120
    parcels = spin.Parcels(mapping["radii"], count=width * height * 2, seed=3,
                           clumps=0, spread="log")
    lit, _ = spin.dots(mapping, parcels, 0.0, width, height, extent, rates,
                       projector=fast.Projector(mapping, (0,)), orders=(0,))
    rows = np.flatnonzero(lit.any(axis=1))
    # An inclined disk seen from above: the lensed far side arcs over the top
    # of the shadow, so light reaches higher above centre than below it.
    centre = height / 2
    assert (centre - rows.min()) > (rows.max() - centre)
