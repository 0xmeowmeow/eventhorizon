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


@pytest.fixture(scope="module")
def field_setup(disk):
    mapping, extent, rates = disk
    width, height = 120, 64
    parcels = spin.Parcels(mapping["radii"], count=int(width * height * 2.5), seed=3,
                           clumps=0, spread="log")
    return mapping, extent, rates, width, height, parcels


def test_dotfield_compiled_matches_numpy(field_setup):
    """The precomputed field lights the same dots on both paths."""
    mapping, extent, rates, width, height, parcels = field_setup
    compiled = spin.DotField(mapping, parcels, width, height, extent, rates,
                             projector=fast.Projector(mapping, (0, 1)))
    plain = spin.DotField(mapping, parcels, width, height, extent, rates, projector=None)
    assert np.allclose(compiled.chance, plain.chance, rtol=0, atol=1e-12)
    for t in (0.0, 3.3, 27.0):
        assert np.array_equal(compiled.frame(t, rates).copy(), plain.frame(t, rates).copy())


def test_dotfield_keeps_the_look_of_per_frame_dots(field_setup):
    """
    Measuring brightness once instead of every frame should not change the
    picture: over time the dots should fall in the same places, about as thickly.
    """
    from scipy.ndimage import uniform_filter

    mapping, extent, rates, width, height, parcels = field_setup
    projector = fast.Projector(mapping, (0, 1))
    field = spin.DotField(mapping, parcels, width, height, extent, rates,
                          gamma=0.6, projector=projector)
    times = np.linspace(0, 40, 12)
    per_frame = np.mean([spin.dots(mapping, parcels, t, width, height, extent, rates,
                                   gamma=0.6, projector=projector)[0] for t in times], axis=0)
    once = np.mean([field.frame(t, rates).copy() for t in times], axis=0)

    assert abs(once.mean() - per_frame.mean()) < 0.25 * per_frame.mean()
    a = uniform_filter(per_frame.astype(float), 6).ravel()
    b = uniform_filter(once.astype(float), 6).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.95
