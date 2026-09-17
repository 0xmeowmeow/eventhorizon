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


def test_isolines_trace_the_rings_and_flow(disk):
    """Isoradials land on the grid, and flowing dashes move with the gas."""
    mapping, extent, rates = disk
    width, height = 160, 96
    ext_x, ext_y = spin.fit_extent(extent[0], width, height, reach_y=extent[1])
    lines = spin.Isolines(mapping, width, height, ext_x, ext_y)
    assert len(lines.lines) >= 6

    solid = lines.frame(0.0, rates).copy()
    assert solid.any()
    early = lines.frame(0.0, rates, flowing=True).copy()
    later = lines.frame(2.0, rates, flowing=True).copy()
    # Dashes are a subset of the solid lines, and they move over time.
    assert not (early & ~solid).any()
    assert (early ^ later).any()


def test_braille_backgrounds_are_sent_per_cell():
    from luminet import cells

    mask = np.ones((4, 4), dtype=bool)
    fg = np.array([[[200, 0, 0], [200, 0, 0]]], dtype=np.uint8)
    bg = np.array([[[0, 0, 50], [0, 0, 90]]], dtype=np.uint8)
    out = cells.braille(mask, colours=fg, backgrounds=bg)
    assert out.count("38;2;200;0;0") == 1
    assert "48;2;0;0;50" in out and "48;2;0;0;90" in out
