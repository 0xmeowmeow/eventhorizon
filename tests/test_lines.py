import numpy as np
import pytest

from luminet import cells, fast, lines, spin


@pytest.fixture(scope="module")
def lineset():
    settings = {"mass": 1.0, "incl": 1.4, "acc": 1.0, "outer_edge": 120.0}
    mapping = spin.lensing_map(settings, n_rings=40, n_angles=90, spacing="log")
    rx, ry = spin.reach(mapping, max_radius=30)
    rates = spin.true_rates(mapping["radii"], 1.0)
    saved = spin.CELL_ASPECT
    spin.CELL_ASPECT = 1.0
    try:
        grid_w, grid_h = 80, 50
        parcels = spin.Parcels(mapping["radii"], count=grid_w * grid_h * 2, seed=2,
                               clumps=0, spread="log")
        field = spin.DotField(mapping, parcels, grid_w, grid_h, (rx * 0.8, ry * 0.95), rates,
                              projector=fast.Projector(mapping, (0, 1)) if fast.available else None)
        ls = lines.LineSet(mapping, field, 320, 200, field.ext_x, field.ext_y)
    finally:
        spin.CELL_ASPECT = saved
    return ls, rates


ALL = {"radii", "redshift", "flux"}


@pytest.mark.parametrize("family", ["radii", "redshift", "flux"])
def test_each_family_draws_on_the_canvas(lineset, family):
    ls, rates = lineset
    idx, rgb = ls.draw(0.0, rates, {family})
    assert len(idx) > 50
    assert idx.min() >= 0 and idx.max() < ls.width * ls.height
    assert rgb.shape == (len(idx), 3)
    assert len(np.unique(idx)) == len(idx)


def test_solid_and_dotted_hold_still(lineset):
    ls, rates = lineset
    for style in ("solid", "dotted"):
        a = ls.draw(0.0, rates, ALL, style=style)
        b = ls.draw(5.0, rates, ALL, style=style)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    solid = len(ls.draw(0.0, rates, ALL, style="solid")[0])
    dotted = len(ls.draw(0.0, rates, ALL, style="dotted")[0])
    assert dotted < solid


def test_flowing_moves_and_is_a_subset_of_solid(lineset):
    ls, rates = lineset
    solid = set(ls.draw(0.0, rates, {"radii"}, style="solid")[0].tolist())
    early = ls.draw(0.0, rates, {"radii"}, style="flowing")[0]
    later = ls.draw(1.5, rates, {"radii"}, style="flowing")[0]
    assert set(early.tolist()) <= solid
    assert set(early.tolist()) != set(later.tolist())


def test_pulse_changes_brightness_not_shape(lineset):
    ls, rates = lineset
    a_idx, a_rgb = ls.draw(0.0, rates, {"radii"}, style="pulse")
    b_idx, b_rgb = ls.draw(1.0, rates, {"radii"}, style="pulse")
    assert np.array_equal(a_idx, b_idx)
    assert not np.array_equal(a_rgb, b_rgb)


def test_sweep_moves_a_line_and_leaves_nothing_behind(lineset):
    ls, rates = lineset
    solid, _ = ls.draw(2.0, rates, {"radii"}, style="solid")
    a_idx, a_rgb = ls.draw(2.0, rates, {"radii"}, style="sweep")
    b_idx, _ = ls.draw(4.0, rates, {"radii"}, style="sweep")
    assert set(a_idx.tolist()) != set(b_idx.tolist())
    assert 0 < len(a_idx) < len(solid) / 2
    assert a_rgb.max(axis=1).min() > 0          # no black pixels from a faded line


def test_sweeps_only_ever_move_one_way():
    from luminet.lines import LineSet

    seen = []
    for t in np.arange(0.0, 40.0, 0.05):
        sweeps = LineSet._sweeps(t, 8.0)
        assert 1 <= len(sweeps) <= 2
        seen.append(sweeps)
    # Following the newest sweep, progress only rises, then a new one starts low
    # while the old one is still visible.
    for before, after in zip(seen, seen[1:]):
        if len(after) == 2 and len(before) == 1:
            assert after[1][0] < 0.05 and after[0][0] > before[0][0]


def test_width_thickens_without_duplicates(lineset):
    ls, rates = lineset
    thin = ls.draw(0.0, rates, ALL, width=1)[0]
    thick = ls.draw(0.0, rates, ALL, width=3)[0]
    assert len(thick) > 1.5 * len(thin)
    assert len(np.unique(thick)) == len(thick)


@pytest.mark.parametrize("colour", ["blue", "ink", "palette", "redshift", "flux", "spectrum"])
def test_every_colour_mode_draws(lineset, colour):
    ls, rates = lineset
    idx, rgb = ls.draw(0.0, rates, ALL, colour=colour, palette=cells.palette("magma"))
    assert len(idx) and rgb.max() > 40
