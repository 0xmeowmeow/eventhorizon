import numpy as np
import pytest

from luminet import bank, fast, spin


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory, monkeypatch_module=None):
    """A bank of three coarse maps in a private cache, not the user's."""
    root = tmp_path_factory.mktemp("cache")
    saved = (bank.RINGS, bank.ANGLES, bank.OUTER)
    import os
    old_env = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(root)
    bank.RINGS, bank.ANGLES, bank.OUTER = 24, 60, 200.0
    try:
        for incl in (1.35, 1.40, 1.45):
            bank.solve(incl)
        yield bank.Bank()
    finally:
        bank.RINGS, bank.ANGLES, bank.OUTER = saved
        if old_env is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = old_env


def test_banked_inclination_is_exact(small_bank):
    m = small_bank.mapping(1.40)
    direct = spin.lensing_map({"mass": 1.0, "incl": 1.40, "acc": 1.0, "outer_edge": 200.0},
                              n_rings=24, n_angles=60, spacing="log")
    assert np.allclose(m["tables"][0][0], direct["tables"][0][0], equal_nan=True)


def test_between_inclinations_blends_neighbours(small_bank):
    lo, hi, f = small_bank.neighbours(1.425)
    assert (lo, hi) == (1.40, 1.45) and f == pytest.approx(0.5)
    a, b, m = small_bank.mapping(1.40), small_bank.mapping(1.45), small_bank.mapping(1.425)
    both = np.isfinite(a["tables"][1][0]) & np.isfinite(b["tables"][1][0])
    assert np.allclose(m["tables"][1][0][both],
                       0.5 * (a["tables"][1][0][both] + b["tables"][1][0][both]))


def test_mass_scales_the_banked_map(small_bank):
    one, three = small_bank.mapping(1.40, mass=1.0), small_bank.mapping(1.40, mass=3.0)
    assert np.allclose(three["radii"], 3 * one["radii"])
    assert np.allclose(three["tables"][0][0], 3 * one["tables"][0][0], equal_nan=True)
    assert np.allclose(three["tables"][0][1], one["tables"][0][1], equal_nan=True)
    assert three["bh"].critical_b == pytest.approx(3 * 3 * np.sqrt(3))


def test_missing_neighbour_is_not_ready(small_bank):
    assert small_bank.ready_for(1.42)
    assert not small_bank.ready_for(1.10)


def test_reach_never_collapses_inside_the_inner_edge(small_bank):
    m = small_bank.mapping(1.40, mass=3.0)
    rx, ry = spin.reach(m, (0, 1), max_radius=5.0)      # inside the inner edge at 18
    assert rx > 3 * 5.0 and ry > 3.0


@pytest.mark.skipif(not fast.available, reason="numba is not installed")
def test_live_estimate_from_a_subset_matches_the_settled_field(small_bank):
    from scipy.ndimage import uniform_filter

    m = small_bank.mapping(1.40)
    rates = spin.true_rates(m["radii"], 1.0)
    rx, ry = spin.reach(m, (0, 1), max_radius=30)
    extent = (rx * 0.8, ry * 0.95)
    projector = fast.Projector(m, (0, 1))
    width, height = 120, 70
    parcels = spin.Parcels(m["radii"], count=width * height * 3, seed=4, clumps=0,
                           spread="log")
    field = spin.DotField(m, parcels, width, height, extent, rates, projector=projector,
                          floor=0.01, cell_aspect=1.0)
    live = spin.LiveField(width, height, gamma=0.6, floor=0.01, scale=field.scale,
                          cell_aspect=1.0, stride=3)
    settled = np.mean([field.frame(t, rates).copy() for t in (0, 2, 4)], axis=0)
    moving = np.mean([live.measure(parcels, t, rates, extent, projector).frame(t, rates).copy()
                      for t in (0, 2, 4)], axis=0)
    assert abs(moving.mean() - settled.mean()) < 0.3 * settled.mean()
    a = uniform_filter(settled, 8).ravel()
    b = uniform_filter(moving, 8).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.9
