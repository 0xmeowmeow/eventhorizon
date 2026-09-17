from types import SimpleNamespace

import numpy as np
import pytest

from luminet import black_hole_math as bhm
from luminet import effects, spin


@pytest.fixture(scope="module")
def mapping():
    return spin.lensing_map({"mass": 1.0, "incl": 1.3, "acc": 1.0, "outer_edge": 60.0},
                            n_rings=24, n_angles=90, spacing="log")


def fake_live(mapping):
    rates = spin.true_rates(mapping["radii"], 1.0, 0.12)
    return SimpleNamespace(mapping=mapping, rates=rates, clock=0.0, cols=80, rows=24,
                           settings={"incl": 1.3, "mass": 1.0, "outer_edge": 60.0},
                           o=SimpleNamespace(fps=30), paused=False, projector=None)


def test_glyph_encodings():
    assert effects.seven_segment("12.5 M") == "\U0001FBF1\U0001FBF2.\U0001FBF5 M"
    signs = effects.encode_message("hello, 6 M")
    assert all(0x12000 <= ord(c) < 0x12100 for c in signs)
    assert effects.decode_message(signs) == "hello, 6 M"


def test_projection_matches_the_table_and_the_solver(mapping):
    radii, angles = mapping["radii"], mapping["angles"]
    for order in (0, 1):
        b, z = effects.project(mapping, radii[5], angles[20], order)
        assert b[0] == pytest.approx(mapping["tables"][order][0][5, 20])
        assert z[0] == pytest.approx(mapping["tables"][order][1][5, 20])
    # Between table nodes, close to a direct solve.
    r, a = (radii[8] + radii[9]) / 2, 1.0
    b, _ = effects.project(mapping, r, a, 0)
    assert b[0] == pytest.approx(bhm.solve_for_impact_parameter(r, 1.3, a, 1.0, 0), rel=0.03)
    b, _ = effects.project(mapping, [radii[0] * 0.9, radii[-1] * 1.1], [1.0, 1.0], 0)
    assert np.all(np.isnan(b))


def test_the_probe_slows_reddens_and_never_crosses(mapping):
    probe = effects.Probe(mapping, framed_radius=40.0, rng=np.random.default_rng(0))
    radii, shifts = [], []
    for _ in range(4000):
        probe.fall(0.5)
        radii.append(probe.r)
        shifts.append(probe.shift())
    assert np.all(np.diff(radii) <= 0)
    assert np.all(np.diff(shifts) >= -1e-9)
    assert radii[-1] > 2.0 and radii[-1] < 2.01
    assert shifts[-1] > 1 / effects.Probe.LOST_BELOW
    # Seen from far away it stalls: the last thousand M of time barely move it.
    assert radii[-2000] - radii[-1] < 0.01


def test_the_probe_image_is_the_solved_one(mapping):
    probe = effects.Probe(mapping, framed_radius=40.0, rng=np.random.default_rng(3))
    probe.fall(30.0)
    probe.locate(1.3)
    assert probe.images[0] == pytest.approx(
        bhm.solve_for_impact_parameter(probe.r, 1.3, probe.alpha, 1.0, 0))


def test_a_transmission_arrives_in_full(mapping):
    live = fake_live(mapping)
    fx = effects.Effects(seed=2, messages=["abc"], events=False)
    fx.launch_transmission(live)
    tr = fx.transmission
    for i in range(int((tr.FLIGHT + 3 * tr.GAP + tr.SWALLOW + 1) * 30)):
        fx.step(live, i / 30, 1 / 30)
    assert fx.transmission is None
    assert "abc" in fx.readout[0]


def test_the_warp_comes_back(mapping):
    w = effects.Warp()
    zooms, directions = [], []
    while not w.done:
        zooms.append(w.zoom())
        directions.append(w.direction())
        w.age += 1 / 30
    assert zooms[0] == 1.0 and min(zooms) == pytest.approx(0.38, abs=0.01)
    assert w.zoom() == 1.0 and w.direction() == 1.0 and not w.inverted()
    assert -1.0 in directions


def test_pixel_plot_blends_and_clips():
    img = np.zeros((20, 30, 3), np.uint8)
    canvas = effects.PixelCanvas(img, 10.0, 10.0, 15, 5)
    canvas.plot(np.array([0.0, 0.5, 1.2]), np.array([0.0, 0.5, 0.5]), (200, 100, 50),
                np.array([1.0, 0.5, 1.0]), size=3)
    assert tuple(img[0, 0]) == (200, 100, 50) and tuple(img[1, 1]) == (200, 100, 50)
    assert tuple(img[10, 15]) == (100, 50, 25)
    assert img[:, 25:].sum() == 0          # the off-picture mark drew nothing


def test_dot_plot_lights_dots_and_tints_the_cell():
    lit = np.zeros((8, 8), bool)
    colours = np.zeros((2, 4, 3), np.float32)
    backgrounds = np.zeros((2, 4, 3), np.float32)
    canvas = effects.DotCanvas(lit, colours, backgrounds, 10.0, 10.0, 4, 2)
    canvas.plot(np.array([1.0, 0.0]), np.array([1.0, 0.0]), (90, 90, 90), np.array([1.0, 0.1]))
    assert lit[7, 7] and not lit[0, 0]
    assert tuple(colours[1, 3]) == (90, 90, 90)
    assert colours[0, 0, 0] == pytest.approx(9.0)


def test_effects_paint_and_overlay_on_both_canvases(mapping):
    live = fake_live(mapping)
    fx = effects.Effects(seed=4, events=False)
    fx.hud = True
    fx.launch_probe(live)
    fx.launch_transmission(live)
    fx.launch_warp(live)
    for i in range(60):
        fx.step(live, i / 30, 1 / 30)
    pixels = effects.PixelCanvas(np.full((120, 400, 3), 14, np.uint8), 40.0, 20.0, 80, 24)
    fx.paint(pixels, live)
    text = fx.overlay(live, clear=True)
    assert "INCL" in text and "PROBE" in text
    dots = effects.DotCanvas(np.zeros((96, 160), bool), np.zeros((24, 80, 3), np.float32),
                             np.zeros((24, 80, 3), np.float32), 40.0, 20.0, 80, 24)
    fx.paint(dots, live)
    assert dots.lit.any()
    assert "INCL" in fx.overlay(live, clear=False)
