import base64
import re
import zlib

import numpy as np
import pytest

from luminet import fast, pixels, spin
from luminet.live import keys_in

INK, PAPER, HOLE = (238, 230, 210), (14, 16, 13), (0, 0, 0)


@pytest.fixture(scope="module")
def view():
    settings = {"mass": 1.0, "incl": 1.4, "acc": 1.0, "outer_edge": 80.0}
    mapping = spin.lensing_map(settings, n_rings=40, n_angles=90)
    rx, ry = spin.reach(mapping, max_radius=30)
    rates = spin.true_rates(mapping["radii"], 1.0)
    projector = fast.Projector(mapping, (0, 1)) if fast.available else None
    v = pixels.PixelView(mapping, (rx * 0.8, ry * 0.95), rates, projector, (0, 1),
                         60, 20, (10.0, 22.0), pixels.Transport("direct"))
    yield v, rates
    v.transport.close()


def unpack(escape):
    parts = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\", escape)
    keys = dict(kv.split("=") for kv in parts[0][0].split(","))
    raw = zlib.decompress(base64.standard_b64decode("".join(p for _, p in parts)))
    return np.frombuffer(raw, np.uint8).reshape(int(keys["v"]), int(keys["s"]), 3), keys


def test_direct_escape_carries_the_frame(view):
    v, rates = view
    v.restyle("ink", "match", 0.0, True, False, False, INK, PAPER, HOLE)
    img, keys = unpack(v.frame(1.0, rates))
    assert img.shape == (v.px_h, v.px_w, 3)
    assert keys["a"] == "T" and keys["q"] == "2" and keys["c"] == "60" and keys["r"] == "20"
    assert (img == INK).all(axis=2).any()


def test_mask_is_round_in_pixels(view):
    """The shadow mask is a circle at the display's own resolution."""
    v, rates = view
    v.restyle("ink", "match", 0.0, True, False, False, INK, PAPER, HOLE)
    black = (v.background == HOLE).all(axis=2)
    ys, xs = np.nonzero(black)
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert height / width == pytest.approx(1.0, abs=0.06)


@pytest.mark.skipif(not fast.available, reason="numba is not installed")
def test_compiled_and_numpy_painters_agree(view):
    """
    The same pixels are painted either way. With a palette, dots from two cells
    can land on one pixel, and which colour wins then depends on paint order,
    which the two painters do not share; nowhere else may the colour differ.
    """
    v, rates = view
    for palette in ("ink", "inferno"):
        v.restyle(palette, "match", 0.0, False, False, False, INK, PAPER, HOLE)
        for t in (0.0, 3.0):
            a = v.background.copy()
            v.projector.paint(v.parcels, t, rates, v.ext_x, v.ext_y, v.grid_w, v.grid_h,
                              v.field.chance, v.cell_rgb, a, v.dot)
            b = v.background.copy()
            v._paint_numpy(b, t, rates)
            painted_a = (a != v.background).any(axis=2)
            painted_b = (b != v.background).any(axis=2)
            assert np.array_equal(painted_a, painted_b)
            if palette == "ink":
                assert np.array_equal(a, b)
            else:
                assert (a != b).any(axis=2).mean() < 0.002


def test_graphics_replies_are_not_keys():
    reply = "\x1b_Gi=31;OK\x1b\\"
    assert list(keys_in("a" + reply + "b")) == ["a", "b"]
    assert list(keys_in("\x1b]11;rgb:0000/0000/0000\x07q")) == ["q"]
