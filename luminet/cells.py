"""Draw with what the terminal actually has, rather than with ASCII.

A character cell in kitty and Ghostty measures 10 by 22 pixels, so one sample
per cell is nearly twice as tall as it is wide. The fix is to use the glyphs
that subdivide a cell, and the choice between them is decided by colour rather
than by resolution: a cell carries exactly one foreground and one background
colour, whatever glyph is in it.

    half-block  1x2   10x11 subpixel, aspect 0.91   two independent colours
    sextant     2x3   5x7.3 subpixel, aspect 0.68   two colours in total
    octant      2x4   5x5.5 subpixel, aspect 0.91   two colours in total

So half-block is right for a full-colour image, since each of its two subpixels
can be any colour. Sextants win where the picture is locally two-tone, which
line art is, and trade colour for spatial detail. Octants are better still but
are Unicode 16, which Python's own tables do not yet name, so they are left for
later rather than hardcoded from memory.

Both terminals were confirmed to render every family by `tuilab preflight` in
the tui-research project, which draws a contact sheet of each one.
"""

import unicodedata

import numpy as np

UPPER_HALF = "▀"     # the top half lit; foreground is the top subpixel
FULL = "█"
LEFT_HALF = "▌"
RIGHT_HALF = "▐"
RESET = "\033[0m"


def _sextant_table():
    """Map a 6-bit pattern to its glyph, read out of the Unicode names.

    Bits run left to right, top to bottom:  1 2 / 3 4 / 5 6, bit 0 being 1.
    Four of the 64 patterns are older characters and are not in the sextant
    block, so they are filled in directly.
    """
    table = {
        0b000000: " ",
        0b111111: FULL,
        0b010101: LEFT_HALF,      # 1, 3, 5
        0b101010: RIGHT_HALF,     # 2, 4, 6
    }
    for cp in range(0x1FB00, 0x1FB40):
        try:
            name = unicodedata.name(chr(cp))
        except ValueError:
            continue
        if not name.startswith("BLOCK SEXTANT-"):
            continue
        bits = 0
        for digit in name.rsplit("-", 1)[1]:
            bits |= 1 << (int(digit) - 1)
        table[bits] = chr(cp)
    return table


SEXTANTS = _sextant_table()


def _rgb(r, g, b, background=False):
    return f"\033[{48 if background else 38};2;{int(r)};{int(g)};{int(b)}m"


def half_block(rgb):
    """Render an (H, W, 3) image, H even, two rows to a cell.

    The glyph is always an upper half block, so the foreground paints the top
    subpixel and the background the bottom one.
    """
    rgb = np.asarray(rgb, dtype=np.uint32)
    height = rgb.shape[0] - rgb.shape[0] % 2
    top, bottom = rgb[0:height:2], rgb[1:height:2]

    # Pack each colour into one integer so runs of identical cells can be found
    # by numpy rather than by comparing tuples in Python. Most of a frame is
    # unchanged from its neighbour, so emitting one escape pair per run instead
    # of per cell cuts both the work and the bytes sent by a large factor.
    tkey = (top[..., 0] << 16) | (top[..., 1] << 8) | top[..., 2]
    bkey = (bottom[..., 0] << 16) | (bottom[..., 1] << 8) | bottom[..., 2]

    lines = []
    for row in range(tkey.shape[0]):
        tr, br = tkey[row], bkey[row]
        starts = np.flatnonzero(
            np.r_[True, (tr[1:] != tr[:-1]) | (br[1:] != br[:-1])]
        )
        ends = np.r_[starts[1:], len(tr)]
        out = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            t, b = int(tr[start]), int(br[start])
            out.append(f"\033[38;2;{t >> 16};{(t >> 8) & 255};{t & 255}"
                       f";48;2;{b >> 16};{(b >> 8) & 255};{b & 255}m")
            out.append(UPPER_HALF * (end - start))
        lines.append("".join(out) + RESET)
    return "\n".join(lines)


def sextant(mask, foreground=(255, 255, 255), background=(0, 0, 0)):
    """Render a two-tone (H, W) boolean mask at 2x3 per cell."""
    mask = np.asarray(mask, dtype=bool)
    height = mask.shape[0] - mask.shape[0] % 3
    width = mask.shape[1] - mask.shape[1] % 2

    head = _rgb(*foreground) + _rgb(*background, background=True)
    lines = []
    for row in range(0, height, 3):
        out = [head]
        for col in range(0, width, 2):
            bits = 0
            for dy in range(3):
                for dx in range(2):
                    if mask[row + dy, col + dx]:
                        bits |= 1 << (dy * 2 + dx)
            out.append(SEXTANTS[bits])
        lines.append("".join(out) + RESET)
    return "\n".join(lines)


# Braille packs 2x4 dots into one cell: the finest grid the terminal offers
# without leaving text behind. One dot is about 5x5.5px against a half block's
# 10x11, so a single dot makes a convincingly small star. The cost is colour,
# since a cell still carries only one foreground.
BRAILLE_BITS = np.array([[0x01, 0x08],
                         [0x02, 0x10],
                         [0x04, 0x20],
                         [0x40, 0x80]], dtype=np.uint8)


def braille(mask, foreground=(235, 225, 205), background=(0, 0, 0)):
    """Render a two-tone (H, W) mask at 2x4 dots per cell."""
    mask = np.asarray(mask, dtype=bool)
    height = mask.shape[0] - mask.shape[0] % 4
    width = mask.shape[1] - mask.shape[1] % 2
    block = mask[:height, :width].reshape(height // 4, 4, width // 2, 2)
    bits = (block * BRAILLE_BITS[None, :, None, :]).sum(axis=(1, 3)).astype(np.uint16)

    head = _rgb(*foreground) + _rgb(*background, background=True)
    lines = []
    for row in bits:
        lines.append(head + "".join(chr(0x2800 + int(v)) for v in row) + RESET)
    return "\n".join(lines)


def stipple(values, strength=1.0):
    """Turn a brightness field into dots, denser where it is brighter.

    Luminet's 1979 figure was computed on an IBM 7040 and inked dot by dot onto
    negative paper, so its tone is carried by how close together the dots are
    rather than by how bright each one is. An ordered threshold reproduces that
    and, unlike random dithering, holds still between frames instead of boiling.
    """
    h, w = values.shape
    tile = np.tile(BAYER8 + 0.5, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return values * strength > tile


# ------------------------------------------------------------------- colouring

def downsample(rgb, factor):
    """Average an image down, which is what makes the terminal copy look sharp.

    Rendering above the cell grid and averaging back gives the half blocks
    anti-aliased edges, rather than the hard steps of sampling once per cell.
    """
    if factor <= 1:
        return rgb
    h = rgb.shape[0] // factor * factor
    w = rgb.shape[1] // factor * factor
    block = rgb[:h, :w].reshape(h // factor, factor, w // factor, factor, 3)
    return block.mean(axis=(1, 3)).astype(np.uint8)


def _ramp(t, stops):
    """Linear interpolation through a list of RGB stops."""
    t = np.clip(t, 0.0, 1.0)
    positions = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(t, positions, [s[i] for s in stops]) for i in range(3)], axis=-1)


TINT = 0.55   # how far the redshift is allowed to colour the flux image

# Colour is about 5% of a frame's cost, against a lensing map that is solved
# once, so the palette is free to be anything. These are ramps from the shared
# terminal aesthetics taxonomy; any of matplotlib's 182 colormaps works too,
# by name.
PALETTES = {
    "ember": [(0, 0, 0), (70, 40, 20), (190, 120, 45), (255, 220, 150), (255, 255, 255)],
    "phosphor": [(0, 0, 0), (0, 35, 12), (0, 150, 60), (130, 255, 175), (225, 255, 240)],
    "amber": [(0, 0, 0), (40, 18, 0), (170, 85, 0), (255, 185, 55), (255, 240, 200)],
    "ice": [(0, 0, 0), (8, 28, 60), (40, 110, 180), (150, 210, 245), (255, 255, 255)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "mono": [(0, 0, 0), (255, 255, 255)],
    "bw": [(0, 0, 0), (0, 0, 0), (255, 255, 255), (255, 255, 255)],
    "redshift": [(70, 130, 255), (170, 205, 255), (245, 245, 245), (255, 170, 140), (230, 60, 40)],
}


def palette(name):
    """Colour stops by name: one of ours, or any matplotlib colormap."""
    if not name:
        return PALETTES["ember"]
    if name in PALETTES:
        return PALETTES[name]
    import matplotlib

    if name in matplotlib.colormaps:
        sampled = matplotlib.colormaps[name](np.linspace(0, 1, 32))
        return [tuple(int(v * 255) for v in row[:3]) for row in sampled]
    known = ", ".join(sorted(PALETTES))
    raise ValueError(f"Unknown palette {name!r}. Ours: {known}. "
                     f"Any matplotlib colormap name also works, e.g. inferno.")


def feather(rgb, mask, amount=1.0):
    """Soften the silhouette, so the disk does not end in a staircase.

    The disk's outer edge really is an edge - the material stops - but it should
    land on a cell as a fraction of coverage, not as on or off. Sampled once per
    cell it is all or nothing, and a curve crossing the grid at a shallow angle
    turns into steps.

    Blurring the coverage rather than the picture fixes the outline without
    touching anything inside it: the interior stays saturated because the blurred
    mask is pushed back above one there, and only the boundary keeps a partial
    value to fade through.
    """
    if amount <= 0:
        return rgb
    from scipy.ndimage import gaussian_filter

    coverage = gaussian_filter(mask.astype(np.float32), sigma=0.9 * amount)
    # Lift the interior back to solid; only the rim is left part-covered.
    alpha = np.clip(coverage * 1.9, 0.0, 1.0)
    return (rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)


def starfield(width, height, density=0.012, seed=11):
    """A fixed field of background stars.

    These are decoration, not physics: real starlight this close to a black hole
    is lensed into arcs and an Einstein ring, and none of that is modelled here.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((height, width), dtype=np.float32)
    count = max(1, int(width * height * density))
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    # Mostly faint, a few bright: an even spread reads as noise.
    field[ys, xs] = rng.random(count) ** 4.0
    return field


def add_stars(rgb, field, mask, brightness=190.0, twinkle=None):
    """Put the stars behind the disk."""
    if field is None:
        return rgb
    lit = field * brightness
    if twinkle is not None:
        lit = lit * twinkle
    behind = ~mask
    out = rgb.astype(np.float32)
    out[behind] += lit[behind][..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 64.0 - 0.5


def dither(values, strength=1.0):
    """Ordered dither, to break up banding in a short ramp."""
    if strength <= 0:
        return values
    h, w = values.shape
    tile = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return np.clip(values + tile * strength, 0.0, 1.0)


def scanlines(rgb, strength=0.25):
    """Darken alternate rows."""
    if strength <= 0:
        return rgb
    out = rgb.astype(np.float32)
    out[1::2] *= (1.0 - strength)
    return out.astype(np.uint8)


def vignette(rgb, strength=0.35):
    """Darken towards the corners."""
    if strength <= 0:
        return rgb
    h, w = rgb.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    fade = 1.0 - strength * np.clip((x * x + y * y) / 2.0, 0, 1)
    return (rgb.astype(np.float32) * fade[..., None]).astype(np.uint8)


def bloom(rgb, amount=0.6, radius=2.0):
    """Let the bright parts spill into their surroundings.

    Cheap, and it suits something whose whole subject is light: the lensed arc
    reads as glare rather than as a hard-edged shape.
    """
    if amount <= 0:
        return rgb
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(rgb.astype(np.float32), sigma=(radius, radius, 0))
    return np.clip(rgb.astype(np.float32) + blurred * amount, 0, 255).astype(np.uint8)


FLUX_STOPS = [(0, 0, 0), (70, 40, 20), (190, 120, 45), (255, 220, 150), (255, 255, 255)]
REDSHIFT_STOPS = [(70, 130, 255), (170, 205, 255), (245, 245, 245), (255, 170, 140), (230, 60, 40)]


def colourise(grid, channel="both", gamma=1.0, name=None):
    """Turn a rasterised field into an (H, W, 3) image.

    'both' is the one worth looking at: brightness carries the flux, the thing a
    camera would record, while hue carries the redshift, which the rendered
    image throws away entirely.
    """
    from luminet import fields

    stops = palette(name)
    shape = grid["flux"].shape
    out = np.zeros((*shape, 3))

    if channel == "flux":
        t = fields.normalise(grid, "flux", gamma=gamma)
        out = _ramp(np.nan_to_num(t), stops)
    elif channel == "redshift":
        t = fields.normalise(grid, "z")
        out = _ramp(np.nan_to_num(t, nan=0.5), REDSHIFT_STOPS)
    elif channel == "radius":
        t = fields.normalise(grid, "radius")
        out = _ramp(np.nan_to_num(t), [(20, 20, 60), (60, 160, 200), (230, 250, 210)])
    elif channel == "order":
        ghost = np.nan_to_num(grid["order"], nan=0.0) > 0.5
        out[...] = np.where(ghost[..., None], np.array([255, 140, 60]),
                            np.array([120, 170, 255]))
    else:  # both
        # An earlier version lifted the whole image to make the hue readable and
        # flattened it into grey soup; the flux view beat it easily. Contrast is
        # what makes the picture read, so keep the flux ramp exactly as it is
        # and only tint it, which costs saturation rather than dynamic range.
        base = _ramp(np.nan_to_num(fields.normalise(grid, "flux", gamma=gamma)), stops)
        tint = _ramp(np.nan_to_num(fields.normalise(grid, "z"), nan=0.5), REDSHIFT_STOPS)
        luminance = base.mean(axis=-1, keepdims=True)
        tint = tint / np.maximum(tint.max(axis=-1, keepdims=True), 1.0)
        out = base * (1 - TINT) + tint * luminance * TINT

    out[~grid["mask"]] = 0
    return np.clip(out, 0, 255).astype(np.uint8)
