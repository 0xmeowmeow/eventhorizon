"""Record eventhorizon as video or stills, without a terminal.

The widget is driven exactly as it runs, but on a fixed clock: frame n is drawn
at n / fps seconds, so recordings are smooth whatever the machine is doing. The
pixel picture is taken from the frame the widget would have sent to the
terminal, and text it would have written over it - HUD readouts, cuneiform
signs, the probe's telemetry - is drawn on with real fonts at the same cells.

    python tools/capture.py plate.webp --preset "the 1979 plate" --at 4
    python tools/capture.py plate.mp4 --preset "the 1979 plate" --seconds 8
    python tools/capture.py probe.mp4 --preset ember --keys 0.5:! --seconds 14
    python tools/capture.py braille.webp --braille --at 4
    python tools/capture.py flat.webp --flat          # light in straight lines

--keys takes time:key pairs, comma separated, pressed at those times.
"""

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from luminet import cli, config, live as live_module, pixels, spin

CELL = (10, 22)
FONT_DIR = "/usr/share/fonts/truetype"
FONTS = {
    "text": [os.path.expanduser("~/.local/share/fonts/TerminessNerdFontMono-Regular.ttf"),
             f"{FONT_DIR}/dejavu/DejaVuSansMono.ttf"],
    "cuneiform": [f"{FONT_DIR}/noto/NotoSansCuneiform-Regular.ttf"],
    "segments": [f"{FONT_DIR}/noto/NotoSansSymbols2-Regular.ttf"],
}


def font(kind, size):
    for path in FONTS[kind]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_live(preset, cols, rows, extra):
    parser = cli.parser_for_spin()
    args = parser.parse_args(["--width", str(cols), "--height", str(rows), *extra])
    args.config, args.explicit = {}, cli.explicit_options(parser, extra)
    args.status, args.events, args.pixels_mode = False, False, "on"
    args.presets = [dict(p) for p in config.STARTERS]
    names = [p["name"] for p in args.presets]
    args.preset_index = names.index(preset) if preset else None
    settings = {**cli.DEFAULTS}
    return live_module.Live(settings, args)


class Recorder:
    def __init__(self, a):
        self.a = a
        self.captured = None
        recorder = self

        def escape(transport, img, cols, rows):
            recorder.captured = np.array(img, copy=True)
            return ""

        pixels.Transport.escape = escape
        # A kitty or Ghostty cell at the size the widget was tuned in.
        live_module.cell_size = lambda: (tuple(map(float, CELL)), True)
        extra = ["--no-pixels"] if a.braille else []
        with contextlib.redirect_stdout(io.StringIO()):
            self.live = make_live(a.preset, a.cols, a.rows, extra)
            self.live.pixels_on = not a.braille
            self.live.transport_mode = "file"
            self.live.focused = True
            self.live.solve()
            self.live.fit()
        self.keys = sorted((float(t), k) for t, k in
                           (item.split(":", 1) for item in a.keys.split(",") if item))

    def frame(self, now, dt):
        lv = self.live
        while self.keys and self.keys[0][0] <= now:
            with contextlib.redirect_stdout(io.StringIO()):
                lv.handle(self.keys.pop(0)[1])
        lv.clock += dt * lv.effects.time_direction()
        with contextlib.redirect_stdout(io.StringIO()):
            lv.advance(now, dt)
            if lv.rebuilding is not None:
                lv.rebuilding.join()           # offline, wait rather than skip
                lv.advance(now, 0.0)
            lv.effects.step(lv, now, dt)
            if lv.resized:
                lv.fit()
            self.captured = None
            pane = lv.draw()
        img = self.captured if self.captured is not None else braille_image(pane, lv)
        text = lv.effects.overlay(lv, clear=False) if (
            lv.effects.active() or lv.effects.readout[0]) else ""
        return draw_text(img, text, lv)


# ------------------------------------------------------------------ drawing text

SGR = re.compile(r"\x1b\[([0-9;]*)m")
MOVE = re.compile(r"\x1b\[(\d+);(\d+)H")


def glyph_font(ch, size):
    code = ord(ch)
    if 0x12000 <= code < 0x12500:
        return font("cuneiform", size)
    if 0x1FB00 <= code < 0x1FC00:
        return font("segments", size)
    return font("text", size)


def draw_runs(draw, x, y, text, fg, sx, sy):
    size = int(round(19 * sy))
    for i, ch in enumerate(text):
        f = glyph_font(ch, size)
        draw.text((x + i * CELL[0] * sx, y + 1 * sy), ch, font=f, fill=fg)


def draw_text(img, text, lv):
    if not text:
        return img
    im = Image.fromarray(img)
    sx = im.width / (lv.cols * CELL[0])
    sy = im.height / (lv.rows * CELL[1])
    draw = ImageDraw.Draw(im)
    pos, fg, bg = (0, 0), (255, 255, 255), None
    i = 0
    while i < len(text):
        m = MOVE.match(text, i)
        if m:
            pos = (int(m.group(2)) - 1, int(m.group(1)) - 1)
            i = m.end()
            continue
        m = SGR.match(text, i)
        if m:
            parts = [int(p) for p in m.group(1).split(";") if p]
            if parts[:2] == [38, 2]:
                fg = tuple(parts[2:5])
            elif parts[:2] == [48, 2]:
                bg = tuple(parts[2:5])
            elif not parts or parts == [0]:
                fg, bg = (255, 255, 255), None
            i = m.end()
            continue
        j = i
        while j < len(text) and text[j] != "\x1b":
            j += 1
        run = text[i:j]
        x, y = pos[0] * CELL[0] * sx, pos[1] * CELL[1] * sy
        if bg is not None:
            draw.rectangle([x, y, x + len(run) * CELL[0] * sx, y + CELL[1] * sy], fill=bg)
        draw_runs(draw, x, y, run, fg, sx, sy)
        pos = (pos[0] + len(run), pos[1])
        i = j
    return np.asarray(im)


def braille_image(pane, lv):
    """The braille picture as the terminal draws it, glyph by glyph."""
    im = Image.new("RGB", (lv.cols * CELL[0], lv.rows * CELL[1]), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    f = font("text", 19)
    for r, line in enumerate(pane.split("\n")[:lv.rows]):
        fg, bg, col, i = (255, 255, 255), (0, 0, 0), 0, 0
        while i < len(line):
            m = SGR.match(line, i)
            if m:
                parts = [int(p) for p in m.group(1).split(";") if p]
                k = 0
                while k < len(parts):
                    if parts[k] in (38, 48) and parts[k + 1] == 2:
                        colour = tuple(parts[k + 2:k + 5])
                        if parts[k] == 38:
                            fg = colour
                        else:
                            bg = colour
                        k += 5
                    else:
                        k += 1
                i = m.end()
                continue
            x, y = col * CELL[0], r * CELL[1]
            draw.rectangle([x, y, x + CELL[0] - 1, y + CELL[1] - 1], fill=bg)
            if line[i] != "⠀":
                draw.text((x, y + 1), line[i], font=f, fill=fg)
            col += 1
            i += 1
    return np.asarray(im)


# ------------------------------------------------------------------- flat

def flat_image(a):
    """The same disk with light travelling in straight lines, for comparison.

    Same inclination, framing and dot rule as the plate, and the same intrinsic
    brightness of the disk, but no bending and no Doppler or gravitational
    shift: each parcel lands where a straight line from it meets the screen, and
    the horizon hides only what is directly behind it.
    """
    rec = Recorder(a)
    lv = rec.live
    view = lv.pixel_view
    incl, mass = float(lv.settings["incl"]), float(lv.settings["mass"])
    h, w = view.px_h, view.px_w
    rng = np.random.default_rng(1)
    radii = lv.mapping["radii"]
    n = 900_000
    r = np.exp(rng.uniform(np.log(radii[0]), np.log(radii[-1]), n))
    alpha = rng.uniform(0, 2 * np.pi, n)
    rr = r / mass
    sr, s3, s6 = np.sqrt(rr), np.sqrt(3.0), np.sqrt(6.0)
    flux = (3 * mass / (8 * np.pi) / ((rr - 3) * rr ** 2.5)
            * (sr - s6 + s3 / 2 * np.log((sr + s3) * (s6 - s3) / ((sr - s3) * (s6 + s3)))))
    x, y = r * np.sin(alpha), -r * np.cos(alpha) * np.cos(incl)
    # The screen maps like the widget's: row from -y.
    xf = (x / view.ext_x + 1) / 2
    yf = (-y / view.ext_y + 1) / 2
    behind = (np.cos(alpha) < 0) & (np.hypot(x, y) < 2 * mass)
    keep = (xf >= 0) & (xf < 1) & (yf >= 0) & (yf < 1) & ~behind
    # As in the widget, a patch's brightness is the mean flux of the gas seen
    # there - surface brightness - however densely the parcels happen to fall.
    gw, gh = view.grid_w, view.grid_h
    gi = (yf[keep] * gh).astype(int) * gw + (xf[keep] * gw).astype(int)
    total = np.bincount(gi, weights=flux[keep], minlength=gw * gh)
    count = np.bincount(gi, minlength=gw * gh)
    mean = total / np.maximum(count, 1)
    target = np.clip(mean / np.percentile(mean[count > 0], 99.3), 0, 1) ** 0.6
    target = np.where(count > 0, 0.012 + 0.988 * target, 0)
    expected = 2.5 * (1 - (1 - target) ** (1 / 2.5))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = live_module.PAPER
    grain = w // gw
    lots = rng.random(gw * gh)
    dots = np.floor(expected).astype(int) + (lots < expected % 1)
    cells = np.repeat(np.arange(gw * gh), dots)
    cx = (cells % gw) * grain + rng.integers(0, grain, cells.size)
    cy = (cells // gw) * grain + rng.integers(0, grain, cells.size)
    for dy in range(2):
        for dx in range(2):
            img[np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1)] = live_module.INK
    # The horizon itself, a small black disk, as seen without lensing.
    yy, xx = np.mgrid[0:h, 0:w]
    px = ((xx / (w - 1)) * 2 - 1) * view.ext_x
    py = ((yy / (h - 1)) * 2 - 1) * view.ext_y
    img[np.hypot(px, py) < 2 * mass] = live_module.HOLE
    return img


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--preset", default="the 1979 plate")
    p.add_argument("--cols", type=int, default=190)
    p.add_argument("--rows", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--at", type=float, default=3.0, help="stills: the moment to take")
    p.add_argument("--keys", default="", help="time:key pairs, e.g. 0.5:!,3:#")
    p.add_argument("--width", type=int, default=1280, help="output width in pixels")
    p.add_argument("--crf", type=int, default=24)
    p.add_argument("--crop", type=float, default=1.0,
                   help="keep this fraction of the width and height, centred on the hole")
    p.add_argument("--braille", action="store_true")
    p.add_argument("--flat", action="store_true")
    a = p.parse_args()

    still = a.out.endswith((".png", ".webp", ".jpg"))
    if a.flat:
        img = flat_image(a)
        save_still(img, a)
        return
    rec = Recorder(a)
    dt = 1.0 / a.fps
    if still:
        img = None
        for n in range(int(a.at * a.fps) + 1):
            img = rec.frame(n * dt, dt)
        save_still(crop(img, a.crop), a)
        return

    first = crop(rec.frame(0.0, dt), a.crop)
    h, w = first.shape[:2]
    out_h = int(round(a.width * h / w / 2)) * 2
    codec = (["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(a.crf + 8), "-row-mt", "1"]
             if a.out.endswith(".webm") else
             ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
              "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
         "-vf", f"scale={a.width}:{out_h}:flags=area", *codec, a.out],
        stdin=subprocess.PIPE)
    ff.stdin.write(np.ascontiguousarray(first).tobytes())
    for n in range(1, int(a.seconds * a.fps)):
        img = crop(rec.frame(n * dt, dt), a.crop)
        if img.shape[:2] != (h, w):
            img = np.asarray(Image.fromarray(img).resize((w, h)))
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB", file=sys.stderr)


def crop(img, keep):
    if keep >= 1.0:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * keep) // 2 * 2, int(w * keep) // 2 * 2
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw]


def save_still(img, a):
    im = Image.fromarray(img)
    if a.width and im.width != a.width:
        im = im.resize((a.width, int(round(im.height * a.width / im.width))), Image.LANCZOS)
    kwargs = {"quality": 88, "method": 6} if a.out.endswith(".webp") else {}
    im.save(a.out, **kwargs)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e3:.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
