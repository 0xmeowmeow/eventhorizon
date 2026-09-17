"""The live instrument: runs forever, reshapes itself, and takes instructions.

This is meant to sit in a corner of somebody's desktop, so it has to behave like
furniture. It never repeats, it survives the window being resized, and every
setting can be changed while it runs.

The split that makes it possible: the lensing map is expensive and fixed, the
picture is cheap. Anything that changes only how the light is shown - palette,
bloom, stars, scanlines, how fast the gas turns - is applied to the next frame
immediately. Anything that changes the gravity - inclination, mass, the extent
of the disk - needs the map solved again, so those move by a step at a time and
show a bar while they go.
"""

import os
import select
import shutil
import signal
import sys
import termios
import time
import tty
from collections import deque

import numpy as np

from luminet import cells, spin

# How each encoding subdivides a cell, and whether it can carry colour.
ENCODINGS = {
    "half":    {"x": 1, "y": 2, "colour": True},
    "sextant": {"x": 2, "y": 3, "colour": False},
    "braille": {"x": 2, "y": 4, "colour": False},
    "plot1979": {"x": 2, "y": 4, "colour": False},
}
ENCODING_CYCLE = ["half", "sextant", "braille", "plot1979"]

# Luminet inked his 1979 figure dot by dot on negative paper; this is that.
INK = (238, 230, 210)
PAPER = (14, 16, 13)     # the print is not quite black
LINE = (150, 205, 235)   # isoradials drawn over the dots
HOLE = (0, 0, 0)         # the shadow, darker than the paper
GLOW_CYCLE = ["match"]   # the glow's own palette; "match" follows the dots
OVERLAYS = ["off", "lines", "flowing"]

PALETTE_CYCLE = ["ink", "ember", "inferno", "magma", "amber", "phosphor", "ice",
                 "plasma", "cividis", "bone", "copper", "gameboy", "bw"]
GLOW_CYCLE += PALETTE_CYCLE

HELP = [
    ("q  esc", "quit"),
    ("space", "pause"),
    ("p  P", "palette forward / back"),
    ("b  B", "bloom less / more"),
    ("c  C", "fewer / more arms in the gas"),
    ("s  S", "slower / faster"),
    ("g", "stars"),
    ("d", "dither"),
    ("n", "scanlines"),
    ("v", "vignette"),
    ("f", "soften the edge"),
    ("a", "antialiasing (costs frame rate)"),
    ("r", "re-seed the gas"),
    ("[  ]", "inclination  (re-solves)"),
    ("-  =", "disk size  (re-solves)"),
    (",  .", "mass  (re-solves)"),
    ("e  E", "encoding: half, sextant, braille, 1979 plot"),
    ("i", "1979: isoradials off, drawn, flowing"),
    ("k  K", "1979: glow palette, separate from the dots"),
    ("m", "1979: keep the glow out of the shadow"),
    ("o", "1979: hide or show the dots"),
    ("y", "cycle everything, hands off"),
    ("h  ?", "this list"),
]


def keys_in(text):
    """The keypresses in a chunk of terminal input, with escape sequences removed.

    Arrow and function keys arrive as ESC followed by [ or O, parameters, and a
    final byte from @ to ~. Those are dropped whole. A lone ESC is kept, since
    on its own it is the escape key.
    """
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b" and i + 1 < len(text) and text[i + 1] in "[O":
            j = i + 2
            while j < len(text) and not ("@" <= text[j] <= "~"):
                j += 1
            i = j + 1
            continue
        yield ch
        i += 1


class Live:
    def __init__(self, settings, opts):
        self.settings = dict(settings)
        self.o = opts
        self.palette_at = (PALETTE_CYCLE.index(opts.palette)
                           if opts.palette in PALETTE_CYCLE else 0)
        self.bloom = opts.bloom
        self.clumps = opts.clumps
        self.speed = opts.speed
        self.stars_on = not opts.no_stars
        self.dither_on = opts.dither
        self.scanlines_on = opts.scanlines
        self.vignette_on = opts.vignette
        self.feather_on = True
        self.encoding = opts.encoding
        self.overlay = 0
        self.dots_on = True
        self.glow_at = 0
        self.mask_on = True
        self.note = ("", 0.0)
        self.cycling = opts.cycle
        self.next_change = 0.0
        self.ss = max(1, opts.supersample)
        self.paused = False
        self.show_help = False
        self.seed = opts.seed
        self.clock = 0.0
        self.mapping = None
        self.cols = self.rows = 0
        self.resized = True
        signal.signal(signal.SIGWINCH, self._winch)

    # ------------------------------------------------------------ housekeeping

    def _winch(self, *_):
        self.resized = True

    def fit(self):
        """Match the window. Called at the start and whenever it changes."""
        size = shutil.get_terminal_size((90, 28))
        self.cols = self.o.width or max(24, size.columns - 1)
        self.rows = self.o.height or max(8, size.lines - 2)
        sub = ENCODINGS[self.encoding]
        self.w, self.h = self.cols * sub["x"], self.rows * sub["y"]
        # Only half-block needs supersampling: the others already sample well
        # below a cell, which is what smooths their edges.
        use_ss = self.ss if self.encoding == "half" else 1
        self.rw, self.rh = self.w * use_ss, self.h * use_ss
        self.use_ss = use_ss

        # Keep the gas at a constant density whatever the window is doing.
        count = int(self.w * self.h * self.o.density * (2 if self.use_ss > 1 else 1))
        self.parcels = spin.Parcels(self.mapping["radii"], count=count, seed=self.seed,
                                    infall=self.o.infall, clumps=self.clumps,
                                    depth=self.o.depth)
        # Stars are made at the size they are shown. Generated on the
        # supersampled grid they would be averaged down into grey smudges
        # instead of staying points.
        # Two copies: the dot encodings can place a star on a single 5x5px dot,
        # which is far smaller than half-block's 10x11 subpixel.
        self.stars = cells.starfield(self.w, self.h, self.o.star_density, self.seed + 5)
        self.stars_shown = cells.starfield(self.cols, self.rows * 2,
                                           self.o.star_density, self.seed + 5)
        if self.encoding == "plot1979":
            # No tracer: the dots are the material, so the motion shows without
            # one. Many parcels, because most of them are faint and thinly kept.
            self.dust = spin.Parcels(self.mapping["radii"], count=int(self.w * self.h * 2.5),
                                     seed=self.seed, infall=self.o.infall, clumps=0,
                                     spread="log")
            # How bright each cell should be is fixed for a given map and
            # window, so it is measured here once rather than every frame.
            orders = (0,) if self.o.no_ghost else (0, 1)
            self.field = spin.DotField(self.mapping, self.dust, self.w, self.h, self.extent,
                                       self.rates, orders, gamma=self.o.ink_gamma,
                                       projector=self.projector)
            self.isolines = spin.Isolines(self.mapping, self.w, self.h,
                                          self.field.ext_x, self.field.ext_y)
            self.hole = self.shadow_coverage()
        self.resized = False
        # A new grid or a new map makes the old timings about something else,
        # and a re-solve pause would drag the rate down for a second.
        for window in (getattr(self, "shown", None), getattr(self, "costs", None)):
            if window is not None:
                window.clear()
        sys.stdout.write("\033[2J\033[H")

    def physics(self):
        """The settings the map is solved for.

        plot1979 solves a much wider disk than it frames. In Luminet's figure the
        disk runs well past the edges of the picture, so there is faint light,
        and therefore a scattering of dots, over the whole frame - which is why
        the original has no stars and needs none. The framing still follows the
        disk-size setting, so - and = zoom as before.
        """
        if self.encoding != "plot1979":
            return self.settings
        wide = dict(self.settings)
        wide["outer_edge"] = max(160.0, self.settings["outer_edge"] * 4)
        return wide

    def solve(self, why=""):
        """Rebuild the lensing map, showing how far along it is."""
        label = why or "solving the lensing map"
        width = max(20, min(48, self.cols - len(label) - 14)) if self.cols else 30

        def bar(done, total):
            filled = int(width * done / total)
            sys.stdout.write(
                f"\r  {label}  [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
            sys.stdout.flush()

        sys.stdout.write("\033[2J\033[H\n")
        physics = self.physics()
        self.solved_for = self.encoding == "plot1979"
        rings = self.o.rings * 2 if self.solved_for else self.o.rings
        self.mapping = spin.lensing_map(
            physics, n_rings=rings, n_angles=self.o.angles,
            orders=(0,) if self.o.no_ghost else (0, 1),
            on_progress=bar, batches=6,
        )
        orders = (0,) if self.o.no_ghost else (0, 1)
        if self.solved_for:
            rx, ry = spin.reach(self.mapping, orders,
                                max_radius=self.settings["outer_edge"] * 0.85)
            self.extent = (rx * 0.78, ry * 0.95)
        else:
            self.extent = spin.reach(self.mapping, orders)
        self.rates = spin.true_rates(self.mapping["radii"],
                                     float(self.settings["mass"]), self.speed)
        from luminet import fast

        self.projector = (fast.Projector(self.mapping, orders)
                          if fast.available and not self.o.no_compile else None)
        sys.stdout.write("\r\033[2K")
        self.resized = True

    def rebuild_rates(self):
        self.rates = spin.true_rates(self.mapping["radii"],
                                     float(self.settings["mass"]), self.speed)

    # ---------------------------------------------------------------- drawing

    def draw(self):
        if self.encoding == "plot1979":
            return self.draw_1979()
        orders = (0,) if self.o.no_ghost else (0, 1)
        grid = spin.frame(self.mapping, self.parcels, self.clock, self.rw, self.rh,
                          self.extent, None, orders, rates=self.rates,
                          gas_spread=self.o.gas_spread)

        from luminet import fields

        value = np.nan_to_num(fields.normalise(grid, "flux", gamma=self.o.gamma))

        if not ENCODINGS[self.encoding]["colour"]:
            # One colour per cell, so the fine structure comes from which dots
            # are lit and the palette is applied cell by cell on top. Dots only
            # have two states, so the tone is lifted before thresholding: at the
            # picture's own gamma the dim outer disk lit almost nothing.
            lit = cells.stipple(value ** self.o.dot_gamma, 1.0) & grid["mask"]
            star_dots = None
            if self.stars_on:
                star_dots = (self.stars > self.o.star_cut) & ~grid["mask"]
                lit = lit | star_dots

            # plot1979 is monochrome on purpose: the original was one ink.
            colours = None
            if self.encoding != "plot1979":
                colours = self.cell_colours(value, grid["mask"], star_dots)

            if self.encoding == "sextant":
                return cells.sextant(lit, self.ink(), colours=colours)
            return cells.braille(lit, self.ink(), colours=colours)

        if self.dither_on:
            value = cells.dither(value, 0.04)
        img = cells._ramp(value, cells.palette(PALETTE_CYCLE[self.palette_at])).astype(np.uint8)
        img[~grid["mask"]] = 0

        if self.feather_on:
            img = cells.feather(img, grid["mask"], 1.0)
        if self.bloom > 0:
            img = cells.bloom(img, self.bloom)

        img = cells.downsample(img, self.use_ss)
        if self.stars_on:
            covered = grid["mask"]
            if self.use_ss > 1:
                covered = covered[:self.h * self.use_ss, :self.w * self.use_ss].reshape(
                    self.h, self.use_ss, self.w, self.use_ss).any(axis=(1, 3))
            img = cells.add_stars(img, self.stars_shown, covered, self.o.star_brightness)
        if self.scanlines_on:
            img = cells.scanlines(img, 0.22)
        if self.vignette_on:
            img = cells.vignette(img, 0.4)
        return cells.half_block(img)

    def draw_1979(self, grid=None):
        """After Luminet's figure: scattered cream dots on near-black.

        Each dot is a parcel of gas, kept with probability proportional to how
        bright its cell is. The dots therefore are the disk: they orbit,
        brighten into view on the approaching side, and fill the whole frame
        thinly, because the disk solved for this mode runs far past its edges.

        Isoradials can be drawn over them, and the look can take a glow, a
        vignette and scanlines. Those work on colours per cell, since each
        braille cell has one foreground and one background.
        """
        if self.dots_on:
            lit = self.field.frame(self.clock, self.rates).copy()
        else:
            lit = np.zeros((self.h, self.w), dtype=bool)
        line_cells = None
        if self.overlay:
            lines = self.isolines.frame(self.clock, self.rates,
                                        flowing=OVERLAYS[self.overlay] == "flowing")
            lit |= lines
            line_cells = lines[:self.rows * 4, :self.cols * 2].reshape(
                self.rows, 4, self.cols, 2).any(axis=(1, 3))

        name = PALETTE_CYCLE[self.palette_at]
        effects = (self.bloom > 0 or self.scanlines_on or self.vignette_on
                   or (self.mask_on and HOLE != PAPER))
        if name == "ink" and line_cells is None and not effects:
            return cells.braille(lit, INK, PAPER)

        rows, cols = self.rows, self.cols
        bright = self.field.target[:rows * 4, :cols * 2].reshape(rows, 4, cols, 2).max(axis=(1, 3))
        if name == "ink":
            colours = np.empty((rows, cols, 3), dtype=np.float32)
            colours[:] = INK
            glow_colour = np.array(INK, dtype=np.float32)
        else:
            # The spacing of the dots already carries the tone, so the colour is
            # lifted well off black: a faint cell given a dark colour would lose
            # the very dots that show it is faint.
            stops = cells.palette(name)
            tone = 0.4 + 0.6 * np.clip(bright, 0.0, 1.0) ** 0.5
            colours = cells._ramp(tone, stops).astype(np.float32)
            glow_colour = np.array(stops[len(stops) * 2 // 3], dtype=np.float32)
        if line_cells is not None:
            colours[line_cells] = LINE if self.dots_on else INK

        backgrounds = np.empty((rows, cols, 3), dtype=np.float32)
        backgrounds[:] = PAPER
        if self.bloom > 0:
            # Glow belongs in the gaps: a dot picture is mostly the space between
            # dots, so light spilling from the bright lobe shows as the paper
            # behind the dots warming, not as the dots themselves brightening.
            from scipy.ndimage import gaussian_filter

            spill = gaussian_filter(bright.astype(np.float32), sigma=(1.2, 2.4))
            glow_name = GLOW_CYCLE[self.glow_at]
            if glow_name == "match":
                tint = np.broadcast_to(glow_colour, (rows, cols, 3))
            else:
                # Its own palette, ramped by how much light arrives: faint spill
                # takes the dark end, the glow nearest the lobe the bright end.
                # That gives the glow a colour gradient of its own, independent
                # of the dots, so the two can contrast.
                reach = spill / max(float(spill.max()), 1e-6)
                tint = cells._ramp(reach, cells.palette(glow_name)).astype(np.float32)
            backgrounds += tint * (0.6 * self.bloom * spill)[..., None]
        if self.mask_on and getattr(self, "hole", None) is not None:
            # The shadow takes no light from anywhere, so the glow stops at its
            # edge. Only the glow: gas on the near side of the disk passes in
            # front of the hole, and its dots stay.
            cover = self.hole[..., None]
            backgrounds = backgrounds * (1.0 - cover) + np.array(HOLE, np.float32) * cover
        if self.vignette_on:
            y = np.linspace(-1, 1, rows)[:, None]
            x = np.linspace(-1, 1, cols)[None, :]
            fade = (1.0 - 0.45 * np.clip((x * x + y * y) / 2.0, 0, 1))[..., None]
            colours *= fade
            backgrounds *= fade
        if self.scanlines_on:
            colours[1::2] *= 0.72
            backgrounds[1::2] *= 0.72
        return cells.braille(lit, INK, PAPER,
                             colours=np.clip(colours, 0, 255).astype(np.uint8),
                             backgrounds=np.clip(backgrounds, 0, 255).astype(np.uint8))

    def shadow_coverage(self, sub=6):
        """How much of each cell lies inside the black hole's shadow, 0 to 1.

        For a non-rotating black hole the shadow on the observer's screen is an
        exact circle, radius 3 sqrt(3) M, whatever the inclination - so a round
        mask is also the physically right one. Backgrounds are coloured a whole
        cell at a time, and a cell is 10 by 22 pixels, so a hard mask would come
        out as a stepped circle. Each cell is sampled at several points instead
        and gets a fraction, which blends the edge.
        """
        radius = float(self.mapping["bh"].critical_b)
        cols, rows = self.cols, self.rows
        # Sample points spread over each cell's 2x4 dots, in dot coordinates.
        fx = (np.arange(sub) + 0.5) / sub * 2.0
        fy = (np.arange(sub * 2) + 0.5) / (sub * 2) * 4.0
        dot_x = np.arange(cols)[:, None] * 2.0 + fx[None, :]
        dot_y = np.arange(rows)[:, None] * 4.0 + fy[None, :]
        bx = (dot_x / (self.w - 1) * 2 - 1) * self.field.ext_x
        by = (dot_y / (self.h - 1) * 2 - 1) * self.field.ext_y
        inside = (by[:, None, :, None] ** 2 + bx[None, :, None, :] ** 2) < radius ** 2
        return inside.mean(axis=(2, 3)).astype(np.float32)

    def cell_colours(self, value, mask, star_dots):
        """A palette colour for each cell, from the light that falls in it.

        The dots already say how dense the light is, so the colour is lifted
        off black: a dim cell with a dark colour would leave its few dots
        invisible and throw away the structure the dots are there to show.
        """
        sub = ENCODINGS[self.encoding]
        sy, sx, rows, cols = sub["y"], sub["x"], self.rows, self.cols
        v = value[:rows * sy, :cols * sx].reshape(rows, sy, cols, sx)
        mk = mask[:rows * sy, :cols * sx].reshape(rows, sy, cols, sx)
        covered = mk.sum(axis=(1, 3))
        mean = (v * mk).sum(axis=(1, 3)) / np.maximum(covered, 1)

        tone = 0.35 + 0.65 * np.clip(mean, 0.0, 1.0) ** 0.6
        stops = cells.palette(PALETTE_CYCLE[self.palette_at])
        colours = cells._ramp(tone, stops).astype(np.uint8)

        if star_dots is not None:
            star = star_dots[:rows * sy, :cols * sx].reshape(rows, sy, cols, sx).any(axis=(1, 3))
            colours[star & (covered == 0)] = (220, 228, 245)
        return colours

    def ink(self):
        if self.encoding == "plot1979":
            return INK
        stops = cells.palette(PALETTE_CYCLE[self.palette_at])
        return tuple(int(v) for v in stops[-1])

    def status(self):
        s = self.settings
        bits = [f"incl {s['incl']:.2f}", f"mass {s['mass']:.2f}",
                f"disk {s['outer_edge']:.0f}", self.encoding,
                PALETTE_CYCLE[self.palette_at], f"{self.cols}x{self.rows}"]
        if self.cycling:
            bits.append("cycling")
        if self.encoding == "plot1979":
            bits.append("compiled" if self.projector is not None else "numpy")
            if self.overlay:
                bits.append(f"isoradials {OVERLAYS[self.overlay]}")
            if not self.dots_on:
                bits.append("dots hidden")
            if self.bloom > 0:
                bits.append(f"glow {GLOW_CYCLE[self.glow_at]} {self.bloom:.2f}")
            if not self.mask_on:
                bits.append("mask off")
        text, until = getattr(self, "note", ("", 0.0))
        if text and time.monotonic() < until:
            return text
        if self.paused:
            bits.append("PAUSED")
        shown = getattr(self, "shown", ())
        if len(shown) > 1:
            span = shown[-1] - shown[0]
            rate = (len(shown) - 1) / span if span > 0 else 0.0
            cost = 1000 * sum(self.costs) / len(self.costs)
            bits.append(f"{rate:4.1f}fps {cost:3.0f}ms/frame")
        return "  ".join(bits) + "   h for keys, q to quit"

    # ----------------------------------------------------------------- input

    def handle(self, key):
        """Act on one keypress. Returns False to stop."""
        o, s = self.o, self.settings

        if key in ("q", "\x1b", "\x03"):
            return False
        if self.encoding == "plot1979" and key in ("g", "d", "f", "a"):
            what = {"g": "stars", "d": "dither", "f": "edge softening",
                    "a": "antialiasing"}[key]
            self.note = (f"{what} is for half, sextant and braille; plot1979 has none",
                         time.monotonic() + 3.0)
            return True
        if self.encoding != "plot1979" and key in ("i", "o", "k", "K", "m"):
            self.note = ("isoradials, hiding dots, glow palette and mask are plot1979 only",
                         time.monotonic() + 3.0)
            return True
        elif key == " ":
            self.paused = not self.paused
        elif key in ("h", "?"):
            self.show_help = not self.show_help
        elif key == "p":
            self.palette_at = (self.palette_at + 1) % len(PALETTE_CYCLE)
        elif key == "P":
            self.palette_at = (self.palette_at - 1) % len(PALETTE_CYCLE)
        elif key == "b":
            self.bloom = max(0.0, self.bloom - 0.15)
        elif key == "B":
            self.bloom = min(2.0, self.bloom + 0.15)
        elif key in ("c", "C"):
            self.clumps = max(0, self.clumps + (1 if key == "C" else -1))
            self.resized = True          # the pattern lives in the parcels
        elif key in ("s", "S"):
            self.speed = max(0.01, self.speed * (1.3 if key == "S" else 1 / 1.3))
            self.rebuild_rates()
        elif key == "g":
            self.stars_on = not self.stars_on
        elif key == "d":
            self.dither_on = not self.dither_on
        elif key == "n":
            self.scanlines_on = not self.scanlines_on
        elif key == "v":
            self.vignette_on = not self.vignette_on
        elif key == "f":
            self.feather_on = not self.feather_on
        elif key == "a":
            self.ss = 1 if self.ss > 1 else 2
            self.resized = True
        elif key == "r":
            self.seed += 1
            self.resized = True
        elif key in ("e", "E"):
            at = ENCODING_CYCLE.index(self.encoding)
            self.encoding = ENCODING_CYCLE[(at + (1 if key == "e" else -1)) % len(ENCODING_CYCLE)]
            if (self.encoding == "plot1979") != getattr(self, "solved_for", False):
                self.solve("the 1979 disk" if self.encoding == "plot1979" else "the disk")
            self.resized = True
        elif key == "i":
            self.overlay = (self.overlay + 1) % len(OVERLAYS)
        elif key == "o":
            self.dots_on = not self.dots_on
        elif key in ("k", "K"):
            self.glow_at = (self.glow_at + (1 if key == "k" else -1)) % len(GLOW_CYCLE)
        elif key == "m":
            self.mask_on = not self.mask_on
        elif key == "y":
            self.cycling = not self.cycling
            self.next_change = 0.0
        # Everything below changes the gravity, so the map has to be solved again.
        elif key in ("[", "]"):
            step = o.incl_step * (1 if key == "]" else -1)
            s["incl"] = float(np.clip(s["incl"] + step, 0.02, 1.5))
            self.solve(f"inclination {s['incl']:.2f}")
        elif key in ("-", "="):
            step = o.edge_step * (1 if key == "=" else -1)
            s["outer_edge"] = float(np.clip(s["outer_edge"] + step, 8.0, 200.0))
            self.solve(f"disk out to {s['outer_edge']:.0f}")
        elif key in (",", "."):
            step = o.mass_step * (1 if key == "." else -1)
            s["mass"] = float(np.clip(s["mass"] + step, 0.25, 8.0))
            self.solve(f"mass {s['mass']:.2f}")
            self.rebuild_rates()
        return True

    def cycle(self):
        """Hands off: wander through the settings on a timer.

        Only the cheap ones change often. The gravity is stepped much more
        rarely, because each step stops the picture to solve the map again.
        """
        if self.clock < self.next_change:
            return
        self.next_change = self.clock + self.o.cycle_every

        self.step = getattr(self, "step", 0) + 1
        if self.step % 2 == 1:
            self.palette_at = (self.palette_at + 1) % len(PALETTE_CYCLE)
        elif self.step % 6 == 2:
            at = ENCODING_CYCLE.index(self.encoding)
            self.encoding = ENCODING_CYCLE[(at + 1) % len(ENCODING_CYCLE)]
            if (self.encoding == "plot1979") != getattr(self, "solved_for", False):
                self.solve("the 1979 disk" if self.encoding == "plot1979" else "the disk")
            self.resized = True
        elif self.step % 6 == 4:
            self.bloom = 0.0 if self.bloom > 0.3 else 0.5
        elif self.step % 12 == 0:
            s = self.settings
            step = self.o.incl_step * (1 if getattr(self, "tilt_up", True) else -1)
            nxt = s["incl"] + step
            if not 0.25 <= nxt <= 1.5:
                self.tilt_up = not getattr(self, "tilt_up", True)
                nxt = s["incl"] - step
            s["incl"] = float(np.clip(nxt, 0.25, 1.5))
            self.solve(f"inclination {s['incl']:.2f}")

    # ------------------------------------------------------------------- loop

    def run(self):
        self.solve()
        self.fit()

        interval = 1.0 / self.o.fps
        last = time.monotonic()
        drawn = 0
        # The last second or so of frames: when each was shown, and how long
        # each took to make. The rate is capped by --fps, so the time a frame
        # costs is the number that actually differs between modes.
        self.shown = deque(maxlen=60)
        self.costs = deque(maxlen=60)
        began = time.monotonic()

        sys.stdout.write("\033[?25l")
        try:
            while True:
                now = time.monotonic()
                if not self.paused:
                    self.clock += now - last
                last = now

                if self.cycling:
                    self.cycle()
                if self.resized:
                    self.fit()

                made = time.monotonic()
                pane = self.draw()
                self.costs.append(time.monotonic() - made)
                self.shown.append(now)
                # Synchronised output: kitty and Ghostty hold the screen until the
                # end marker, so a frame never appears half drawn. A torn frame
                # reads as a stutter in motion however fast frames arrive.
                sys.stdout.write("\033[?2026h\033[H" + pane + "\n")
                if self.show_help:
                    for i, (key, what) in enumerate(HELP):
                        sys.stdout.write(f"\033[{i + 2};3H\033[2K  {key:<8} {what}")
                    sys.stdout.write(f"\033[H")
                else:
                    sys.stdout.write(f"\033[{self.rows + 1};1H\033[2K{self.status()}")
                sys.stdout.write("\033[?2026l")
                sys.stdout.flush()
                drawn += 1

                # Aim each frame at a fixed beat. Sleeping off whatever time is left
                # lets every frame's own cost push the next one later, so the gaps
                # between frames vary and even motion looks uneven.
                due = getattr(self, "_due", now) + interval
                if due < time.monotonic() - interval:
                    due = time.monotonic()          # fell far behind: start a new beat
                self._due = due
                spare = due - time.monotonic()
                if spare > 0:
                    time.sleep(spare)

                if not self.pump():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\033[?25h\033[2J\033[H")
            sys.stdout.flush()

        ran = time.monotonic() - began
        print(f"{drawn} frames in {ran:.0f}s = {drawn / max(ran, 1e-9):.1f} fps")
        return 0

    def pump(self):
        """Handle everything that has been typed, without waiting for more.

        Reading through sys.stdin one character at a time lost keys: Python
        buffers the stream, so when two keys arrived together the second sat in
        that buffer, invisible to select(), until something else was pressed.
        Reading the descriptor directly takes all of it at once.

        Escape sequences are skipped whole. An arrow key arrives as ESC [ A,
        and handing that over a character at a time made the ESC read as quit.
        A lone ESC, with nothing following it, still quits.
        """
        fd = sys.stdin.fileno()
        while select.select([fd], [], [], 0)[0]:
            data = os.read(fd, 4096)
            if not data:
                return False
            for ch in keys_in(data.decode("utf-8", errors="ignore")):
                if not self.handle(ch):
                    return False
        return True


def run(settings, opts):
    if not sys.stdin.isatty():
        print("the live view needs a terminal", file=sys.stderr)
        return 2
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return Live(settings, opts).run()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
