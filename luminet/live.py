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

import numpy as np

from luminet import cells, spin

PALETTE_CYCLE = ["ember", "inferno", "magma", "amber", "phosphor", "ice",
                 "plasma", "cividis", "bone", "copper", "gameboy", "bw"]

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
    ("h  ?", "this list"),
]


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
        self.w, self.h = self.cols, self.rows * 2          # half-block
        self.rw, self.rh = self.w * self.ss, self.h * self.ss

        # Keep the gas at a constant density whatever the window is doing.
        count = int(self.w * self.h * self.o.density * (2 if self.ss > 1 else 1))
        self.parcels = spin.Parcels(self.mapping["radii"], count=count, seed=self.seed,
                                    infall=self.o.infall, clumps=self.clumps,
                                    depth=self.o.depth)
        # Stars are made at the size they are shown. Generated on the
        # supersampled grid they would be averaged down into grey smudges
        # instead of staying points.
        self.stars = cells.starfield(self.w, self.h, self.o.star_density, self.seed + 5)
        self.resized = False
        sys.stdout.write("\033[2J\033[H")

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
        self.mapping = spin.lensing_map(
            self.settings, n_rings=self.o.rings, n_angles=self.o.angles,
            orders=(0,) if self.o.no_ghost else (0, 1),
            on_progress=bar, batches=6,
        )
        self.extent = spin.reach(self.mapping, (0,) if self.o.no_ghost else (0, 1))
        self.rates = spin.true_rates(self.mapping["radii"],
                                     float(self.settings["mass"]), self.speed)
        sys.stdout.write("\r\033[2K")
        self.resized = True

    def rebuild_rates(self):
        self.rates = spin.true_rates(self.mapping["radii"],
                                     float(self.settings["mass"]), self.speed)

    # ---------------------------------------------------------------- drawing

    def draw(self):
        orders = (0,) if self.o.no_ghost else (0, 1)
        grid = spin.frame(self.mapping, self.parcels, self.clock, self.rw, self.rh,
                          self.extent, None, orders, rates=self.rates,
                          gas_spread=self.o.gas_spread)

        name = PALETTE_CYCLE[self.palette_at]
        from luminet import fields

        value = np.nan_to_num(fields.normalise(grid, "flux", gamma=self.o.gamma))
        if self.dither_on:
            value = cells.dither(value, 0.04)
        img = cells._ramp(value, cells.palette(name)).astype(np.uint8)
        img[~grid["mask"]] = 0

        if self.feather_on:
            img = cells.feather(img, grid["mask"], 1.0)
        if self.bloom > 0:
            img = cells.bloom(img, self.bloom)

        img = cells.downsample(img, self.ss)
        if self.stars_on:
            covered = grid["mask"]
            if self.ss > 1:
                covered = covered[:self.h * self.ss, :self.w * self.ss].reshape(
                    self.h, self.ss, self.w, self.ss).any(axis=(1, 3))
            img = cells.add_stars(img, self.stars, covered, self.o.star_brightness)
        if self.scanlines_on:
            img = cells.scanlines(img, 0.22)
        if self.vignette_on:
            img = cells.vignette(img, 0.4)
        return cells.half_block(img)

    def status(self):
        s = self.settings
        bits = [f"incl {s['incl']:.2f}", f"mass {s['mass']:.2f}",
                f"disk {s['outer_edge']:.0f}", PALETTE_CYCLE[self.palette_at],
                f"{self.cols}x{self.rows}"]
        if self.paused:
            bits.append("PAUSED")
        return "  ".join(bits) + "   h for keys, q to quit"

    # ----------------------------------------------------------------- input

    def handle(self, key):
        """Act on one keypress. Returns False to stop."""
        o, s = self.o, self.settings

        if key in ("q", "\x1b", "\x03"):
            return False
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

    # ------------------------------------------------------------------- loop

    def run(self):
        self.solve()
        self.fit()

        interval = 1.0 / self.o.fps
        last = time.monotonic()
        drawn = 0
        began = time.monotonic()

        sys.stdout.write("\033[?25l")
        try:
            while True:
                now = time.monotonic()
                if not self.paused:
                    self.clock += now - last
                last = now

                if self.resized:
                    self.fit()

                pane = self.draw()
                sys.stdout.write("\033[H" + pane + "\n")
                if self.show_help:
                    for i, (key, what) in enumerate(HELP):
                        sys.stdout.write(f"\033[{i + 2};3H\033[2K  {key:<8} {what}")
                    sys.stdout.write(f"\033[H")
                else:
                    sys.stdout.write(f"\033[{self.rows + 1};1H\033[2K{self.status()}")
                sys.stdout.flush()
                drawn += 1

                spare = interval - (time.monotonic() - now)
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
        """Read whatever has been typed, without waiting for it."""
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                return False
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
