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

from luminet import cells, config, effects, spin

# Tokens keys_in yields for focus reports, which kitty and Ghostty send once
# asked to with CSI ? 1004 h. Neither can be typed as a single key.
FOCUS_IN = "<focus-in>"
FOCUS_OUT = "<focus-out>"

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
from luminet.lines import COLOURS as LINE_COLOURS, STYLES as LINE_STYLES

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
    ("[  ]", "inclination  (smooth once the bank is built)"),
    ("-  =  z  Z", "zoom: disk size  (smooth once the bank is built)"),
    (",  .", "mass  (smooth once the bank is built)"),
    ("e  E", "encoding: half, sextant, braille, 1979 plot"),
    ("i u j", "1979: isoradials, isoredshifts, isofluxlines on or off"),
    ("l  L", "1979: line style: solid, flowing, dotted, pulse, sweep"),
    ("t  T", "1979: line colour: blue, ink, palette, redshift, flux, spectrum"),
    ("w  W", "1979: thinner or thicker lines (pixels)"),
    ("k  K", "1979: glow palette, separate from the dots"),
    ("m", "1979: keep the glow out of the shadow"),
    ("d  D", "1979: less or more dust where the disk is faint"),
    ("x", "1979: real pixels, through the kitty graphics protocol"),
    ("o", "1979: hide or show the dots"),
    ("y", "cycle everything, hands off"),
    ("1-9  0", "presets: pick one, or 0 for a random one"),
    ("+", "save the current look as a preset"),
    ("X X", "delete the current preset (kept in deleted-presets.toml)"),
    ("!", "1979: drop a probe in"),
    ("@", "1979: receive a transmission"),
    ("#", "1979: observatory HUD"),
    ("$", "1979: warp jump"),
    ("A", "1979: events on their own every few minutes"),
    ("tab", "status line"),
    ("h  ?", "this list"),
]


def cell_size():
    """The terminal's character cell in pixels, and whether it was measured.

    The call that reports rows and columns also carries the window's size in
    pixels, which kitty and Ghostty fill in. Dividing one by the other gives
    the real cell, so the picture can be kept in proportion for whatever font
    and size are in use rather than for the one it was first tried with.
    """
    import fcntl
    import struct

    for fd in (sys.stdout.fileno(), sys.stdin.fileno()):
        try:
            rows, cols, px_w, px_h = struct.unpack(
                "HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
        except OSError:
            continue
        if rows and cols and px_w and px_h:
            return (px_w / cols, px_h / rows), True
    return (10.0, 22.0), False


def keys_in(text):
    """The keypresses in a chunk of terminal input, with escape sequences removed.

    Arrow and function keys arrive as ESC followed by [ or O, parameters, and a
    final byte from @ to ~. Replies from the terminal - graphics acknowledgements
    among them - arrive as ESC _ or ESC ] strings. All of those are dropped
    whole. A lone ESC is kept, since on its own it is the escape key.
    """
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b" and i + 1 < len(text) and text[i + 1] in "_]P^":
            # A string the terminal sends back - a graphics reply is ESC _ G ...
            # ESC \ - running to ST or, for OSC, to BEL. Its ESC must not quit.
            j = i + 2
            while j < len(text):
                if text[j] == "\x07":
                    j += 1
                    break
                if text[j] == "\x1b" and j + 1 < len(text) and text[j + 1] == "\\":
                    j += 2
                    break
                j += 1
            i = j
            continue
        if ch == "\x1b" and i + 1 < len(text) and text[i + 1] in "[O":
            j = i + 2
            while j < len(text) and not ("@" <= text[j] <= "~"):
                j += 1
            if text[i + 1] == "[" and j == i + 2 and j < len(text) and text[j] in "IO":
                yield FOCUS_IN if text[j] == "I" else FOCUS_OUT
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
        self.families = set(f for f in getattr(opts, "lines", "").split(",") if f)
        self.line_style = LINE_STYLES.index(getattr(opts, "line_style", "solid"))
        self.line_colour = LINE_COLOURS.index(getattr(opts, "line_colour", "blue"))
        self.line_width = int(getattr(opts, "line_width", 2))
        self.dots_on = True
        self.glow_at = 0
        self.mask_on = True
        self.pixels_on = bool(getattr(opts, "pixels", False))
        # Continuous tilt, zoom and mass: the physics keys set a goal the view
        # eases towards, drawn from the bank of maps while it moves.
        from luminet.bank import Bank

        self.bank = Bank()
        self.using_bank = False
        self.goal = None
        self.moving = False
        self.rest_at = 0.0
        self.generation = 0
        self.rebuilding = None
        self.rebuilt = None
        self.live = None
        self.dust = float(getattr(opts, "dust", 0.012))
        self.transport_mode = None     # found by probing, the first time it is needed
        self.pixel_view = None
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

        # Widget behaviour: settings from the config file, the presets, the
        # events, and whether the window has focus.
        cfg = getattr(opts, "config", None) or {}
        self.status_on = bool(getattr(opts, "status", False))
        self.pixels_auto = getattr(opts, "pixels_mode", "off") == "auto"
        self.unfocused_fps = float(cfg.get("unfocused_fps", 5))
        self.focused = True
        minutes = cfg.get("event_minutes", [4, 10])
        self.effects = effects.Effects(
            seed=self.seed, messages=cfg.get("transmissions") or None,
            events=bool(getattr(opts, "events", True)),
            event_minutes=(float(minutes[0]), float(minutes[-1])))
        self.effects.hud = bool(getattr(opts, "hud", False))
        self.presets = list(getattr(opts, "presets", None) or [])
        self.preset_at = None
        self.pending_delete = None
        self.physics_moved = False
        self.bank_done = None
        self.bank_checked = 0.0
        start = getattr(opts, "preset_index", None)
        if start is not None and 0 <= start < len(self.presets):
            self.preset_at = start
            self.apply_look(self.presets[start], startup=True,
                            keep=getattr(opts, "explicit", set()))

    # ---------------------------------------------------------------- presets

    # A preset's fields, and the command-line option each corresponds to. An
    # option given on the command line wins over the preset opened at start.
    LOOK_OPTIONS = {"palette": "palette", "bloom": "bloom", "speed": "speed",
                    "dust": "dust", "lines": "lines", "line_style": "line_style",
                    "line_colour": "line_colour", "line_width": "line_width",
                    "vignette": "vignette", "scanlines": "scanlines", "incl": "incl",
                    "mass": "mass", "disk": "outer_edge", "hud": "hud"}

    def current_look(self):
        s = self.settings
        physics = self.goal if (self.using_bank and self.goal) else s
        return {
            "palette": PALETTE_CYCLE[self.palette_at], "glow": GLOW_CYCLE[self.glow_at],
            "bloom": round(float(self.bloom), 3), "incl": round(float(physics["incl"]), 4),
            "mass": round(float(physics["mass"]), 4),
            "disk": round(float(physics["outer_edge"]), 2),
            "speed": round(float(self.speed), 4), "dust": round(float(self.dust), 4),
            "lines": sorted(self.families), "line_style": LINE_STYLES[self.line_style],
            "line_colour": LINE_COLOURS[self.line_colour], "line_width": int(self.line_width),
            "dots": bool(self.dots_on), "mask": bool(self.mask_on),
            "vignette": bool(self.vignette_on), "scanlines": bool(self.scanlines_on),
            "hud": bool(self.effects.hud),
        }

    def apply_look(self, look, startup=False, keep=()):
        """Take on a preset's look. Fields it lacks are left as they are.

        At start the physics is simply set, since nothing has been solved yet.
        Later it glides there when the bank allows, or is solved again.
        """
        skip = {k for k, dest in self.LOOK_OPTIONS.items() if dest in keep}

        def has(key):
            return key in look and key not in skip

        if has("palette") and look["palette"] in PALETTE_CYCLE:
            self.palette_at = PALETTE_CYCLE.index(look["palette"])
        if "glow" in look and look["glow"] in GLOW_CYCLE:
            self.glow_at = GLOW_CYCLE.index(look["glow"])
        if has("bloom"):
            self.bloom = float(np.clip(float(look["bloom"]), 0.0, 2.0))
        if has("lines"):
            wanted = look["lines"]
            if isinstance(wanted, str):
                wanted = [w for w in wanted.split(",") if w]
            self.families = {f for f in wanted if f in ("radii", "redshift", "flux")}
        if has("line_style") and look["line_style"] in LINE_STYLES:
            self.line_style = LINE_STYLES.index(look["line_style"])
        if has("line_colour") and look["line_colour"] in LINE_COLOURS:
            self.line_colour = LINE_COLOURS.index(look["line_colour"])
        if has("line_width"):
            self.line_width = int(np.clip(int(look["line_width"]), 1, 6))
        for key, attr in (("dots", "dots_on"), ("mask", "mask_on"),
                          ("vignette", "vignette_on"), ("scanlines", "scanlines_on")):
            if has(key):
                setattr(self, attr, bool(look[key]))
        if has("hud"):
            self.effects.hud = bool(look["hud"])
        if has("dust"):
            self.set_dust(float(look["dust"]), quiet=True)
        if has("speed"):
            self.speed = max(0.01, float(look["speed"]))
            if not startup and self.mapping is not None:
                self.rebuild_rates()

        wanted = {}
        for key, name, low, high in (("incl", "incl", 0.05, 1.55), ("mass", "mass", 0.25, 8.0),
                                     ("disk", "outer_edge", 8.0, 200.0)):
            if has(key):
                wanted[name] = float(np.clip(float(look[key]), low, high))
        if not wanted:
            return
        if startup or self.mapping is None:
            self.settings.update(wanted)
        elif self.using_bank and self.goal is not None:
            self.goal.update(wanted)
        elif any(abs(self.settings[k] - v) > 1e-9 for k, v in wanted.items()):
            self.settings.update(wanted)
            self.solve("the preset's view")
            self.rebuild_rates()

    def set_dust(self, value, quiet=False):
        self.dust = float(np.clip(value, 0.0, 0.1))
        for view in (getattr(self, "field", None),
                     getattr(self.pixel_view, "field", None) if self.pixel_view else None):
            if view is not None:
                view.set_floor(self.dust)
        if self.pixel_view is not None:
            self.pixel_view.look = None          # dot colours follow brightness
        if not quiet:
            self.note = (f"dust {self.dust:.3f}: least chance of a dot wherever light "
                         f"arrives", time.monotonic() + 2.5)

    def load_preset(self, index):
        preset = self.presets[index]
        self.preset_at = index
        self.apply_look(preset)
        self.note = (f"preset {index + 1}: {preset.get('name', 'unnamed')}",
                     time.monotonic() + 3.0)

    def save_preset(self):
        look = self.current_look()
        look = {"name": config.unique_name(self.presets), **look}
        self.presets.append(look)
        try:
            config.save_presets(self.presets)
        except OSError as e:
            self.presets.pop()
            self.note = (f"could not save the preset: {e}", time.monotonic() + 5.0)
            return
        self.preset_at = len(self.presets) - 1
        where = "" if self.preset_at >= 9 else f", key {self.preset_at + 1}"
        self.note = (f"saved as preset '{look['name']}'{where}; rename it in "
                     f"{config.presets_path()}", time.monotonic() + 5.0)

    def delete_preset(self, now):
        if self.preset_at is None or not self.presets:
            self.note = ("no preset is in use to delete", now + 3.0)
            return
        at = self.preset_at
        name = self.presets[at].get("name", "unnamed")
        if not (self.pending_delete and self.pending_delete[0] == at
                and now < self.pending_delete[1]):
            self.pending_delete = (at, now + 3.0)
            self.note = (f"X again to delete preset {at + 1} '{name}'", now + 3.0)
            return
        self.pending_delete = None
        removed = self.presets.pop(at)
        try:
            config.archive_deleted(removed)
            config.save_presets(self.presets)
        except OSError as e:
            self.presets.insert(at, removed)
            self.note = (f"could not delete the preset: {e}", now + 5.0)
            return
        self.preset_at = None
        self.note = (f"deleted '{name}'; a copy is in deleted-presets.toml", now + 4.0)

    # ------------------------------------------------------------ housekeeping

    def _winch(self, *_):
        self.resized = True

    def fit(self):
        """Match the window. Called at the start and whenever it changes."""
        size = shutil.get_terminal_size((90, 28))
        # The whole window. The status line and notes are drawn over the
        # picture's bottom row rather than taking a row of their own.
        self.cols = self.o.width or max(24, size.columns)
        self.rows = self.o.height or max(8, size.lines)
        sub = ENCODINGS[self.encoding]
        self.w, self.h = self.cols * sub["x"], self.rows * sub["y"]
        # Measured every time the window changes: zooming the font sends the
        # same resize signal and changes the cell's shape.
        self.cell_px, self.cell_measured = cell_size()
        spin.CELL_ASPECT = spin.sample_aspect(*self.cell_px, sub["x"], sub["y"])
        self.cell_aspect = spin.CELL_ASPECT
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
        if self.encoding == "plot1979" and self.pixels_on:
            from luminet import pixels

            if self.transport_mode is None:
                self.transport_mode = ("file" if pixels.probe_file_transport()
                                       else "direct")
            if self.pixel_view is not None:
                self.pixel_view.transport.close()
            orders = (0,) if self.o.no_ghost else (0, 1)
            self.pixel_view = pixels.PixelView(
                self.mapping, self.extent, self.rates, self.projector, orders,
                self.cols, self.rows, self.cell_px, pixels.Transport(self.transport_mode),
                seed=self.seed, infall=self.o.infall, gamma=self.o.ink_gamma,
                grain=getattr(self.o, "pixel_grain", 4), floor=self.dust)
        elif self.encoding == "plot1979":
            # No tracer: the dots are the material, so the motion shows without
            # one. Many parcels, because most of them are faint and thinly kept.
            self.dust_parcels = spin.Parcels(self.mapping["radii"], count=int(self.w * self.h * 2.5),
                                     seed=self.seed, infall=self.o.infall, clumps=0,
                                     spread="log")
            # How bright each cell should be is fixed for a given map and
            # window, so it is measured here once rather than every frame.
            orders = (0,) if self.o.no_ghost else (0, 1)
            self.field = spin.DotField(self.mapping, self.dust_parcels, self.w, self.h,
                                       self.extent, self.rates, orders, gamma=self.o.ink_gamma,
                                       projector=self.projector, floor=self.dust,
                                       cell_aspect=self.cell_aspect)
            self.lineset = self.make_lineset(self.field, self.w, self.h)
            self.hole = self.shadow_coverage(self.field, self.mapping)
        self.live = None
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
        # Nearly edge-on, the disk's band on screen is its radius times the
        # cosine of the inclination, so a fixed radius leaves the top and bottom
        # of a tall window empty: at 1.50 a disk to 160 put no light at all in the
        # bottom tenth of a tall window. Widening as 1 / cos(inclination), from a
        # size that fills it comfortably at 1.30, keeps the frame covered.
        base = max(160.0, self.settings["outer_edge"] * 4)
        tilt = np.cos(1.30) / max(np.cos(self.settings["incl"]), 0.02)
        wide["outer_edge"] = float(min(2000.0, base * max(1.0, tilt)))
        return wide

    def framing(self, mapping):
        """How much of the sky to show, from the zoom setting.

        The zoom is a radius to frame, but mass scales the whole geometry, so a
        radius smaller than the disk's own inner edge frames nothing: zoomed
        right in at mass 3 the inner edge is at 18 and nothing lay inside 8, and
        the view collapsed. The framed radius never goes inside half again the
        inner edge.
        """
        orders = (0,) if self.o.no_ghost else (0, 1)
        radius = max(self.settings["outer_edge"] * 0.85, float(mapping["radii"][0]) * 1.5)
        rx, ry = spin.reach(mapping, orders, max_radius=radius)
        return (rx * 0.78, ry * 0.95)

    def solve(self, why=""):
        """Rebuild the lensing map, showing how far along it is.

        plot1979 takes its map from the bank of inclinations when the bank has
        the maps either side, which needs no solving at all and lets the view
        move continuously afterwards. Otherwise it solves, and sets the bank
        filling in the background for next time.
        """
        orders = (0,) if self.o.no_ghost else (0, 1)
        if self.encoding == "plot1979":
            from luminet import bank as bank_module, fast

            s = self.settings
            if self.bank.ready_for(s["incl"]):
                self.mapping = self.bank.mapping(s["incl"], s["mass"], s.get("acc", 1.0))
                self.solved_for = True
                self.using_bank = True
                self.goal = {k: s[k] for k in ("incl", "mass", "outer_edge")}
                self.extent = self.framing(self.mapping)
                self.rates = spin.true_rates(self.mapping["radii"], float(s["mass"]), self.speed)
                self.projector = (fast.Projector(self.mapping, orders)
                                  if fast.available and not self.o.no_compile else None)
                self.resized = True
                return
            bank_module.start_building(s["incl"])
        self.using_bank = False
        label = why or "solving the lensing map"
        width = max(20, min(48, self.cols - len(label) - 14)) if self.cols else 30

        def bar(done, total):
            filled = int(width * done / total)
            sys.stdout.write(
                f"\r  {label}  [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
            sys.stdout.flush()

        if getattr(self, "first_run", False):
            self.first_run = False       # the notice stays up, with the bar under it
        else:
            sys.stdout.write("\033[2J\033[H\n")
        physics = self.physics()
        self.solved_for = self.encoding == "plot1979"
        rings = self.o.rings * 2 if self.solved_for else self.o.rings
        self.mapping = spin.lensing_map(
            physics, n_rings=rings, n_angles=self.o.angles,
            orders=(0,) if self.o.no_ghost else (0, 1),
            on_progress=bar, batches=6,
            spacing="log" if self.solved_for else "linear",
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

    def advance(self, now, dt):
        """Ease the view towards its goal, and rebuild properly once it rests."""
        if self.rebuilt is not None:
            generation, result = self.rebuilt
            self.rebuilt = None
            self.rebuilding = None
            if generation == self.generation:
                self.adopt(result)
        if self.encoding == "plot1979" and self.projector is not None:
            # The warp jump zooms by drawing moving frames at a scaled extent.
            # Nothing about the map changes, so once it ends the still view
            # built before it is right again - unless the view also moved.
            zooming = self.effects.zoom() != 1.0
            if zooming:
                self.moving = True
                self.generation += 1
                self.rest_at = now + 0.25
            elif getattr(self, "was_zooming", False) and not self.physics_moved:
                self.moving = False
                self.live = None
            self.was_zooming = zooming
            if zooming:
                return
        if not (self.encoding == "plot1979" and self.using_bank and self.goal):
            return
        s, g = self.settings, self.goal
        ease = 1.0 - np.exp(-dt * 8.0)
        changed = False
        for key, close in (("incl", 5e-4), ("mass", 1e-3), ("outer_edge", 0.05)):
            gap = g[key] - s[key]
            if gap == 0:
                continue
            s[key] = g[key] if abs(gap) <= close else s[key] + gap * ease
            changed = True
        if changed:
            mapping = self.bank.mapping(s["incl"], s["mass"], s.get("acc", 1.0))
            if mapping is None:
                s.update(g)
                self.solve()
                return
            self.mapping = mapping
            self.rates = spin.true_rates(mapping["radii"], float(s["mass"]), self.speed)
            self.extent = self.framing(mapping)
            if self.projector is None:
                # Without the compiled path a moving frame would be too slow, so
                # jump to the goal and rebuild there.
                s.update(g)
                self.mapping = self.bank.mapping(s["incl"], s["mass"], s.get("acc", 1.0))
                self.extent = self.framing(self.mapping)
                self.resized = True
                return
            self.projector.set_tables(mapping)
            self.physics_moved = True
            self.moving = True
            self.generation += 1
            self.rest_at = now + 0.25
        elif self.moving and self.rebuilding is None and now >= self.rest_at:
            self.start_rebuild()

    def start_rebuild(self):
        """Build the still view's field, lines and mask on a background thread."""
        import threading

        from luminet import fast

        generation = self.generation
        mapping, extent, rates = self.mapping, self.extent, np.array(self.rates)
        orders = (0,) if self.o.no_ghost else (0, 1)
        pixels_on = self.pixels_on and self.pixel_view is not None

        def work():
            try:
                projector = fast.Projector(mapping, orders)
                if pixels_on:
                    from luminet import pixels

                    view = pixels.PixelView(
                        mapping, extent, rates, projector, orders, self.cols, self.rows,
                        self.cell_px, pixels.Transport(self.transport_mode), seed=self.seed,
                        infall=self.o.infall, gamma=self.o.ink_gamma,
                        grain=getattr(self.o, "pixel_grain", 4), floor=self.dust)
                    view.lineset = self.make_lineset(view.field, view.px_w, view.px_h, mapping)
                    result = {"projector": projector, "view": view}
                else:
                    field = spin.DotField(mapping, self.dust_parcels, self.w, self.h, extent,
                                          rates, orders, gamma=self.o.ink_gamma,
                                          projector=projector, floor=self.dust,
                                          cell_aspect=self.cell_aspect)
                    result = {"projector": projector, "field": field,
                              "lineset": self.make_lineset(field, self.w, self.h, mapping),
                              "hole": self.shadow_coverage(field, mapping)}
            except Exception as e:                     # never take the view down
                result = {"error": e}
            self.rebuilt = (generation, result)

        self.rebuilding = threading.Thread(target=work, daemon=True)
        self.rebuilding.start()

    def adopt(self, result):
        if "error" in result:
            self.note = (f"rebuild failed: {result['error']}", time.monotonic() + 4)
            self.moving = False
            self.resized = True
            return
        self.projector = result["projector"]
        self.physics_moved = False
        if "view" in result:
            if self.pixel_view is not None and self.pixel_view is not result["view"]:
                self.pixel_view.transport.close()
            self.pixel_view = result["view"]
        else:
            self.field = result["field"]
            self.lineset = result["lineset"]
            self.hole = result["hole"]
            self.live = None
        self.moving = False

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
        self.effects.canvas = None
        post = None
        if self.effects.active():
            def post(img, ext_x, ext_y):
                self.effects.paint(effects.PixelCanvas(img, ext_x, ext_y, self.cols, self.rows),
                                   self)

        if self.pixels_on and self.pixel_view is not None and self.moving \
                and self.projector is not None:
            return self.pixel_view.frame_moving(
                self.clock, self.rates, self.mapping, self.view_extent(),
                PALETTE_CYCLE[self.palette_at], GLOW_CYCLE[self.glow_at], self.bloom,
                self.mask_on, INK, PAPER, HOLE, dots_on=self.dots_on, post=post)

        if self.pixels_on and self.pixel_view is not None:
            view = self.pixel_view
            view.restyle(PALETTE_CYCLE[self.palette_at], GLOW_CYCLE[self.glow_at],
                         self.bloom, self.mask_on, self.vignette_on, self.scanlines_on,
                         INK, PAPER, HOLE)
            if getattr(view, "lineset", None) is None:
                view.lineset = self.make_lineset(view.field_for_lines, view.px_w, view.px_h)
            return view.frame(self.clock, self.rates, dots_on=self.dots_on,
                              lines=self.line_args() if self.families else None,
                              line_width=self.line_width, post=post)

        source, hole = self.field, self.hole
        if self.moving and self.projector is not None:
            # The map is changing under the view: measure this moment only, and
            # let the lines rest until it settles.
            if self.live is None:
                self.live = spin.LiveField(self.w, self.h, gamma=self.o.ink_gamma,
                                           floor=self.dust, scale=self.field.scale,
                                           cell_aspect=self.cell_aspect, stride=2)
            self.live.floor = self.dust
            source = self.live.measure(self.dust_parcels, self.clock, self.rates,
                                       self.view_extent(), self.projector)
            hole = self.shadow_coverage(source, self.mapping)

        if self.dots_on:
            lit = source.frame(self.clock, self.rates).copy()
        else:
            lit = np.zeros((self.h, self.w), dtype=bool)
        line_cells = None
        if self.families and not self.moving:
            idx, rgb = self.lineset.draw(self.clock, self.rates, width=1, **self.line_args())
            lit.reshape(-1)[idx] = True
            rows_i, cols_i = np.divmod(idx, self.w)
            cell = (rows_i // 4) * self.cols + cols_i // 2
            ok = (rows_i < self.rows * 4) & (cols_i < self.cols * 2)
            line_cells = (cell[ok], rgb[ok])

        name = PALETTE_CYCLE[self.palette_at]
        styled = (self.bloom > 0 or self.scanlines_on or self.vignette_on
                  or (self.mask_on and HOLE != PAPER) or self.effects.active())
        if name == "ink" and line_cells is None and not styled:
            return cells.braille(lit, INK, PAPER)

        rows, cols = self.rows, self.cols
        bright = source.target[:rows * 4, :cols * 2].reshape(rows, 4, cols, 2).max(axis=(1, 3))
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
            # A braille cell has one colour; a line crossing it takes it.
            colours.reshape(-1, 3)[line_cells[0]] = line_cells[1]

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
        if self.mask_on and hole is not None:
            # The shadow takes no light from anywhere, so the glow stops at its
            # edge. Only the glow: gas on the near side of the disk passes in
            # front of the hole, and its dots stay.
            cover = hole[..., None]
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
        if self.effects.active():
            self.effects.paint(effects.DotCanvas(lit, colours, backgrounds, source.ext_x,
                                                 source.ext_y, cols, rows), self)
        return cells.braille(lit, INK, PAPER,
                             colours=np.clip(colours, 0, 255).astype(np.uint8),
                             backgrounds=np.clip(backgrounds, 0, 255).astype(np.uint8))

    def view_extent(self):
        """The framed extent, scaled while the warp jump zooms."""
        zoom = self.effects.zoom()
        if zoom == 1.0:
            return self.extent
        return (self.extent[0] * zoom, self.extent[1] * zoom)

    def make_lineset(self, field, width, height, mapping=None):
        from luminet import lines

        mapping = mapping if mapping is not None else self.mapping

        o = self.o
        def numbers(text, default):
            return tuple(float(v) for v in text.split(",") if v.strip()) if text else default
        return lines.LineSet(
            mapping, field, width, height, field.ext_x, field.ext_y,
            radii=numbers(getattr(o, "iso_radii", ""), (6, 10, 15, 20)),
            ghost=numbers(getattr(o, "iso_ghost", ""), (6, 20, 50, 100)),
            redshifts=numbers(getattr(o, "redshift_levels", ""), lines.DEFAULT_REDSHIFTS),
            flux_levels=numbers(getattr(o, "flux_levels", ""), lines.DEFAULT_FLUX_LEVELS))

    def line_args(self):
        return dict(families=self.families, style=LINE_STYLES[self.line_style],
                    colour=LINE_COLOURS[self.line_colour],
                    palette=cells.palette(PALETTE_CYCLE[self.palette_at]))

    def clear_images(self):
        """Remove any picture placed with the graphics protocol."""
        sys.stdout.write("\033_Ga=d,d=A,q=2\033\\")
        sys.stdout.flush()
        if self.pixel_view is not None:
            self.pixel_view.transport.close()
            self.pixel_view = None

    def shadow_coverage(self, field, mapping, sub=6):
        """How much of each cell lies inside the black hole's shadow, 0 to 1.

        For a non-rotating black hole the shadow on the observer's screen is an
        exact circle, radius 3 sqrt(3) M, whatever the inclination - so a round
        mask is also the physically right one. Backgrounds are coloured a whole
        cell at a time, and a cell is 10 by 22 pixels, so a hard mask would come
        out as a stepped circle. Each cell is sampled at several points instead
        and gets a fraction, which blends the edge.
        """
        radius = float(mapping["bh"].critical_b)
        cols, rows = self.cols, self.rows
        # Sample points spread over each cell's 2x4 dots, in dot coordinates.
        fx = (np.arange(sub) + 0.5) / sub * 2.0
        fy = (np.arange(sub * 2) + 0.5) / (sub * 2) * 4.0
        dot_x = np.arange(cols)[:, None] * 2.0 + fx[None, :]
        dot_y = np.arange(rows)[:, None] * 4.0 + fy[None, :]
        bx = (dot_x / (self.w - 1) * 2 - 1) * field.ext_x
        by = (dot_y / (self.h - 1) * 2 - 1) * field.ext_y
        inside = (by[:, None, :, None] ** 2 + bx[None, :, None, :] ** 2) < radius ** 2
        cover = inside.mean(axis=(2, 3)).astype(np.float32)
        # Gas on the near side of the disk passes in front of the hole, and the
        # glow belongs to that gas, so it is not masked where the gas is. The
        # direct image's footprint is softened a little, since at its edge it is
        # measured from samples and is ragged.
        from scipy.ndimage import gaussian_filter

        front = (field.front > 0).astype(np.float32)
        front = gaussian_filter(front, 1.0)[:rows * 4, :cols * 2]
        in_front = front.reshape(rows, 4, cols, 2).mean(axis=(1, 3))
        return cover * (1.0 - np.clip(in_front * 1.5, 0.0, 1.0))

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
                PALETTE_CYCLE[self.palette_at], f"{self.cols}x{self.rows}"
                + (f" @{self.cell_px[0]:.0f}x{self.cell_px[1]:.0f}px"
                   if getattr(self, "cell_measured", False) else " cell size assumed")]
        if self.cycling:
            bits.append("cycling")
        if self.encoding == "plot1979":
            bits.append("compiled" if self.projector is not None else "numpy")
            if self.pixels_on:
                bits.append(f"pixels by {self.transport_mode or '?'}")
            if self.families:
                names = "+".join(f for f in ("radii", "redshift", "flux") if f in self.families)
                bits.append(f"lines {names} {LINE_STYLES[self.line_style]} "
                            f"{LINE_COLOURS[self.line_colour]}")
            if not self.dots_on:
                bits.append("dots hidden")
            if self.bloom > 0:
                bits.append(f"glow {GLOW_CYCLE[self.glow_at]} {self.bloom:.2f}")
            if not self.mask_on:
                bits.append("mask off")
        if not self.focused:
            bits.append(f"unfocused {self.unfocused_fps:g}fps")
        if self.preset_at is not None and self.preset_at < len(self.presets):
            bits.append(f"preset {self.presets[self.preset_at].get('name', '?')}")
        if self.paused:
            bits.append("PAUSED")
        if self.encoding == "plot1979":
            if self.moving:
                bits.append("moving")
            done = self.bank_count()
            if done < 31:
                bits.append(f"bank {done}/31")
        shown = getattr(self, "shown", ())
        if len(shown) > 1:
            span = shown[-1] - shown[0]
            rate = (len(shown) - 1) / span if span > 0 else 0.0
            cost = 1000 * sum(self.costs) / len(self.costs)
            bits.append(f"{rate:4.1f}fps {cost:3.0f}ms/frame")
        return "  ".join(bits) + "   h for keys, q to quit"

    def bank_count(self):
        now = time.monotonic()
        if self.bank_done is None or now - self.bank_checked > 1.0:
            before = self.bank_done
            self.bank_done = self.bank.count()
            self.bank_checked = now
            if before is not None and before < 31 <= self.bank_done:
                self.note = ("lensing bank complete: tilt, zoom and mass now glide "
                             "(after the next change)", now + 5.0)
        return self.bank_done

    def bottom_line(self):
        """The one line of text under everything: a note, the bank, or the status."""
        now = time.monotonic()
        text, until = self.note
        if text and now < until:
            return text
        if self.encoding == "plot1979" and self.bank_count() < 31:
            from luminet import bank as bank_module

            return (f"first run: building the lensing bank in the background, "
                    f"{self.bank_done}/31 maps, kept in {bank_module.cache_dir().parent}; "
                    f"tilt, zoom and mass step until it is done")
        if self.status_on:
            return self.status()
        return ""

    # ----------------------------------------------------------------- input

    def handle(self, key):
        """Act on one keypress. Returns False to stop."""
        o, s = self.o, self.settings

        now = time.monotonic()
        if key in ("q", "\x1b", "\x03"):
            return False
        if key in (FOCUS_IN, FOCUS_OUT):
            self.focused = key == FOCUS_IN
            return True
        if key == "\t":
            self.status_on = not self.status_on
            return True
        if key.isdigit():
            if not self.presets:
                self.note = (f"no presets yet: + saves one, to {config.presets_path()}",
                             now + 4.0)
            elif key == "0":
                choices = [i for i in range(len(self.presets)) if i != self.preset_at]
                self.load_preset(int(self.effects.rng.choice(choices)) if choices else 0)
            elif int(key) <= len(self.presets):
                self.load_preset(int(key) - 1)
            else:
                self.note = (f"only {len(self.presets)} presets", now + 2.5)
            return True
        if key == "+":
            self.save_preset()
            return True
        if key == "X":
            self.delete_preset(now)
            return True
        if key in ("!", "@", "#", "$", "A"):
            if self.encoding != "plot1979":
                self.note = ("the probe, transmission, HUD and warp are plot1979 only", now + 3.0)
            elif key == "!":
                self.effects.launch_probe(self)
                self.note = ("probe away: falling in from rest", now + 2.5)
            elif key == "@":
                self.effects.launch_transmission(self)
                self.note = ("incoming transmission, one cuneiform sign per byte", now + 3.0)
            elif key == "#":
                self.effects.hud = not self.effects.hud
            elif key == "$":
                if self.projector is None:
                    self.note = ("the warp needs the compiled path (numba); flash only", now + 3.0)
                self.effects.launch_warp(self)
            else:
                self.effects.events = not self.effects.events
                self.effects.next_event = None
                self.note = ("events on their own: " + ("on" if self.effects.events else "off"),
                             now + 2.5)
            return True
        if self.encoding == "plot1979" and key in ("d", "D"):
            self.set_dust(self.dust + (0.004 if key == "D" else -0.004))
            return True
        if self.encoding == "plot1979" and key in ("g", "f", "a"):
            what = {"g": "stars", "f": "edge softening", "a": "antialiasing"}[key]
            self.note = (f"{what} is for half, sextant and braille; plot1979 has none",
                         time.monotonic() + 3.0)
            return True
        if self.encoding != "plot1979" and key in ("i", "u", "j", "l", "L", "t", "T",
                                                     "w", "W", "o", "k", "K", "m", "x"):
            self.note = ("lines, hiding dots, glow palette, mask and pixels are plot1979 only",
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
            self.clear_images()
            at = ENCODING_CYCLE.index(self.encoding)
            self.encoding = ENCODING_CYCLE[(at + (1 if key == "e" else -1)) % len(ENCODING_CYCLE)]
            if (self.encoding == "plot1979") != getattr(self, "solved_for", False):
                self.solve("the 1979 disk" if self.encoding == "plot1979" else "the disk")
            self.resized = True
        elif key in ("i", "u", "j"):
            family = {"i": "radii", "u": "redshift", "j": "flux"}[key]
            self.families ^= {family}
        elif key in ("l", "L"):
            self.line_style = (self.line_style + (1 if key == "l" else -1)) % len(LINE_STYLES)
        elif key in ("t", "T"):
            self.line_colour = (self.line_colour + (1 if key == "t" else -1)) % len(LINE_COLOURS)
        elif key in ("w", "W"):
            self.line_width = int(np.clip(self.line_width + (1 if key == "W" else -1), 1, 6))
            if not self.pixels_on:
                self.note = ("line width is for pixel mode (x); braille lines are one dot",
                             time.monotonic() + 2.5)
        elif key == "o":
            self.dots_on = not self.dots_on
        elif key in ("k", "K"):
            self.glow_at = (self.glow_at + (1 if key == "k" else -1)) % len(GLOW_CYCLE)
        elif key == "m":
            self.mask_on = not self.mask_on
        elif key == "x":
            self.pixels_on = not self.pixels_on
            if not self.pixels_on:
                self.clear_images()
            self.resized = True
        elif key == "y":
            self.cycling = not self.cycling
            self.next_change = 0.0
        # Everything below changes the gravity, so the map has to be solved again.
        elif self.using_bank and key in ("[", "]", "-", "=", "z", "Z", ",", "."):
            from luminet import bank as bank_module

            g = self.goal
            if key in ("[", "]"):
                g["incl"] = float(np.clip(g["incl"] + o.incl_step * (1 if key == "]" else -1),
                                          bank_module.INCLINATIONS[0],
                                          bank_module.INCLINATIONS[-1]))
            elif key in ("-", "=", "z", "Z"):
                wider = key in ("=", "Z")
                g["outer_edge"] = float(np.clip(g["outer_edge"] + o.edge_step * (1 if wider else -1),
                                                8.0, 200.0))
            else:
                g["mass"] = float(np.clip(g["mass"] + o.mass_step * (1 if key == "." else -1),
                                          0.25, 8.0))
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
            if self.using_bank and self.goal is not None:
                self.goal["incl"] = float(np.clip(nxt, 0.25, 1.5))
            else:
                s["incl"] = float(np.clip(nxt, 0.25, 1.5))
                self.solve(f"inclination {s['incl']:.2f}")

    # ------------------------------------------------------------------- loop

    def run(self):
        if self.encoding == "plot1979" and self.pixels_auto and not self.pixels_on:
            from luminet import pixels

            self.pixels_on = pixels.probe_graphics()
        if self.encoding == "plot1979" and not self.bank.ready_for(self.settings["incl"]):
            self.first_run_notice()
        self.solve()
        self.fit()
        if not self.status_on:
            self.note = ("h for keys, tab for the status line, q to quit",
                         time.monotonic() + 4.0)

        interval = 1.0 / self.o.fps
        last = time.monotonic()
        drawn = 0
        # The last second or so of frames: when each was shown, and how long
        # each took to make. The rate is capped by --fps, so the time a frame
        # costs is the number that actually differs between modes.
        self.shown = deque(maxlen=60)
        self.costs = deque(maxlen=60)
        began = time.monotonic()

        # Focus reports, so the frame rate can drop while another window is in use.
        sys.stdout.write("\033[?25l\033[?1004h")
        self.text_rows = set()
        try:
            while True:
                now = time.monotonic()
                if not self.paused:
                    self.clock += (now - last) * self.effects.time_direction()
                last = now

                step = now - getattr(self, "_last_advance", now)
                self.advance(now, step)
                self._last_advance = now
                if self.encoding == "plot1979" and self.mapping is not None:
                    self.effects.step(self, now, step)
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
                sys.stdout.write("\033[?2026h\033[H" + pane)
                over_pixels = self.encoding == "plot1979" and self.pixels_on \
                    and self.pixel_view is not None
                written = set()
                texts = []
                if self.show_help:
                    for i, (key, what) in enumerate(HELP[:max(0, self.rows - 2)]):
                        texts.append(f"\033[{i + 2};3H\033[0m\033[2K  {key:<10} {what}")
                        written.add(i + 2)
                line = self.bottom_line()
                if line:
                    texts.append(f"\033[{self.rows};1H\033[0m\033[2K{line[:self.cols - 1]}")
                    written.add(self.rows)
                if over_pixels:
                    # Text over the picture is never overwritten by it, so lines
                    # that held text last frame and not this one are blanked.
                    for row in self.text_rows - written:
                        sys.stdout.write(f"\033[{row};1H\033[0m\033[2K")
                self.text_rows = written
                if self.encoding == "plot1979" and (self.effects.active()
                                                    or self.effects.last_cells
                                                    or self.effects.readout[0]):
                    sys.stdout.write(self.effects.overlay(self, clear=over_pixels))
                sys.stdout.write("".join(texts) + "\033[0m\033[?2026l")
                sys.stdout.flush()
                drawn += 1

                # Aim each frame at a fixed beat. Sleeping off whatever time is left
                # lets every frame's own cost push the next one later, so the gaps
                # between frames vary and even motion looks uneven.
                # Out of focus, a low rate saves power; 0 keeps the full rate.
                fps = self.o.fps if (self.focused or self.unfocused_fps <= 0) \
                    else min(self.o.fps, self.unfocused_fps)
                interval = 1.0 / fps
                due = getattr(self, "_due", now) + interval
                if due < time.monotonic() - interval:
                    due = time.monotonic()          # fell far behind: start a new beat
                self._due = due
                spare = due - time.monotonic()
                if spare > 0:
                    # Wait on the keyboard rather than sleeping, so a key - or
                    # the window regaining focus - is answered at once even at
                    # a low frame rate.
                    select.select([sys.stdin.fileno()], [], [], spare)

                if not self.pump():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.restore_terminal()

        ran = time.monotonic() - began
        print(f"{drawn} frames in {ran:.0f}s = {drawn / max(ran, 1e-9):.1f} fps")
        return 0

    def first_run_notice(self):
        """Say what the first run is doing, before the first solve holds the screen."""
        from luminet import bank as bank_module

        lines = [
            "First run: building the lensing bank.",
            "",
            "31 maps of where light from the disk lands, one for each tilt, are solved",
            "once in the background and kept in " + str(bank_module.cache_dir().parent) + ".",
            "It takes about a minute. Until it is done, tilt, zoom and mass move a step",
            "at a time; everything else works straight away.",
            "",
            "Solving this view first:",
        ]
        size = shutil.get_terminal_size((90, 28))
        top = max(1, size.lines // 2 - len(lines))
        sys.stdout.write("\033[2J")
        for i, text in enumerate(lines):
            sys.stdout.write(f"\033[{top + i};{max(1, (size.columns - 76) // 2)}H{text}")
        sys.stdout.write(f"\033[{top + len(lines) + 1};1H")
        sys.stdout.flush()
        self.first_run = True

    def restore_terminal(self):
        """Put the terminal back: images, focus reports, cursor, screen.

        Called however the loop ends, including on a signal. If the window has
        already gone, writing fails, and that is fine: there is nothing to restore.
        """
        try:
            if self.pixels_on or self.pixel_view is not None:
                self.clear_images()
            sys.stdout.write("\033[0m\033[?1004l\033[?25h\033[2J\033[H")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
        finally:
            if self.pixel_view is not None:
                self.pixel_view.transport.close()

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

    # Closing the window sends SIGHUP; a session ending or `kill` sends SIGTERM.
    # Either becomes an ordinary exit, so the loop's cleanup runs: images
    # deleted, temporary files removed, the terminal put back.
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        tty.setcbreak(fd)
        return Live(settings, opts).run()
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, OSError):
            pass
