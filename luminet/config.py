"""Settings that outlive a run: the config file and the presets.

Two files, kept apart on purpose, in ~/.config/<app>/:

    config.toml    yours. Written once, commented, the first time the app runs,
                   and never written again, so edits and comments are safe.
    presets.toml   the app's. Rewritten whenever a preset is saved or deleted
                   from inside the app. Hand edits are fine - renaming a preset,
                   nudging a value - but comments in it will not survive.

A deleted preset is not thrown away: it is appended to deleted-presets.toml, so
a look removed by a stray keypress can be copied back.

APP is the one place the app's name is spelled for its directories.
"""

import json
import os
import random
import tomllib
from pathlib import Path

APP = "eventhorizon"

DEFAULT_CONFIG = """\
# Settings for the {app} terminal widget. This file is yours: the app writes it
# once and never again. Delete it to get the defaults back.

# Which preset to open with: "random", "first", or a preset's name.
# "none" starts from the command-line flags alone.
start = "random"

# Frames a second while the window has focus, and while it does not. In pixels,
# 30fps costs about 60% of a core and 20fps about 46%; below 20 the motion
# starts to look like it lags. 0 for unfocused_fps keeps the full rate always.
fps = 30
unfocused_fps = 20

# Show the status line at the bottom. Tab shows or hides it while running.
status = false

# Real pixels through the kitty graphics protocol: "auto" uses them when the
# terminal answers the graphics probe, otherwise braille.
pixels = "auto"

# Events on their own every few minutes - a probe falling in, a transmission.
# Set to false to only have them when you press a key.
events = true
event_minutes = [4, 10]

# Short messages the cuneiform transmission carries. Each byte of the text is
# sent as one sign, so they are readable: a sign at U+12000 + n is byte n.
# transmissions = ["hello from the photon ring"]
"""

# The looks a fresh install opens with: the first five made alongside the
# widget, the rest made by playing with it. They are ordinary presets: delete
# or change them freely.
STARTERS = [
    {"name": "the 1979 plate", "palette": "ink", "glow": "match", "bloom": 0.0, "incl": 1.4,
     "mass": 1.0, "disk": 40.0, "speed": 0.12, "dust": 0.012, "lines": [], "dots": True,
     "mask": True, "vignette": False, "scanlines": False, "hud": False},
    {"name": "ember", "palette": "ember", "glow": "match", "bloom": 0.9, "incl": 1.45,
     "mass": 1.0, "disk": 36.0, "speed": 0.12, "dust": 0.012, "lines": [], "dots": True,
     "mask": True, "vignette": True, "scanlines": False, "hud": False},
    {"name": "cold isoradials", "palette": "ice", "glow": "plasma", "bloom": 0.45,
     "incl": 1.3, "mass": 1.0, "disk": 40.0, "speed": 0.1, "dust": 0.012,
     "lines": ["radii"], "line_style": "flowing", "line_colour": "blue", "line_width": 2,
     "dots": True, "mask": True, "vignette": False, "scanlines": False, "hud": False},
    {"name": "observatory", "palette": "phosphor", "glow": "match", "bloom": 0.3,
     "incl": 1.48, "mass": 1.0, "disk": 44.0, "speed": 0.12, "dust": 0.012, "lines": [],
     "dots": True, "mask": True, "vignette": True, "scanlines": True, "hud": True},
    {"name": "redshift survey", "palette": "ink", "glow": "match", "bloom": 0.0,
     "incl": 1.35, "mass": 1.0, "disk": 40.0, "speed": 0.12, "dust": 0.012,
     "lines": ["redshift"], "line_style": "pulse", "line_colour": "redshift",
     "line_width": 2, "dots": True, "mask": True, "vignette": False, "scanlines": False,
     "hud": False},
    {"name": "inferno sweep", "palette": "inferno", "glow": "copper", "bloom": 1.35,
     "incl": 1.3, "mass": 1.0, "disk": 40.0, "speed": 0.1, "dust": 0.02,
     "lines": ["flux", "redshift"], "line_style": "sweep", "line_colour": "redshift",
     "line_width": 2, "dots": True, "mask": True, "vignette": True, "scanlines": False,
     "hud": False},
    {"name": "magma copper", "palette": "magma", "glow": "copper", "bloom": 1.35,
     "incl": 1.3, "mass": 1.0, "disk": 40.0, "speed": 0.1, "dust": 0.02, "lines": [],
     "line_style": "sweep", "line_colour": "redshift", "line_width": 2, "dots": True,
     "mask": True, "vignette": True, "scanlines": False, "hud": False},
    {"name": "ice edge-on", "palette": "ice", "glow": "bw", "bloom": 1.35, "incl": 1.55,
     "mass": 1.0, "disk": 36.0, "speed": 0.1, "dust": 0.02, "lines": [],
     "line_style": "sweep", "line_colour": "redshift", "line_width": 2, "dots": True,
     "mask": True, "vignette": True, "scanlines": False, "hud": False},
    {"name": "bone, wide", "palette": "bone", "glow": "bw", "bloom": 0.45, "incl": 1.55,
     "mass": 1.0, "disk": 60.0, "speed": 0.1, "dust": 0.02, "lines": [],
     "line_style": "sweep", "line_colour": "redshift", "line_width": 2, "dots": True,
     "mask": True, "vignette": True, "scanlines": False, "hud": False},
    {"name": "copper spectrum", "palette": "copper", "glow": "bw", "bloom": 0.0,
     "incl": 1.55, "mass": 1.0, "disk": 60.0, "speed": 0.1, "dust": 0.02,
     "lines": ["redshift"], "line_style": "sweep", "line_colour": "spectrum",
     "line_width": 2, "dots": True, "mask": True, "vignette": True, "scanlines": False,
     "hud": False},
    {"name": "gameboy", "palette": "gameboy", "glow": "match", "bloom": 0.6, "incl": 1.4,
     "mass": 1.0, "disk": 40.0, "speed": 0.1, "dust": 0.02, "lines": [],
     "line_style": "sweep", "line_colour": "spectrum", "line_width": 2, "dots": True,
     "mask": True, "vignette": True, "scanlines": False, "hud": False},
]


def config_dir():
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_config():
    """The config file's settings over the defaults, writing the file if absent.

    A file that cannot be parsed is left exactly as it is and the defaults are
    used, with the problem reported back rather than raised.
    """
    defaults = tomllib.loads(DEFAULT_CONFIG.format(app=APP))
    defaults["transmissions"] = []
    path = config_dir() / "config.toml"
    if not path.exists():
        try:
            _atomic_write(path, DEFAULT_CONFIG.format(app=APP))
        except OSError:
            pass
        return defaults, None
    try:
        mine = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return defaults, f"{path}: {e}; using defaults"
    return {**defaults, **mine}, None


# ------------------------------------------------------------------- presets

def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(round(float(v), 4)) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    # JSON's string escapes are all valid in a TOML basic string.
    return json.dumps(str(v), ensure_ascii=False)


def dump_presets(presets, header=""):
    out = [header] if header else []
    for p in presets:
        out.append("[[preset]]")
        for key, v in p.items():
            out.append(f"{key} = {_value(v)}")
        out.append("")
    return "\n".join(out)


PRESETS_HEADER = """\
# Presets for {app}, managed by the app: 1-9 pick one while it runs, < and >
# step through them all, + saves the current look, X twice deletes the current
# one. Renaming or editing values by hand is fine; comments here are not kept.
"""


def presets_path():
    return config_dir() / "presets.toml"


def load_presets():
    """The saved presets, seeding the starters the first time.

    If the file is unreadable it is left alone - never overwritten - and the
    starters are used for this run only.
    """
    path = presets_path()
    if not path.exists():
        presets = [dict(p) for p in STARTERS]
        try:
            save_presets(presets)
        except OSError:
            pass
        return presets, None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [dict(p) for p in STARTERS], f"{path}: {e}; not changing it"
    return [p for p in data.get("preset", []) if isinstance(p, dict)], None


def save_presets(presets):
    _atomic_write(presets_path(), dump_presets(presets, PRESETS_HEADER.format(app=APP)))


def archive_deleted(preset):
    """Keep a deleted preset where it can be copied back from."""
    path = config_dir() / "deleted-presets.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(dump_presets([preset]) + "\n")


def unique_name(presets, stem="look"):
    taken = {p.get("name") for p in presets}
    n = len(presets) + 1
    while f"{stem} {n}" in taken:
        n += 1
    return f"{stem} {n}"


def choose(presets, start, rng=None):
    """The index of the preset to open with, or None for none."""
    if not presets or start in (None, "none", ""):
        return None
    if start == "first":
        return 0
    if start == "random":
        return (rng or random).randrange(len(presets))
    for i, p in enumerate(presets):
        if p.get("name") == start:
            return i
    return None
