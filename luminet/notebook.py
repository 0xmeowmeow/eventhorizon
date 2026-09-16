"""A lab notebook for renders.

Every render is recorded here with the settings that produced it, so a later run
can start from an earlier one instead of from the defaults, and so an observation
can be kept next to the parameters it came from.

The notebook stores settings and notes, not pictures. An image is fully determined
by its settings, so any past run can be drawn again from its record; that is also
why the record, and not the PNG, is the thing worth keeping.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# One known home, next to the project, so there is nothing to remember or set up.
NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebook"
RUNS_FILE = NOTEBOOK_DIR / "runs.json"

# Settings recorded for every run. Anything not listed is presentation, not physics.
RECORDED = [
    "plot", "mass", "incl", "acc", "outer_edge", "resolution",
    "color_by", "cmap", "line_color", "lw", "radii", "ghost_radii",
]


# What each recorded setting should be read as. Values reach `vary` as text.
TYPES = {
    "plot": str, "mass": float, "incl": float, "acc": float, "outer_edge": float,
    "resolution": int, "color_by": str, "cmap": str, "line_color": str,
    "lw": float, "radii": str, "ghost_radii": str,
}


def cast(name, value):
    """Read one setting's value as its proper type, leaving it alone if it cannot be."""
    if value is None:
        return None
    try:
        return TYPES.get(name, str)(value)
    except (TypeError, ValueError):
        return value


def load():
    """Every run recorded so far, oldest first."""
    if not RUNS_FILE.exists():
        return []
    try:
        return json.loads(RUNS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save(runs):
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2) + "\n")


def image_path(run_id):
    return NOTEBOOK_DIR / f"{run_id:03d}.png"


def record(settings, parent=None, note=""):
    """Add a run, and return it."""
    runs = load()
    run = {
        "id": len(runs) + 1,
        "when": datetime.now().isoformat(timespec="seconds"),
        "parent": parent,
        "note": note,
        "settings": {k: settings.get(k) for k in RECORDED},
    }
    runs.append(run)
    save(runs)
    return run


def get(run_id):
    """One run by id, or None."""
    for run in load():
        if run["id"] == run_id:
            return run
    return None


def latest():
    runs = load()
    return runs[-1] if runs else None


def annotate(run_id, note):
    """Attach an observation to a run. Returns the run, or None if there is no such run."""
    runs = load()
    for run in runs:
        if run["id"] == run_id:
            run["note"] = note
            save(runs)
            return run
    return None


def differences(a, b):
    """Which settings differ between two runs, as {name: (before, after)}."""
    out = {}
    for key in RECORDED:
        before, after = a.get(key), b.get(key)
        if str(before) != str(after):
            out[key] = (before, after)
    return out


def describe(run, compared_to=None):
    """A one-line summary of a run, showing only what makes it distinctive."""
    settings = run["settings"]
    if compared_to is not None:
        changed = differences(compared_to["settings"], settings)
        if changed:
            return ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in changed.items())
        return "no change"
    parts = [f"{settings['plot']}"]
    for key in ("incl", "mass", "acc", "outer_edge"):
        parts.append(f"{key}={settings[key]}")
    return ", ".join(parts)


def lineage(run_id):
    """A run and everything it was derived from, oldest ancestor first."""
    chain, seen = [], set()
    run = get(run_id)
    while run and run["id"] not in seen:
        seen.add(run["id"])
        chain.append(run)
        run = get(run["parent"]) if run.get("parent") else None
    return list(reversed(chain))
