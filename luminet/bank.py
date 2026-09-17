"""A bank of lensing maps across inclination, so tilting can be continuous.

Solving a map takes over a second, so tilting one solve at a time can only ever
step. But a map changes smoothly with inclination: blending the two maps either
side of an inclination, 0.05 radians apart, puts the light within half a unit of
where a direct solve puts it on screen - a few braille dots at most - and nothing
changes whether light arrives at all. So maps are solved once, at every 0.05
from 0.05 to 1.55, cached on disk, and blended while the view moves. Where the
view comes to rest on a banked inclination the map is exact.

Mass costs nothing extra. The geometry scales exactly with mass - a map at mass
2 is the mass-1 map with radii and impact parameters doubled, to 1e-11 - so the
bank is solved at mass 1 and scaled.

Every map in the bank shares one set of radii, a disk out to 1000 M spaced
geometrically, wide enough to fill a tall window nearly edge-on. Shared radii
are what let a parcel of gas keep its ring as the view tilts.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VERSION = 1
STEP = 0.05
INCLINATIONS = tuple(round(STEP * k, 2) for k in range(1, 32))     # 0.05 .. 1.55
OUTER = 1000.0
RINGS = 120
ANGLES = 180


def cache_dir():
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    from luminet.config import APP

    return base / APP / f"maps-v{VERSION}-r{RINGS}-a{ANGLES}-o{OUTER:.0f}"


def path_for(incl):
    return cache_dir() / f"incl-{incl:.2f}.npz"


def solve(incl):
    """Solve one banked map at mass 1 and store it."""
    from luminet import spin

    mapping = spin.lensing_map(
        {"mass": 1.0, "incl": float(incl), "acc": 1.0, "outer_edge": OUTER},
        n_rings=RINGS, n_angles=ANGLES, spacing="log")
    out = path_for(incl)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".writing.npz")
    np.savez(tmp, radii=mapping["radii"], angles=mapping["angles"],
             b0=mapping["tables"][0][0], z0=mapping["tables"][0][1],
             b1=mapping["tables"][1][0], z1=mapping["tables"][1][1])
    os.replace(tmp, out)          # never leave a half-written map to be read


def build(start=1.4):
    """Solve every missing map, nearest the given inclination first."""
    for incl in sorted(INCLINATIONS, key=lambda i: abs(i - start)):
        if not path_for(incl).exists():
            solve(incl)


def start_building(start):
    """Fill the bank in a background process, at low priority."""
    if all(path_for(i).exists() for i in INCLINATIONS):
        return None
    lock = cache_dir() / "building.pid"
    try:
        pid = int(lock.read_text())
        os.kill(pid, 0)
        return None                            # another one is already at it
    except (OSError, ValueError):
        pass
    cache_dir().mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "luminet.cli", "_bank", "--start", f"{start:.2f}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=lambda: os.nice(10), start_new_session=True)
    lock.write_text(str(proc.pid))
    return proc


class Bank:
    """The cached maps, loaded as they appear, blended on request."""

    def __init__(self):
        self.loaded = {}

    def _load(self, incl):
        if incl in self.loaded:
            return self.loaded[incl]
        path = path_for(incl)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                self.loaded[incl] = {k: data[k] for k in data.files}
        except (OSError, ValueError):
            return None
        return self.loaded[incl]

    def count(self):
        return sum(1 for i in INCLINATIONS if path_for(i).exists())

    def neighbours(self, incl):
        incl = float(np.clip(incl, INCLINATIONS[0], INCLINATIONS[-1]))
        k = int(np.clip(np.floor((incl - INCLINATIONS[0]) / STEP), 0, len(INCLINATIONS) - 2))
        lo, hi = INCLINATIONS[k], INCLINATIONS[k + 1]
        f = (incl - lo) / (hi - lo)
        if f < 1e-6:
            return lo, lo, 0.0
        if f > 1 - 1e-6:
            return hi, hi, 0.0
        return lo, hi, f

    def ready_for(self, incl):
        lo, hi, _ = self.neighbours(incl)
        return self._load(lo) is not None and self._load(hi) is not None

    def mapping(self, incl, mass=1.0, acc=1.0):
        """A map for any inclination in range, blended and scaled to the mass."""
        lo, hi, f = self.neighbours(incl)
        a, b = self._load(lo), self._load(hi)
        if a is None or b is None:
            return None

        def blend(key):
            x, y = a[key], b[key]
            if f == 0.0:
                return x.copy()
            out = (1 - f) * x + f * y
            # Where only one neighbour has light, take that one rather than
            # leave a hole; where neither does, there is none.
            only_x = np.isfinite(x) & ~np.isfinite(y)
            only_y = np.isfinite(y) & ~np.isfinite(x)
            out[only_x] = x[only_x]
            out[only_y] = y[only_y]
            return out

        mass = float(mass)
        tables = {0: (blend("b0") * mass, blend("z0")),
                  1: (blend("b1") * mass, blend("z1"))}
        bh = SimpleNamespace(mass=mass, acc=float(acc),
                             critical_b=3.0 * np.sqrt(3.0) * mass,
                             incl=float(incl))
        return {"bh": bh, "radii": a["radii"] * mass, "angles": a["angles"],
                "tables": tables, "banked": True}
