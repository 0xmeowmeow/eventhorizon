"""Bake eventhorizon views into data a web page can draw without the physics.

The site version draws one settled picture per visit, so it does not need the
lensing maps themselves - only what the widget measures from them for a still
view: the brightness each patch of the picture should have (DotField.target),
where the near side of the disk passes in front of the shadow (DotField.front),
the shadow's radius, and the isoradials. The page stipples dots from the
brightness with its own seed.

    python tools/web_export.py > horizon-data.json

Each geometry is an inclination and a disk size. Brightness is kept to 4 bits
a patch, which is finer than the dots can show.
"""

import base64
import json
import sys

import numpy as np

from luminet import effects, spin
from luminet.bank import Bank

GRID_W, GRID_H = 260, 114          # 4 px patches on a 1040 x 455 canvas
GEOMETRIES = {"plate": (1.40, 40.0), "tilted": (1.30, 40.0), "edge": (1.55, 36.0)}
DIRECT_RADII = (6, 10, 15, 20)
GHOST_RADII = (6, 20, 50, 100)


def b64(raw):
    return base64.b64encode(bytes(raw)).decode()


def export(incl, disk):
    mapping = Bank().mapping(incl, 1.0)
    orders = (0, 1)
    # The widget's own framing (Live.framing) and pixel-mode field.
    radius = max(disk * 0.85, float(mapping["radii"][0]) * 1.5)
    rx, ry = spin.reach(mapping, orders, max_radius=radius)
    extent = (rx * 0.78, ry * 0.95)
    rates = spin.true_rates(mapping["radii"], 1.0, 0.12)
    parcels = spin.Parcels(mapping["radii"], count=int(GRID_W * GRID_H * 2.5), seed=0,
                           clumps=0, spread="log")
    from luminet import fast

    projector = fast.Projector(mapping, orders) if fast.available else None
    field = spin.DotField(mapping, parcels, GRID_W, GRID_H, extent, rates, orders,
                          gamma=0.6, projector=projector, floor=0.012, cell_aspect=1.0)
    target = np.clip(field.target, 0, 1)
    nib = np.rint(target * 15).astype(np.uint8).ravel()
    if nib.size % 2:
        nib = np.append(nib, 0)
    packed = (nib[0::2] << 4) | nib[1::2]
    front = np.packbits((field.front > 0).ravel())

    canvas = type("C", (), {"ext_x": field.ext_x, "ext_y": field.ext_y})()
    lines = []
    angles = np.linspace(0, 2 * np.pi, 361)
    for order, radii in ((0, DIRECT_RADII), (1, GHOST_RADII)):
        for r in radii:
            b, _ = effects.project(mapping, np.full_like(angles, float(r) * 1.0001), angles, order)
            xf, yf = effects.to_screen(canvas, b, angles)
            pts = np.where(np.isfinite(b)[:, None], np.round(np.stack([xf, yf], -1) * 10000), -32768)
            pts = np.clip(pts, -32768, 32767).astype("<i2")
            lines.append({"order": order, "r": r, "pts": b64(pts.tobytes())})
    crit = float(mapping["bh"].critical_b)
    return {"incl": incl, "disk": disk, "w": GRID_W, "h": GRID_H,
            "target": b64(packed), "front": b64(front),
            "hole": [crit / (2 * field.ext_x), crit / (2 * field.ext_y)], "lines": lines}


if __name__ == "__main__":
    out = {name: export(*g) for name, g in GEOMETRIES.items()}
    json.dump(out, sys.stdout, separators=(",", ":"))
