#!/usr/bin/env python
"""Render a grid of black holes across a range of settings, as one contact sheet.

Give any parameter a comma-separated list to sweep it. Sweeping one parameter
gives a row, sweeping two gives a grid; anything left alone keeps its default.

Examples:
    python sweep.py --incl 0.2,0.6,1.0,1.4
    python sweep.py --incl 0.4,0.8,1.2 --acc 0.5,1,2
    python sweep.py --incl 0.6,1.0,1.4 --cmap Greys_r,inferno,RdBu_r
    python sweep.py --incl 0.6,1.4 --color-by flux,redshift
    python sweep.py --incl 0.2,0.6,1.0,1.4 --plot isoradials --no-show
"""

import argparse
import itertools
import time

import matplotlib

matplotlib.use("Agg")  # a contact sheet is written to a file, never shown live
import matplotlib.pyplot as plt
import numpy as np

from luminet.black_hole import BlackHole

# Parameters that can be swept, and how to read one value of each.
SWEPT = {
    "mass": float,
    "incl": float,
    "acc": float,
    "outer_edge": float,
    "resolution": int,
    "plot": str,
    "cmap": str,
    "color_by": str,
}

DEFAULTS = {
    "mass": 1.0,
    "incl": 1.4,
    "acc": 1.0,
    "outer_edge": 40.0,
    "resolution": 100,
    "plot": "image",
    "cmap": None,
    "color_by": "flux",
}

PLOTS = ["image", "isoradials", "isoredshifts", "isofluxlines"]


def render_one(ax, settings):
    """Draw a single black hole onto an existing polar axis."""
    bh = BlackHole(
        mass=settings["mass"],
        incl=settings["incl"],
        acc=settings["acc"],
        outer_edge=settings["outer_edge"],
        angular_resolution=settings["resolution"],
        radial_resolution=settings["resolution"],
    )

    kwargs = {}
    if settings["cmap"]:
        kwargs["cmap"] = settings["cmap"]

    mode = settings["plot"]
    if mode == "image":
        radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, settings["resolution"])
        bh.plot_isoradials(
            direct_r=radii, ghost_r=radii, color_by=settings["color_by"], ax=ax, **kwargs
        )
    elif mode == "isoradials":
        radii = range(6, int(settings["outer_edge"]) + 1, 2)
        bh.plot_isoradials(
            direct_r=radii, ghost_r=[], color_by=settings["color_by"], ax=ax, **kwargs
        )
    elif mode == "isoredshifts":
        bh.plot_isoredshifts(ax=ax)
    elif mode == "isofluxlines":
        bh.plot_isofluxlines(ax=ax)
    else:
        raise ValueError(f"Unknown plot mode {mode!r}, expected one of {PLOTS}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mass", default="1.0", help="black hole mass (natural units)")
    parser.add_argument("--incl", default="1.4",
                        help="observer inclination in radians: 0 face-on, pi/2 edge-on")
    parser.add_argument("--acc", default="1.0", help="accretion rate; scales brightness only")
    parser.add_argument("--outer-edge", default="40", help="outer edge of the accretion disk")
    parser.add_argument("--resolution", default="100",
                        help="angular and radial resolution. Lower is faster and rougher")
    parser.add_argument("--plot", default="image", help=f"what to draw, one of {PLOTS}")
    parser.add_argument("--cmap", default="", help="matplotlib colormap, e.g. inferno")
    parser.add_argument("--color-by", default="flux", help="'flux' or 'redshift'")
    parser.add_argument("-o", "--output", default="sweep.png", help="where to write the sheet")
    parser.add_argument("--tile", type=float, default=3.0, help="size of each tile in inches")
    parser.add_argument("--dpi", type=int, default=110, help="resolution of the sheet")
    parser.add_argument("--no-show", action="store_true",
                        help="do not display the sheet in kitty afterwards")
    args = parser.parse_args()

    # Parse each sweepable parameter into a list of values.
    values = {}
    for name, cast in SWEPT.items():
        raw = getattr(args, name)
        if cast is str:
            items = [v.strip() for v in raw.split(",") if v.strip()]
            values[name] = items or [DEFAULTS[name]]
        else:
            values[name] = [cast(v) for v in raw.split(",") if v.strip()]

    varying = [name for name, vals in values.items() if len(vals) > 1]
    if len(varying) > 2:
        parser.error(
            f"can sweep at most 2 parameters at once, got {len(varying)}: "
            f"{', '.join(varying)}. A sheet is 2D."
        )

    # Lay the grid out: first varying parameter across columns, second down rows.
    if len(varying) == 2:
        col_name, row_name = varying
        cols, rows = values[col_name], values[row_name]
    elif len(varying) == 1:
        col_name, row_name = varying[0], None
        cols, rows = values[col_name], [None]
    else:
        col_name, row_name = None, None
        cols, rows = [None], [None]

    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(args.tile * len(cols), args.tile * len(rows)),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    total = len(rows) * len(cols)
    started = time.time()

    for (r, row_value), (c, col_value) in itertools.product(
        enumerate(rows), enumerate(cols)
    ):
        settings = {name: vals[0] for name, vals in values.items()}
        if col_name:
            settings[col_name] = col_value
        if row_name:
            settings[row_name] = row_value

        ax = axes[r][c]
        ax.set_theta_zero_location("S")
        ax.set_facecolor("black")
        ax.axis("off")

        n = r * len(cols) + c + 1
        label = ", ".join(
            f"{name}={settings[name]}" for name in (col_name, row_name) if name
        )
        print(f"[{n}/{total}] {label or 'default'} ...", end="", flush=True)

        try:
            render_one(ax, settings)
            print(f" {time.time() - started:.0f}s")
        except Exception as e:
            # One bad cell should not cost the whole sheet.
            print(f" FAILED: {type(e).__name__}: {e}")
            ax.text(0, 0, "failed", color="red", ha="center", va="center")

        if label:
            ax.set_title(label, color="white", fontsize=9, pad=6)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    print(f"wrote {args.output} ({total} renders in {time.time() - started:.0f}s)")

    if not args.no_show:
        show_in_kitty(args.output)


def show_in_kitty(path):
    """Display an image inline, if we are running inside kitty."""
    import os
    import shutil
    import subprocess
    import sys

    if "kitty" not in os.environ.get("TERM", "") and not os.environ.get("KITTY_WINDOW_ID"):
        print("not running in kitty; open the file yourself to view it")
        return

    # icat draws by talking to the controlling terminal, so it cannot do
    # anything useful when output is piped or captured.
    if not sys.stdout.isatty():
        print(f"output is not a terminal; view it with:  kitten icat {path}")
        return

    kitten = shutil.which("kitten") or shutil.which("kitty")
    if not kitten:
        print("kitty found no 'kitten' binary; open the file yourself to view it")
        return

    cmd = [kitten, "icat"] if kitten.endswith("kitten") else [kitten, "+kitten", "icat"]
    subprocess.run(cmd + [path], check=False)


if __name__ == "__main__":
    main()
