#!/usr/bin/env python
"""Command line interface for luminet.

Run `luminet` with no arguments for an interactive menu, or use a subcommand
directly: `luminet render`, `luminet sweep`, `luminet gallery`, `luminet photons`.
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import time

PLOTS = ["image", "isoradials", "isoredshifts", "isofluxlines"]
COLOR_BY = ["flux", "redshift"]

DEFAULTS = {
    "mass": 1.0,
    "incl": 1.4,
    "acc": 1.0,
    "outer_edge": 40.0,
    "resolution": 100,
    "plot": "image",
    "cmap": "",
    "color_by": "flux",
}

# Parameters a sweep can vary, and how to read one value of each.
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


# ---------------------------------------------------------------- presentation

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


def styled(text, *codes):
    """Wrap text in ANSI codes, unless output is redirected."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


def in_kitty():
    return "kitty" in os.environ.get("TERM", "") or bool(os.environ.get("KITTY_WINDOW_ID"))


def show_in_kitty(path):
    """Draw an image inline in the terminal. Returns True if it was shown."""
    if not in_kitty():
        return False
    # icat draws by talking to the controlling terminal, so it can do nothing
    # useful when output is piped or captured.
    if not sys.stdout.isatty():
        return False
    kitten = shutil.which("kitten") or shutil.which("kitty")
    if not kitten:
        return False
    cmd = [kitten, "icat"] if kitten.endswith("kitten") else [kitten, "+kitten", "icat"]
    return subprocess.run(cmd + [str(path)], check=False).returncode == 0


def deliver(path, how):
    """Show a rendered file the way the user asked for."""
    print(f"wrote {styled(str(path), BOLD)}")
    if how == "kitty" and not show_in_kitty(path):
        print(f"could not draw inline; view it with:  kitten icat {path}")


# -------------------------------------------------------------------- plotting

def _pyplot(headless):
    """Import pyplot with a backend chosen before it is first imported."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    return __import__("matplotlib.pyplot", fromlist=["pyplot"])


def build(settings):
    from luminet.black_hole import BlackHole

    return BlackHole(
        mass=settings["mass"],
        incl=settings["incl"],
        acc=settings["acc"],
        outer_edge=settings["outer_edge"],
        angular_resolution=settings["resolution"],
        radial_resolution=settings["resolution"],
    )


def draw(ax, settings):
    """Draw one black hole onto an existing polar axis."""
    import numpy as np

    bh = build(settings)
    kwargs = {"cmap": settings["cmap"]} if settings["cmap"] else {}
    mode = settings["plot"]

    if mode == "image":
        radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, settings["resolution"])
        bh.plot_isoradials(direct_r=radii, ghost_r=radii,
                           color_by=settings["color_by"], ax=ax, **kwargs)
    elif mode == "isoradials":
        radii = range(6, int(settings["outer_edge"]) + 1, 2)
        bh.plot_isoradials(direct_r=radii, ghost_r=[],
                           color_by=settings["color_by"], ax=ax, **kwargs)
    elif mode == "isoredshifts":
        bh.plot_isoredshifts(ax=ax)
    elif mode == "isofluxlines":
        bh.plot_isofluxlines(ax=ax)
    else:
        raise ValueError(f"Unknown plot mode {mode!r}, expected one of {PLOTS}")


# ------------------------------------------------------------------- commands

def cmd_render(args):
    """Render a single black hole."""
    settings = {k: getattr(args, k) for k in DEFAULTS}
    headless = args.view != "window"
    plt = _pyplot(headless)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    started = time.time()
    print(f"rendering {settings['plot']} at resolution {settings['resolution']} ...",
          end="", flush=True)
    draw(ax, settings)
    print(f" {time.time() - started:.0f}s")

    if args.view == "window":
        plt.show(block=True)
        return 0

    fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    deliver(args.output, args.view)
    return 0


def cmd_sweep(args):
    """Render a grid of black holes over one or two parameters."""
    plt = _pyplot(headless=True)

    values = {}
    for name, cast in SWEPT.items():
        raw = str(getattr(args, name))
        if cast is str:
            items = [v.strip() for v in raw.split(",") if v.strip()]
            values[name] = items or [DEFAULTS[name]]
        else:
            values[name] = [cast(v) for v in raw.split(",") if v.strip()]

    varying = [name for name, vals in values.items() if len(vals) > 1]
    if len(varying) > 2:
        print(f"error: can sweep at most 2 parameters at once, got {len(varying)}: "
              f"{', '.join(varying)}. A contact sheet is 2D.", file=sys.stderr)
        return 2

    # First varying parameter runs across the columns, the second down the rows.
    if len(varying) == 2:
        col_name, row_name = varying
        cols, rows = values[col_name], values[row_name]
    elif len(varying) == 1:
        col_name, row_name = varying[0], None
        cols, rows = values[col_name], [None]
    else:
        col_name = row_name = None
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

    for (r, row_value), (c, col_value) in itertools.product(enumerate(rows), enumerate(cols)):
        settings = {name: vals[0] for name, vals in values.items()}
        if col_name:
            settings[col_name] = col_value
        if row_name:
            settings[row_name] = row_value

        ax = axes[r][c]
        ax.set_theta_zero_location("S")
        ax.set_facecolor("black")
        ax.axis("off")

        label = ", ".join(f"{n}={settings[n]}" for n in (col_name, row_name) if n)
        print(f"[{r * len(cols) + c + 1}/{total}] {label or 'default'} ...", end="", flush=True)
        try:
            draw(ax, settings)
            print(f" {time.time() - started:.0f}s")
        except Exception as e:
            # One bad cell should not cost the whole sheet.
            print(f" {styled('FAILED', BOLD)}: {type(e).__name__}: {e}")
            ax.text(0, 0, "failed", color="red", ha="center", va="center")
        if label:
            ax.set_title(label, color="white", fontsize=9, pad=6)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    print(f"{total} renders in {time.time() - started:.0f}s")
    deliver(args.output, args.view)
    return 0


def cmd_gallery(args):
    """A preset tour of the inclination range, the parameter that changes most."""
    args.incl = "0.1,0.5,0.9,1.3,1.5"
    for name in SWEPT:
        if name != "incl":
            setattr(args, name, DEFAULTS[name])
    print(styled("gallery: the accretion disk from face-on to edge-on", DIM))
    return cmd_sweep(args)


def cmd_photons(args):
    """Sample photons off the accretion disk and print them."""
    bh = build({**DEFAULTS, "incl": args.incl, "mass": args.mass,
                "acc": args.acc, "outer_edge": args.outer_edge})
    print(f"sampling {args.number} photons ...", end="", flush=True)
    started = time.time()
    photons, ghosts = bh.sample_photons(args.number, seed=args.seed)
    print(f" {time.time() - started:.0f}s")

    sample = ghosts if args.ghost else photons
    header = f"{'radius':>10} {'alpha':>9} {'b':>10} {'1+z':>9} {'flux':>12}"
    print(styled(header, BOLD))
    for p in sample[: args.number]:
        print(f"{p.radius:10.4f} {p.alpha:9.4f} {p.impact_parameter:10.4f} "
              f"{p.z_factor:9.4f} {p.flux_o:12.4e}")
    return 0


# ----------------------------------------------------------------------- menu

def ask(prompt, default, cast=str, choices=None):
    """Prompt for one value, accepting empty input as the default."""
    hint = f" [{default}]" if default != "" else " []"
    if choices:
        hint = f" ({'/'.join(choices)}) [{default}]"
    while True:
        try:
            raw = input(styled(f"  {prompt}{hint}: ", CYAN)).strip()
        except EOFError:
            print()
            raise SystemExit(0)
        if not raw:
            return default
        if choices and raw not in choices:
            print(f"    pick one of: {', '.join(choices)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print(f"    not a valid {cast.__name__}")


def ask_common(args, sweeping=False):
    """Prompt for the parameters shared by render and sweep."""
    note = " (comma-separated to sweep)" if sweeping else ""
    args.plot = ask("what to draw", DEFAULTS["plot"], str, PLOTS if not sweeping else None)
    args.incl = ask(f"inclination in radians, 0 face-on 1.57 edge-on{note}",
                    DEFAULTS["incl"], str if sweeping else float)
    args.mass = ask(f"mass{note}", DEFAULTS["mass"], str if sweeping else float)
    args.acc = ask(f"accretion rate{note}", DEFAULTS["acc"], str if sweeping else float)
    args.outer_edge = ask(f"outer edge of the disk{note}",
                          DEFAULTS["outer_edge"], str if sweeping else float)
    args.resolution = ask(f"resolution, higher is slower{note}",
                          DEFAULTS["resolution"], str if sweeping else int)
    args.color_by = ask(f"colour by{note}", DEFAULTS["color_by"],
                        str, COLOR_BY if not sweeping else None)
    args.cmap = ask(f"colormap, blank for the default{note}", DEFAULTS["cmap"], str)
    return args


def ask_view(args, default_output):
    if in_kitty():
        args.view = ask("view", "kitty", str, ["kitty", "window", "file"])
    else:
        args.view = ask("view", "window", str, ["window", "file"])
    if args.view != "window":
        args.output = ask("write to", default_output, str)
    return args


def menu():
    """The interactive menu shown when luminet is run with no arguments."""
    if not sys.stdin.isatty():
        print("luminet: no menu without a terminal. Try `luminet --help`.", file=sys.stderr)
        return 2

    items = [
        ("1", "render", "one black hole, in a window or a file"),
        ("2", "sweep", "a grid of variations as one contact sheet"),
        ("3", "gallery", "preset tour of the inclination range"),
        ("4", "photons", "sample photons off the accretion disk"),
        ("5", "help", "the full list of flags"),
        ("q", "quit", ""),
    ]

    while True:
        print()
        print(styled("  luminet", BOLD) + styled("  Schwarzschild black hole renderer", DIM))
        print()
        for key, name, blurb in items:
            print(f"   {styled(key, BOLD)}  {name:<9} {styled(blurb, DIM)}")
        print()

        try:
            choice = input(styled("  choose: ", CYAN)).strip().lower()
        except EOFError:
            print()
            return 0

        if choice in ("q", "quit", "exit"):
            return 0

        args = argparse.Namespace(**DEFAULTS)
        args.dpi, args.tile, args.seed, args.ghost = 110, 3.0, None, False

        print()
        if choice in ("1", "render"):
            ask_common(args, sweeping=False)
            ask_view(args, "luminet.png")
            print()
            cmd_render(args)
        elif choice in ("2", "sweep"):
            ask_common(args, sweeping=True)
            args.tile = ask("tile size in inches", 3.0, float)
            ask_view(args, "sweep.png")
            print()
            cmd_sweep(args)
        elif choice in ("3", "gallery"):
            ask_view(args, "gallery.png")
            args.tile = 3.0
            print()
            cmd_gallery(args)
        elif choice in ("4", "photons"):
            args.number = ask("how many photons", 10, int)
            args.incl = ask("inclination in radians", DEFAULTS["incl"], float)
            args.seed = ask("seed, blank for random", "", str)
            args.seed = int(args.seed) if str(args.seed).strip() else None
            args.ghost = ask("which image", "direct", str, ["direct", "ghost"]) == "ghost"
            print()
            cmd_photons(args)
        elif choice in ("5", "help", "h", "?"):
            build_parser().print_help()
        else:
            print(f"  no such option: {choice}")


# ------------------------------------------------------------------- argparse

def add_common(p, sweeping):
    note = " (comma-separated to sweep)" if sweeping else ""
    kind = str if sweeping else None
    p.add_argument("--mass", type=kind or float, default=DEFAULTS["mass"],
                   help=f"black hole mass in natural units, G=c=1{note}")
    p.add_argument("--incl", type=kind or float, default=DEFAULTS["incl"],
                   help=f"observer inclination in radians: 0 face-on, 1.57 edge-on{note}")
    p.add_argument("--acc", type=kind or float, default=DEFAULTS["acc"],
                   help=f"accretion rate; scales brightness only{note}")
    p.add_argument("--outer-edge", type=kind or float, default=DEFAULTS["outer_edge"],
                   help=f"outer edge of the accretion disk{note}")
    p.add_argument("--resolution", type=kind or int, default=DEFAULTS["resolution"],
                   help=f"angular and radial resolution; higher is slower{note}")
    p.add_argument("--plot", default=DEFAULTS["plot"],
                   help=f"what to draw, one of {', '.join(PLOTS)}{note}")
    p.add_argument("--cmap", default=DEFAULTS["cmap"],
                   help=f"matplotlib colormap, e.g. inferno{note}")
    p.add_argument("--color-by", default=DEFAULTS["color_by"],
                   help=f"one of {', '.join(COLOR_BY)}{note}")


def add_view(p, default_output, allow_window):
    choices = ["kitty", "file"] + (["window"] if allow_window else [])
    p.add_argument("--view", default="kitty" if in_kitty() else "file", choices=choices,
                   help="kitty draws inline in the terminal; window opens a plot window")
    p.add_argument("-o", "--output", default=default_output, help="where to write the image")
    p.add_argument("--dpi", type=int, default=110, help="resolution of the written image")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="luminet",
        description="Simulate and visualise Schwarzschild black holes, after Luminet (1979). "
                    "Run with no arguments for an interactive menu.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("render", help="render a single black hole")
    add_common(p, sweeping=False)
    add_view(p, "luminet.png", allow_window=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("sweep", help="render a grid of variations as one contact sheet")
    add_common(p, sweeping=True)
    add_view(p, "sweep.png", allow_window=False)
    p.add_argument("--tile", type=float, default=3.0, help="size of each tile in inches")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("gallery", help="preset tour of the inclination range")
    add_view(p, "gallery.png", allow_window=False)
    p.add_argument("--tile", type=float, default=3.0, help="size of each tile in inches")
    p.set_defaults(func=cmd_gallery)

    p = sub.add_parser("photons", help="sample photons off the accretion disk")
    p.add_argument("-n", "--number", type=int, default=10, help="how many to sample")
    p.add_argument("--mass", type=float, default=DEFAULTS["mass"])
    p.add_argument("--incl", type=float, default=DEFAULTS["incl"])
    p.add_argument("--acc", type=float, default=DEFAULTS["acc"])
    p.add_argument("--outer-edge", type=float, default=DEFAULTS["outer_edge"])
    p.add_argument("--seed", type=int, default=None, help="seed, for reproducible sampling")
    p.add_argument("--ghost", action="store_true", help="show the ghost image instead")
    p.set_defaults(func=cmd_photons)

    sub.add_parser("menu", help="the interactive menu")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    if not argv or argv == ["menu"]:
        return menu()

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
