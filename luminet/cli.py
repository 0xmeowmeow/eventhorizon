#!/usr/bin/env python
"""Command line interface for luminet.

Run `luminet` with no arguments for an interactive menu, or use a subcommand
directly: `luminet render`, `luminet sweep`, `luminet gallery`, `luminet photons`.
"""

import argparse
import itertools
import json
import os
import shutil
import shutil
import subprocess
import sys
import time
from pathlib import Path

from luminet import guide, notebook

PLOTS = ["image", "lines", "flat", "isoradials", "isoredshifts", "isofluxlines"]
COLOR_BY = ["flux", "redshift"]

# Radii behind the line drawing in the docs. Ghost radii may exceed the disk's
# outer edge: light from far material still wraps into view behind the hole.
LINE_RADII = [6, 10, 15, 20]
LINE_GHOST_RADII = [6, 20, 50, 100]

DEFAULTS = {
    "mass": 1.0,
    "incl": 1.4,
    "acc": 1.0,
    "outer_edge": 40.0,
    "resolution": 100,
    "plot": "image",
    "cmap": "",
    "color_by": "flux",
    "line_color": "white",
    "lw": 1.0,
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
    "line_color": str,
    "lw": float,
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


def parse_radii(raw, fallback):
    """Read a comma-separated list of radii, falling back when none is given."""
    if raw is None or not str(raw).strip():
        return list(fallback)
    return [float(v) for v in str(raw).split(",") if v.strip()]


def draw(ax, settings):
    """Draw one black hole onto an existing polar axis."""
    import numpy as np

    bh = build(settings)
    kwargs = {"cmap": settings["cmap"]} if settings["cmap"] else {}
    mode = settings["plot"]

    if mode == "flat":
        # The Newtonian limit: what the disk would look like if light were not
        # bent. Each ring is just a tilted circle, i.e. an ellipse, so the far
        # side stays behind the hole and there is no second image.
        from luminet import black_hole_math as bhmath

        angles = np.linspace(0, 2 * np.pi, max(int(settings["resolution"]), 60))
        for r in parse_radii(settings.get("radii"), LINE_RADII):
            b = [bhmath.ellipse(r, a, settings["incl"]) for a in angles]
            ax.plot(angles, b, color=settings["line_color"], lw=settings["lw"])
    elif mode == "lines":
        # The classic Luminet line drawing: plain isoradials, direct image above
        # and ghost image below. plot_isoradials always builds a flux gradient,
        # but an explicit `colors` overrides it with one flat colour.
        bh.plot_isoradials(
            direct_r=parse_radii(settings.get("radii"), LINE_RADII),
            ghost_r=parse_radii(settings.get("ghost_radii"), LINE_GHOST_RADII),
            ax=ax, colors=settings["line_color"], lw=settings["lw"],
        )
    elif mode == "image":
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
    settings["radii"] = getattr(args, "radii", "")
    settings["ghost_radii"] = getattr(args, "ghost_radii", "")
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

    run = notebook.record(settings, parent=getattr(args, "parent", None))
    print(f"recorded as run {styled(str(run['id']), BOLD)}"
          f"   (luminet vary {run['id']} --incl ... to build on it)")

    if args.view == "window":
        plt.show(block=True)
        return 0

    path = notebook.image_path(run["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    if args.output:
        fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    deliver(args.output or path, args.view)
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
        settings["radii"] = getattr(args, "radii", "")
        settings["ghost_radii"] = getattr(args, "ghost_radii", "")
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

    base = {name: vals[0] for name, vals in values.items()}
    swept = "; ".join(f"{n} = {', '.join(str(v) for v in values[n])}"
                      for n in (col_name, row_name) if n)
    run = notebook.record(base, note=f"sweep of {swept}" if swept else "sweep")
    path = notebook.image_path(run["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    if args.output:
        fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    print(f"{total} renders in {time.time() - started:.0f}s")
    print(f"recorded as run {styled(str(run['id']), BOLD)}"
          f"   (luminet vary {run['id']} --incl ... to build on it)")
    deliver(args.output or path, args.view)
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


def cmd_explain(args):
    """Print what the controls are and what they do."""
    print(guide.explain(args.control))
    return 0


def cmd_log(args):
    """List what has been rendered so far."""
    runs = notebook.load()
    if not runs:
        print("nothing recorded yet. `luminet render` records every run it draws.")
        return 0
    for run in runs[-args.limit:]:
        parent = notebook.get(run["parent"]) if run.get("parent") else None
        summary = notebook.describe(run, compared_to=parent)
        lead = f"  {styled(str(run['id']).rjust(3), BOLD)}"
        if parent:
            print(f"{lead}  from {parent['id']}  {summary}")
        else:
            print(f"{lead}  {summary}")
        if run.get("note"):
            print(f"       {styled(run['note'], DIM)}")
    print(f"\n  {styled('luminet show <n>', BOLD)} to redraw one, "
          f"{styled('luminet vary <n> --incl 0.9', BOLD)} to change one thing,\n"
          f"  {styled('luminet note <n> \'...\'', BOLD)} to keep an observation with it.")
    return 0


def cmd_show(args):
    """Draw an earlier run again, from its recorded settings."""
    run = notebook.get(args.run) if args.run else notebook.latest()
    if run is None:
        print("no such run. `luminet log` lists them.", file=sys.stderr)
        return 2

    for step in notebook.lineage(run["id"]):
        parent = notebook.get(step["parent"]) if step.get("parent") else None
        print(f"  run {step['id']}: {notebook.describe(step, compared_to=parent)}")
        if step.get("note"):
            print(f"    {styled(step['note'], DIM)}")

    plt = _pyplot(headless=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    draw(ax, run["settings"])
    path = notebook.image_path(run["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    deliver(path, args.view)
    return 0


def cmd_vary(args):
    """Repeat an earlier run with one thing changed, beside the original."""
    base = notebook.get(args.run) if args.run else notebook.latest()
    if base is None:
        print("nothing recorded yet. Run `luminet render` first.", file=sys.stderr)
        return 2

    overrides = {k: notebook.cast(k, v) for k, v in vars(args).items()
                 if k in notebook.RECORDED and v is not None}
    if not overrides:
        print("give at least one thing to change, for example --incl 0.9.\n"
              "`luminet explain` lists what there is to change.", file=sys.stderr)
        return 2

    before = dict(base["settings"])
    after = {**before, **overrides}
    changed = notebook.differences(before, after)
    label = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in changed.items())

    plt = _pyplot(headless=True)
    fig, axes = plt.subplots(1, 2, figsize=(args.tile * 2, args.tile),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    fig.patch.set_facecolor("black")

    started = time.time()
    for ax, settings, title in ((axes[0][0], before, f"run {base['id']}, unchanged"),
                                (axes[0][1], after, label)):
        ax.set_theta_zero_location("S")
        ax.set_facecolor("black")
        ax.axis("off")
        print(f"  drawing {title} ...", end="", flush=True)
        draw(ax, settings)
        print(f" {time.time() - started:.0f}s")
        ax.set_title(title, color="white", fontsize=9, pad=6)

    run = notebook.record(after, parent=base["id"])
    path = notebook.image_path(run["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    print(f"recorded as run {styled(str(run['id']), BOLD)}, from run {base['id']}")
    deliver(path, args.view)
    return 0


def cmd_note(args):
    """Keep an observation with a run."""
    run = notebook.annotate(args.run, args.text)
    if run is None:
        print("no such run. `luminet log` lists them.", file=sys.stderr)
        return 2
    print(f"noted on run {run['id']}: {run['note']}")
    return 0


def cmd_draw(args):
    """Render one image from JSON settings. Used by the TUI, which renders out of process."""
    settings = {**DEFAULTS, "radii": "", "ghost_radii": "", **json.loads(args.settings)}
    plt = _pyplot(headless=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ax.set_theta_zero_location("S")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw(ax, settings)
    fig.savefig(args.output, dpi=args.dpi, facecolor="black", bbox_inches="tight")
    return 0


def cmd_tui(args):
    """Open the interactive instrument."""
    from luminet import tui

    return tui.run()


def cmd_animate(args):
    """Pre-render a loop between two settings."""
    from luminet import animate

    if args.control:
        base = notebook.get(args.run) if args.run else notebook.latest()
        settings = dict(base["settings"]) if base else {**DEFAULTS, "radii": "", "ghost_radii": ""}
        start = {**settings, args.control: notebook.cast(args.control, args.start)}
        end = {**settings, args.control: notebook.cast(args.control, args.end)}
        what = f"{args.control} {args.start} to {args.end}"
    else:
        if args.between:
            try:
                a, b = animate.endpoints_from_runs(*args.between)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            start, end = a, b
            what = f"run {args.between[0]} to run {args.between[1]}"
        else:
            runs = notebook.load()
            if len(runs) < 2:
                print("nothing to animate between yet. Render two things, or name a\n"
                      "control and two values: luminet animate incl 0.2 1.5",
                      file=sys.stderr)
                return 2
            start, end = dict(runs[-2]["settings"]), dict(runs[-1]["settings"])
            what = f"run {runs[-2]['id']} to run {runs[-1]['id']}"

    if args.final:
        start["resolution"] = end["resolution"] = max(int(end.get("resolution") or 100), 300)
        args.frames = max(args.frames, 60)
        args.dpi = max(args.dpi, 200)

    print(f"animating {what}, {args.frames} frames"
          f"{' then back' if not args.once else ''} at {args.fps}fps")
    try:
        out, total = animate.build(
            start, end, args.output, frames=args.frames, fps=args.fps, dpi=args.dpi,
            bounce=not args.once, jobs=args.jobs, keep_frames=args.keep_frames,
        )
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    seconds = total / args.fps
    print(f"wrote {styled(str(out), BOLD)}  {total} frames, {seconds:.1f}s"
          f"{', loops seamlessly' if not args.once else ''}")
    if args.view == "kitty" and not show_in_kitty(out):
        print(f"view it with:  kitten icat {out}")
    return 0


def _terminal_grid(width_cells, height_cells, encoding):
    """How many samples fit, given how each glyph subdivides its cell."""
    if encoding == "sextant":
        return width_cells * 2, height_cells * 3
    return width_cells, height_cells * 2          # half-block


def cmd_term(args):
    """Draw straight into the terminal, using sub-cell glyphs rather than ASCII."""
    import numpy as np

    from luminet import cells, fields

    size = shutil.get_terminal_size((96, 30))
    width_cells = args.width or max(20, size.columns - 2)
    height_cells = args.height or max(10, size.lines - 6)
    width, height = _terminal_grid(width_cells, height_cells, args.encoding)

    settings = {**DEFAULTS, "radii": "", "ghost_radii": ""}
    run = notebook.get(args.run) if args.run else notebook.latest()
    if run:
        settings.update(run["settings"])
    for name in ("incl", "mass", "acc", "outer_edge"):
        if getattr(args, name, None) is not None:
            settings[name] = getattr(args, name)

    print(f"sampling the observer plane ...", end="", flush=True)
    started = time.time()
    orders = fields.compute(settings, n_radii=args.radii_samples, n_angles=args.angle_samples)
    grid = fields.rasterise(orders, width, height)
    print(f" {time.time() - started:.0f}s   "
          f"{width}x{height} samples in {width_cells}x{height_cells} cells "
          f"({args.encoding})")

    if args.encoding == "sextant":
        # Two colours per cell, so the field has to become a yes-or-no: either
        # which image order this is, or whether it is bright enough to light.
        if args.channel == "order":
            lit = np.nan_to_num(grid["order"], nan=-1.0) > 0.5
        else:
            lit = np.nan_to_num(fields.normalise(grid, "flux")) > args.threshold
        print(cells.sextant(lit))
    else:
        print(cells.half_block(cells.colourise(grid, args.channel)))

    filled = int(grid["mask"].sum())
    print(f"{filled} of {width * height} samples carry light "
          f"({100 * filled / (width * height):.0f}%)")
    recorded = notebook.record(settings,
                               note=f"term, {args.channel} channel, {args.encoding} cells")
    print(f"recorded as run {recorded['id']}")
    return 0


def cmd_spin(args):
    """Animate the disk turning, in the terminal."""
    import numpy as np

    from luminet import cells, fields, spin

    size = shutil.get_terminal_size((96, 30))
    width_cells = args.width or max(20, size.columns - 2)
    height_cells = args.height or max(10, size.lines - 4)
    width, height = width_cells, height_cells * 2      # half-block

    settings = {**DEFAULTS, "radii": "", "ghost_radii": ""}
    run = notebook.get(args.run) if args.run else None
    if run:
        settings.update(run["settings"])
    for name in ("incl", "mass", "acc", "outer_edge"):
        if getattr(args, name, None) is not None:
            settings[name] = getattr(args, name)

    orders = (0,) if args.no_ghost else (0, 1)
    print(f"solving the lensing map once ({args.rings} rings) ...", end="", flush=True)
    started = time.time()
    mapping = spin.lensing_map(settings, n_rings=args.rings, n_angles=args.angles,
                               orders=orders)
    extent = spin.reach(mapping, orders)
    turns = spin.ring_turns(mapping["radii"], float(settings["mass"]), args.frames,
                            max_turns=args.max_turns)
    parcels = spin.Parcels(mapping["radii"], count=args.parcels, seed=args.seed,
                           infall=args.infall, clumps=args.clumps, depth=args.depth)
    hotspots = None
    if args.hotspots:
        hotspots = spin.Parcels(mapping["radii"], count=args.hotspots,
                                seed=args.seed + 1, infall=args.infall, clumps=0)
    print(f" {time.time() - started:.0f}s")
    print(f"  inner ring turns {turns[0]}x per loop, outer ring {turns[-1]}x")

    recorded = notebook.record(
        settings,
        note=(f"spin, {args.channel} channel, {args.frames} frames, "
              f"{args.hotspots} hotspots, infall {args.infall}"),
    )
    print(f"recorded as run {recorded['id']}")

    print(f"pre-rendering {args.frames} frames ...", end="", flush=True)
    started = time.time()
    images = []
    for i in range(args.frames):
        grid = spin.frame(mapping, parcels, i / args.frames, width, height,
                          extent, turns, orders, hotspots=hotspots,
                          hot_gain=args.hot_gain, hot_spread=args.hot_spread)
        images.append(cells.colourise(grid, args.channel))
    print(f" {time.time() - started:.0f}s")

    if args.save_frames:
        from PIL import Image

        out = Path(args.save_frames)
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            Image.fromarray(img).save(out / f"spin_{i:03d}.png")
        print(f"wrote {len(images)} frames to {out}")
        return 0

    # Everything below is display: the physics is already done.
    panes = [cells.half_block(img) for img in images]
    delay = 1.0 / args.fps
    print(f"\n{width}x{height} samples, {width_cells}x{height_cells} cells. "
          f"ctrl-c to stop.\n")
    sys.stdout.write("\033[?25l")           # hide the cursor
    try:
        first = True
        while True:
            for pane in panes:
                if not first:
                    sys.stdout.write(f"\033[{height_cells}A")   # back to the top
                sys.stdout.write(pane + "\n")
                sys.stdout.flush()
                first = False
                time.sleep(delay)
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
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
    if "lines" in str(args.plot):
        args.line_color = ask(f"line colour{note}", DEFAULTS["line_color"], str)
        args.lw = ask(f"line width{note}", DEFAULTS["lw"], str if sweeping else float)
        args.radii = ask(f"radii, blank for {LINE_RADII}", "", str)
        args.ghost_radii = ask(f"ghost radii, blank for {LINE_GHOST_RADII}", "", str)
    if str(args.plot) != "lines":
        args.color_by = ask(f"colour by{note}", DEFAULTS["color_by"],
                            str, COLOR_BY if not sweeping else None)
        args.cmap = ask(f"colormap, blank for the default{note}", DEFAULTS["cmap"], str)
    return args


def ask_view(args, default_output):
    if in_kitty():
        args.view = ask("view", "kitty", str, ["kitty", "window", "file"])
    else:
        args.view = ask("view", "window", str, ["window", "file"])
    if args.view != "window" and default_output is not None:
        args.output = ask("write to", default_output, str)
    else:
        args.output = None
    return args


def menu():
    """The interactive menu shown when luminet is run with no arguments."""
    if not sys.stdin.isatty():
        print("luminet: no menu without a terminal. Try `luminet --help`.", file=sys.stderr)
        return 2

    items = [
        ("0", "explore", "the interactive instrument: parameters and a preview"),
        ("1", "render", "one black hole, in a window or a file"),
        ("2", "sweep", "a grid of variations as one contact sheet"),
        ("3", "gallery", "preset tour of the inclination range"),
        ("4", "photons", "sample photons off the accretion disk"),
        ("5", "explain", "what each control is and what it changes"),
        ("6", "log", "what you have rendered, and notes you kept"),
        ("7", "vary", "repeat a run with one change, beside the original"),
        ("8", "help", "the full list of flags"),
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
        args.radii, args.ghost_radii = "", ""

        print()
        if choice in ("0", "explore", "tui"):
            cmd_tui(None)
        elif choice in ("1", "render"):
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
        elif choice in ("5", "explain"):
            which = ask("which control, blank for all", "", str)
            print()
            cmd_explain(argparse.Namespace(control=which or None))
        elif choice in ("6", "log"):
            print()
            cmd_log(argparse.Namespace(limit=30))
        elif choice in ("7", "vary"):
            latest = notebook.latest()
            if latest is None:
                print("  nothing recorded yet: render something first.")
                continue
            which = ask("which run", latest["id"], int)
            base = notebook.get(which)
            if base is None:
                print(f"  no run {which}")
                continue
            print(f"  run {which}: {notebook.describe(base)}")
            name = ask("which control to change", "incl", str, list(notebook.RECORDED))
            print(f"    {guide.SHORT.get(name, '')}")
            value = ask(f"new {name}", base["settings"].get(name), str)
            v = argparse.Namespace(run=which, tile=3.5, dpi=110)
            for k in notebook.RECORDED:
                setattr(v, k, None)
            setattr(v, name, value)
            ask_view(v, None)
            print()
            cmd_vary(v)
        elif choice in ("8", "help", "h", "?"):
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
    p.add_argument("--line-color", default=DEFAULTS["line_color"],
                   help=f"colour of the lines drawn by --plot lines{note}")
    p.add_argument("--lw", type=kind or float, default=DEFAULTS["lw"],
                   help=f"line width for --plot lines{note}")
    p.add_argument("--radii", default="",
                   help=f"comma-separated radii for --plot lines, default {LINE_RADII}")
    p.add_argument("--ghost-radii", default="",
                   help=f"comma-separated ghost radii, default {LINE_GHOST_RADII}")


def add_view(p, default_output, allow_window):
    choices = ["kitty", "file"] + (["window"] if allow_window else [])
    p.add_argument("--view", default="kitty" if in_kitty() else "file", choices=choices,
                   help="kitty draws inline in the terminal; window opens a plot window")
    p.add_argument("-o", "--output", default=None if default_output is None else default_output,
                   help="also write the image here; it always goes to the notebook")
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
    add_view(p, None, allow_window=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("sweep", help="render a grid of variations as one contact sheet")
    add_common(p, sweeping=True)
    add_view(p, None, allow_window=False)
    p.add_argument("--tile", type=float, default=3.0, help="size of each tile in inches")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("gallery", help="preset tour of the inclination range")
    add_view(p, None, allow_window=False)
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

    p = sub.add_parser("explain", help="what each control is and what it changes")
    p.add_argument("control", nargs="?", help="one control, or leave blank for all of them")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("log", help="what you have rendered so far")
    p.add_argument("--limit", type=int, default=30, help="how many to show")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("show", help="draw an earlier run again")
    p.add_argument("run", nargs="?", type=int, help="run number; the latest if omitted")
    add_view(p, None, allow_window=False)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("vary", help="repeat a run with one thing changed, beside the original")
    p.add_argument("run", nargs="?", type=int, help="run number; the latest if omitted")
    for name in notebook.RECORDED:
        p.add_argument(f"--{name.replace('_', '-')}", default=None,
                       help=f"change {name}; see `luminet explain {name}`")
    add_view(p, None, allow_window=False)
    p.add_argument("--tile", type=float, default=3.5, help="size of each panel in inches")
    p.set_defaults(func=cmd_vary)

    p = sub.add_parser("note", help="keep an observation with a run")
    p.add_argument("run", type=int)
    p.add_argument("text")
    p.set_defaults(func=cmd_note)

    p = sub.add_parser("animate", help="pre-render a loop between two settings")
    p.add_argument("control", nargs="?", help="one control to move, e.g. incl")
    p.add_argument("start", nargs="?", help="its value at one end")
    p.add_argument("end", nargs="?", help="its value at the other end")
    p.add_argument("--between", nargs=2, type=int, metavar=("A", "B"),
                   help="animate between two recorded runs instead")
    p.add_argument("--run", type=int, help="which run supplies the other settings")
    p.add_argument("--frames", type=int, default=30, help="frames on the way out")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--jobs", type=int, default=4, help="how many frames to render at once")
    p.add_argument("--once", action="store_true",
                   help="play straight through instead of returning; will not loop cleanly")
    p.add_argument("--final", action="store_true",
                   help="high resolution, more frames: for the finished piece")
    p.add_argument("--keep-frames", help="keep the individual frames in this directory")
    p.add_argument("-o", "--output", default="loop.gif", help=".gif, .mp4 or .webp")
    p.add_argument("--view", default="kitty" if in_kitty() else "file", choices=["kitty", "file"])
    p.set_defaults(func=cmd_animate)

    p = sub.add_parser("term", help="draw in the terminal with sub-cell glyphs")
    p.add_argument("--channel", default="flux",
                   choices=["flux", "both", "redshift", "radius", "order"],
                   help="what to show. 'both' puts flux in the brightness and redshift in the hue")
    p.add_argument("--encoding", default="half", choices=["half", "sextant"],
                   help="half-block gives every subpixel its own colour; "
                        "sextant trades colour for detail")
    p.add_argument("--run", type=int, help="settings from a recorded run")
    p.add_argument("--incl", type=float)
    p.add_argument("--mass", type=float)
    p.add_argument("--acc", type=float)
    p.add_argument("--outer-edge", type=float)
    p.add_argument("--width", type=int, help="cells across; defaults to the window")
    p.add_argument("--height", type=int, help="cells down; defaults to the window")
    p.add_argument("--radii-samples", type=int, default=140)
    p.add_argument("--angle-samples", type=int, default=240)
    p.add_argument("--threshold", type=float, default=0.06,
                   help="sextant only: how bright a sample must be to light a subpixel")
    p.set_defaults(func=cmd_term)

    p = sub.add_parser("spin", help="animate the disk turning, in the terminal")
    p.add_argument("--channel", default="flux",
                   choices=["flux", "both", "redshift"], help="what the colour means")
    p.add_argument("--frames", type=int, default=48, help="frames in one loop")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--parcels", type=int, default=9000, help="how much gas to track")
    p.add_argument("--rings", type=int, default=44, help="radii the map is solved at")
    p.add_argument("--angles", type=int, default=180)
    p.add_argument("--max-turns", type=int, default=None,
                   help="cap on how many times the inner ring laps per loop")
    p.add_argument("--infall", type=float, default=0.0,
                   help="inward drift per loop, 0 to 1. Costs the exact loop")
    p.add_argument("--clumps", type=int, default=3,
                   help="how many bright patches to mark the gas with, so the rotation "
                        "is visible. A tracer, not something the model predicts. 0 for none")
    p.add_argument("--depth", type=float, default=0.75, help="how pronounced the patches are")
    p.add_argument("--hotspots", type=int, default=14,
                   help="bright spots carried round with the gas, so the rotation reads. "
                        "A marker on the gas, not a prediction. 0 for none")
    p.add_argument("--hot-gain", type=float, default=55.0, help="how bright those spots are")
    p.add_argument("--hot-spread", type=float, default=1.3, help="how many cells each covers")
    p.add_argument("--no-ghost", action="store_true", help="hide the second image")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--once", action="store_true", help="play one loop and stop")
    p.add_argument("--run", type=int, help="settings from a recorded run")
    p.add_argument("--incl", type=float)
    p.add_argument("--mass", type=float)
    p.add_argument("--acc", type=float)
    p.add_argument("--outer-edge", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--save-frames", help="write the frames as PNGs instead of playing")
    p.set_defaults(func=cmd_spin)

    sub.add_parser("tui", help="the interactive instrument: parameters and a live preview")
    sub.add_parser("menu", help="the prompt-based menu")

    p = sub.add_parser("_draw")  # internal: used by the TUI to render out of process
    p.add_argument("--settings", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dpi", type=int, default=110)
    p.set_defaults(func=cmd_draw)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    if argv == ["tui"]:
        return cmd_tui(None)
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
    except notebook.NotebookUnreadable as e:
        print(f"notebook: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
