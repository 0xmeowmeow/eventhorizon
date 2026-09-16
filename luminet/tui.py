"""An interactive terminal instrument for exploring the renderer.

Every parameter is on screen with the values it will accept, so there is nothing
to look up. A render can be pinned as the baseline and stays in view while the
next one changes, so the effect of a single control is something you see rather
than something you remember.

Renders happen in a subprocess. The renderer parallellizes over a multiprocessing
Pool, and forking that from inside a threaded UI process is a good way to
deadlock, so the work is handed to a fresh interpreter instead.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static

from luminet import guide, notebook

try:
    from textual_image.widget import Image
except ImportError:  # pragma: no cover - textual-image is an optional extra
    Image = None


# Every control, with the values it accepts. Enumerations are what the user
# arrows through; ranges clamp and step. This list is the answer to "what are
# my options", so it stays visible rather than living in a help page.
SPEC = [
    ("plot", "enum", ["image", "lines", "flat", "isoradials", "isoredshifts", "isofluxlines"]),
    ("incl", "range", (0.0, 1.5707, 0.1)),
    ("outer_edge", "range", (7.0, 200.0, 5.0)),
    ("mass", "range", (0.25, 8.0, 0.25)),
    ("acc", "range", (0.1, 100.0, 1.0)),
    ("resolution", "range", (20, 400, 20)),
    ("color_by", "enum", ["flux", "redshift"]),
    ("cmap", "enum", ["", "inferno", "magma", "plasma", "viridis", "cividis",
                      "Greys_r", "bone", "copper", "RdBu_r"]),
    ("line_color", "enum", ["white", "#66ccff", "#ff6688", "#ffcc44", "#88ff99"]),
    ("lw", "range", (0.5, 5.0, 0.5)),
    ("radii", "text", ""),
    ("ghost_radii", "text", ""),
]

SPEC_BY_NAME = {name: (kind, opts) for name, kind, opts in SPEC}

START = {
    "plot": "lines", "incl": 1.4, "mass": 1.0, "acc": 1.0, "outer_edge": 40.0,
    "resolution": 100, "color_by": "flux", "cmap": "", "line_color": "white",
    "lw": 1.0, "radii": "", "ghost_radii": "",
}

# Controls that cannot change the picture, flagged in place so the time is not
# spent finding that out by trial. Both were checked against the renderer.
INERT = {
    "acc": "cannot change the image, only flux numbers",
    "cmap": "presentation only",
    "line_color": "presentation only",
    "lw": "presentation only",
    "resolution": "smoothness and time only",
}


def render_in_subprocess(settings, path, dpi=110):
    """Draw one image in a fresh interpreter. Returns (ok, message)."""
    proc = subprocess.run(
        [sys.executable, "-m", "luminet.cli", "_draw",
         "--settings", json.dumps(settings), "-o", str(path), "--dpi", str(dpi)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, tail[-1] if tail else f"render failed ({proc.returncode})"
    return True, ""


class ParamRow(Static):
    """One control: its name, its current value, and everything it will accept."""

    def __init__(self, name, value, selected=False):
        super().__init__()
        self.param = name
        self.value = value
        self.selected = selected

    def render(self):
        kind, opts = SPEC_BY_NAME[self.param]
        marker = "[b]>[/b]" if self.selected else " "
        name = f"[b]{self.param:<12}[/b]" if self.selected else f"{self.param:<12}"

        if kind == "enum":
            shown = []
            for opt in opts:
                label = opt if opt != "" else "none"
                if opt == self.value:
                    shown.append(f"[reverse] {label} [/reverse]")
                else:
                    shown.append(f"[dim]{label}[/dim]")
            body = " ".join(shown)
        elif kind == "range":
            lo, hi, _ = opts
            body = f"[b]{self.value}[/b]  [dim]{lo} to {hi}[/dim]"
        else:
            shown = self.value if str(self.value).strip() else "(default)"
            body = f"[b]{shown}[/b]  [dim]comma-separated, e.g. 6,10,15,20[/dim]"

        note = INERT.get(self.param)
        tail = f"  [yellow dim]{note}[/yellow dim]" if note else ""
        return f"{marker} {name} {body}{tail}"


class Explain(ModalScreen):
    """The full explanation of one control."""

    BINDINGS = [Binding("escape,q,question_mark", "dismiss", "close")]

    def __init__(self, text):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="explain-box"):
            yield Static(self.text)
        yield Static("[dim]esc to close[/dim]", id="explain-foot")


class NotePrompt(ModalScreen):
    """Keep an observation with the current run."""

    BINDINGS = [Binding("escape", "dismiss", "cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="note-box"):
            yield Static("What did you notice?")
            yield Input(placeholder="the ghost image separates here", id="note-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class LuminetTUI(App):
    """Change one thing, keep the last one beside it."""

    CSS = """
    Screen { background: $surface; }
    #params { height: auto; padding: 0 1; border-bottom: solid $panel; }
    ParamRow { height: 1; }
    #panes { height: 1fr; }
    .pane { width: 1fr; border: solid $panel; padding: 0 1; }
    .pane-title { height: 1; color: $text-muted; }
    #status { height: auto; padding: 0 1; border-top: solid $panel; }
    #explain-box {
        width: 80%; height: 80%; margin: 2 4; padding: 1 2;
        background: $surface; border: thick $primary;
    }
    #explain-foot { width: 80%; margin: 0 4; padding: 0 2; }
    #note-box {
        width: 60%; height: auto; margin: 4 8; padding: 1 2;
        background: $surface; border: thick $primary;
    }
    """

    BINDINGS = [
        Binding("up,k", "move(-1)", "prev control"),
        Binding("down,j", "move(1)", "next control"),
        Binding("left,h", "change(-1)", "lower"),
        Binding("right,l", "change(1)", "raise"),
        Binding("r,enter", "render", "render"),
        Binding("space", "pin", "pin as baseline"),
        Binding("a", "animate", "animate baseline -> current"),
        Binding("n", "note", "note"),
        Binding("question_mark", "explain", "explain"),
        Binding("q,escape", "quit", "quit"),
    ]

    def __init__(self):
        super().__init__()
        self.settings = dict(START)
        self.cursor = 0
        self.baseline = None          # the pinned run record
        self.current_run = None       # the run just drawn
        self.tmp = Path(tempfile.mkdtemp(prefix="luminet-tui-"))
        self.busy = False

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="params"):
            for i, (name, _, _) in enumerate(SPEC):
                yield ParamRow(name, self.settings[name], selected=(i == 0))
        with Horizontal(id="panes"):
            with Vertical(classes="pane", id="baseline-pane"):
                yield Static("baseline · nothing pinned", classes="pane-title", id="baseline-title")
                if Image:
                    yield Image(id="baseline-image")
            with Vertical(classes="pane", id="current-pane"):
                yield Static("current · not rendered", classes="pane-title", id="current-title")
                if Image:
                    yield Image(id="current-image")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "luminet"
        self.sub_title = "change one thing, keep the last beside it"
        self.refresh_status()
        self.action_render()

    # ------------------------------------------------------------------- state

    @property
    def rows(self):
        return list(self.query(ParamRow))

    def refresh_rows(self) -> None:
        for i, row in enumerate(self.rows):
            row.selected = i == self.cursor
            row.value = self.settings[row.param]
            row.refresh()

    def refresh_status(self, message="") -> None:
        bits = []
        if self.baseline and self.current_run:
            changed = notebook.differences(self.baseline["settings"], self.settings)
            if changed:
                bits.append("changed: " + ", ".join(
                    f"[b]{k}[/b] {v[0]} -> {v[1]}" for k, v in changed.items()))
            else:
                bits.append("[dim]identical to the baseline[/dim]")
        elif not self.baseline:
            bits.append("[dim]press space to pin this render as the baseline[/dim]")
        if message:
            bits.append(message)
        self.query_one("#status", Static).update("   ".join(bits))

    # ---------------------------------------------------------------- actions

    def action_move(self, delta: int) -> None:
        self.cursor = (self.cursor + delta) % len(SPEC)
        self.refresh_rows()

    def action_change(self, delta: int) -> None:
        name, kind, opts = SPEC[self.cursor]
        value = self.settings[name]

        if kind == "enum":
            i = opts.index(value) if value in opts else 0
            self.settings[name] = opts[(i + delta) % len(opts)]
        elif kind == "range":
            lo, hi, step = opts
            nxt = value + delta * step
            nxt = max(lo, min(hi, nxt))
            self.settings[name] = int(round(nxt)) if isinstance(step, int) else round(nxt, 3)
        else:
            # Free text is edited in the note-style prompt rather than arrowed.
            self.notify("press enter on this row to type a value", timeout=3)
            return

        self.refresh_rows()
        self.refresh_status()

    def action_pin(self) -> None:
        if self.current_run is None:
            self.notify("render something first", timeout=3)
            return
        self.baseline = self.current_run
        path = self.tmp / f"baseline.png"
        try:
            path.write_bytes((self.tmp / "current.png").read_bytes())
        except OSError:
            pass
        if Image:
            self.query_one("#baseline-image", Image).image = str(path)
        self.query_one("#baseline-title", Static).update(
            f"baseline · run {self.baseline['id']} · {notebook.describe(self.baseline)}")
        self.refresh_status("pinned")

    def action_explain(self) -> None:
        name = SPEC[self.cursor][0]
        self.push_screen(Explain(guide.explain(name)))

    def action_note(self) -> None:
        if self.current_run is None:
            self.notify("render something first", timeout=3)
            return

        def keep(text):
            if text:
                notebook.annotate(self.current_run["id"], text)
                self.refresh_status(f"noted on run {self.current_run['id']}")

        self.push_screen(NotePrompt(), keep)

    def action_animate(self) -> None:
        """Loop between the pinned baseline and what is on screen now."""
        if self.baseline is None or self.current_run is None:
            self.notify("pin a baseline with space, change something, then press a", timeout=4)
            return
        if self.busy:
            return
        self.busy = True
        out = notebook.NOTEBOOK_DIR / f"loop_{self.baseline['id']:03d}_{self.current_run['id']:03d}.gif"
        self.refresh_status("[b]rendering the loop ...[/b]")
        self.do_animate(dict(self.baseline["settings"]), dict(self.settings), out)

    @work(thread=True, exclusive=True)
    def do_animate(self, start, end, out) -> None:
        from luminet import animate

        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            path, total = animate.build(start, end, out, frames=24, fps=20,
                                        dpi=100, jobs=4, progress=False)
            message = f"wrote {path.name}, {total} frames, loops seamlessly"
        except (ValueError, RuntimeError) as e:
            message = f"[red]{e}[/red]"
        self.call_from_thread(self.animated, message)

    def animated(self, message) -> None:
        self.busy = False
        self.refresh_status(message)

    def action_render(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.query_one("#current-title", Static).update("current · rendering ...")
        self.do_render(dict(self.settings))

    @work(thread=True, exclusive=True)
    def do_render(self, settings) -> None:
        path = self.tmp / "current.png"
        ok, message = render_in_subprocess(settings, path)
        self.call_from_thread(self.rendered, settings, path, ok, message)

    def rendered(self, settings, path, ok, message) -> None:
        self.busy = False
        if not ok:
            self.query_one("#current-title", Static).update("current · failed")
            self.refresh_status(f"[red]{message}[/red]")
            return

        run = notebook.record(settings, parent=self.baseline["id"] if self.baseline else None)
        self.current_run = run
        if Image:
            widget = self.query_one("#current-image", Image)
            widget.image = str(path)
        self.query_one("#current-title", Static).update(
            f"current · run {run['id']} · {notebook.describe(run)}")
        self.refresh_status()


def run():
    if Image is None:
        print("the TUI needs textual-image: pip install textual-image", file=sys.stderr)
        return 2
    LuminetTUI().run()
    return 0
