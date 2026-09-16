"""Pre-render an animation between two settings, as a loop.

What can actually move
----------------------

The renderer is not a 3D scene. It solves what a camera at one inclination sees
and draws that, so there is nothing to fly a camera through. The one meaningful
camera freedom is inclination, and tilting through it reads as three-dimensional
for the same reason a turntable does.

Inclination is also more limited than it looks. The renderer returns the same
image for incl, pi - incl, pi + incl and 2*pi - incl, checked by rendering all
four and comparing the files. Only 0 to pi/2 is distinct; going further retraces
what you have already seen, so a one-directional cycle through inclination is
not available.

That settles how the loop is built. Frames are rendered once from one end to the
other and then played back in reverse, which closes the loop exactly rather than
approximately: the last frame's neighbour is the first frame. It also halves the
work, since the return leg costs nothing to render.

Any pair of settings can be the two ends, not just inclination.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from luminet import notebook

# Settings that can be moved through continuously. Everything else is a choice
# rather than a quantity, and has to be the same at both ends.
CONTINUOUS = ["incl", "mass", "acc", "outer_edge", "lw"]
DISCRETE = ["plot", "color_by", "cmap", "line_color", "radii", "ghost_radii", "resolution"]


def interpolate(start, end, t):
    """One frame's settings, a fraction t of the way from start to end."""
    frame = dict(end)
    for name in CONTINUOUS:
        a, b = start.get(name), end.get(name)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            frame[name] = round(a + (b - a) * t, 5)
    return frame


def differing_discrete(start, end):
    """Discrete settings that differ, which cannot be animated through."""
    return [n for n in DISCRETE if str(start.get(n)) != str(end.get(n))]


def render_frames(frames, directory, dpi, jobs=4, on_done=None):
    """Draw every frame, a few at a time. Returns the paths in order."""
    paths = [directory / f"frame_{i:04d}.png" for i in range(len(frames))]

    def one(i):
        proc = subprocess.run(
            [sys.executable, "-m", "luminet.cli", "_draw",
             "--settings", json.dumps(frames[i]), "-o", str(paths[i]), "--dpi", str(dpi)],
            capture_output=True, text=True,
        )
        if on_done:
            on_done(i, proc.returncode == 0)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(tail[-1] if tail else f"frame {i} failed")
        return paths[i]

    # Each render starts its own worker pool, so only a few run at once.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, range(len(frames))))
    return paths


def assemble(paths, output, fps):
    """Write the frames out as a looping animation."""
    output = Path(output)
    if output.suffix.lower() == ".gif":
        return _gif(paths, output, fps)
    return _ffmpeg(paths, output, fps)


def _gif(paths, output, fps):
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in paths]
    # A shared adaptive palette stops the colours shifting from frame to frame.
    frames = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(
        output, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True, disposal=2,
    )
    return output


def _ffmpeg(paths, output, fps):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed for this format; use a .gif instead")

    listing = paths[0].parent / "frames.txt"
    listing.write_text("".join(f"file '{p.name}'\nduration {1 / fps:.5f}\n" for p in paths))

    codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p",
             # even dimensions, which h264 requires
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if output.suffix.lower() == ".webp":
        codec = ["-c:v", "libwebp", "-loop", "0", "-lossless", "0", "-q:v", "80"]

    proc = subprocess.run(
        ["ffmpeg", "-y", "-r", str(fps), "-f", "concat", "-safe", "0",
         "-i", str(listing), *codec, "-r", str(fps), str(output)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()
        raise RuntimeError(tail[-1] if tail else "ffmpeg failed")
    return output


def build(start, end, output, frames=30, fps=20, dpi=110, bounce=True,
          jobs=4, keep_frames=None, progress=True):
    """Render a loop between two settings. Returns (path, frames written)."""
    clashing = differing_discrete(start, end)
    if clashing:
        raise ValueError(
            "these cannot be animated through, only switched: "
            + ", ".join(f"{n} ({start.get(n)} vs {end.get(n)})" for n in clashing)
        )

    plan = [interpolate(start, end, i / max(frames - 1, 1)) for i in range(frames)]

    workdir = Path(keep_frames) if keep_frames else Path(tempfile.mkdtemp(prefix="luminet-anim-"))
    workdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    done = [0]

    def tick(_i, ok):
        done[0] += 1
        if progress:
            print(f"\r  rendered {done[0]}/{len(plan)} frames "
                  f"({time.time() - started:.0f}s)", end="", flush=True)

    paths = render_frames(plan, workdir, dpi, jobs=jobs, on_done=tick)
    if progress:
        print()

    # Play the frames back in reverse to close the loop. The turning points are
    # not repeated, so no frame is ever shown twice in a row.
    sequence = paths + list(reversed(paths[1:-1])) if bounce and len(paths) > 2 else paths

    out = assemble(sequence, output, fps)
    if not keep_frames:
        shutil.rmtree(workdir, ignore_errors=True)
    return out, len(sequence)


def endpoints_from_runs(start_id, end_id):
    """The settings of two recorded runs, as the two ends of an animation."""
    a, b = notebook.get(start_id), notebook.get(end_id)
    if a is None or b is None:
        missing = start_id if a is None else end_id
        raise ValueError(f"no run {missing}. `luminet log` lists them.")
    return dict(a["settings"]), dict(b["settings"])
