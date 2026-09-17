"""plot1979 drawn in real pixels, through the kitty graphics protocol.

Braille gives a dot grid of two by four per character cell. kitty and Ghostty
can also show an image placed over the cells, at the screen's own resolution, so
the ink dots can be a pixel or two across and the shadow's edge and the
isoradials can be as fine as the display allows.

Most of each picture does not change from frame to frame. The glow, the round
shadow mask and the vignette depend only on the map, the window and the settings,
so they are painted into a background once and reused. A frame copies that
background and paints the dots and lines over it.

Sending the picture is the expensive part. A full-screen frame is about six
megabytes of pixels. Two ways of getting it to the terminal:

    file    write the pixels to a file in /dev/shm and send only its name.
            Measured at 1.6 ms a frame. The terminal has to support it.
    direct  send the pixels inside the escape sequence, zlib-compressed and
            base64-encoded, at half resolution for the terminal to scale up.
            About 0.2 MB and 5 ms a frame. This is the route the tui-research
            preflight used, and both terminals displayed it.

probe() asks the terminal whether it will read a file, and the view uses file
transmission only if it says yes.
"""

import base64
import os
import select
import sys
import time
import zlib

import numpy as np

from luminet import cells, spin

APC_END = "\x1b\\"


def probe_file_transport(timeout=0.6):
    """Ask the terminal whether it accepts images by temporary file.

    Sends a one-pixel query - it displays nothing - and waits briefly for the
    answer. The answer arrives on standard input, so it is read here before the
    key handler can mistake it for keypresses.
    """
    path = f"/dev/shm/tty-graphics-protocol-luminet-probe-{os.getpid()}"
    try:
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00")
    except OSError:
        return False
    payload = base64.standard_b64encode(path.encode()).decode()
    sys.stdout.write(f"\x1b_Gi=31,s=1,v=1,a=q,t=t,f=24;{payload}{APC_END}")
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    seen = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
            seen += os.read(fd, 4096)
            if b"_Gi=31;" in seen and seen.rstrip().endswith(b"\x1b\\"):
                break
    try:
        os.remove(path)            # the terminal deletes it after reading, if it read it
    except OSError:
        pass
    return b"_Gi=31;OK" in seen


class Transport:
    """Gets a frame's pixels to the terminal, as one escape sequence."""

    def __init__(self, mode):
        self.mode = mode
        self.slot = 0
        self.names = [f"/dev/shm/tty-graphics-protocol-luminet-{os.getpid()}-{i}"
                      for i in range(3)]

    def escape(self, img, cols, rows):
        h, w = img.shape[:2]
        placement = f"i=1,p=1,c={cols},r={rows},q=2,C=1"
        if self.mode == "file":
            # A few names in rotation: the terminal deletes a file once read,
            # and never overwriting the one it may still be reading avoids a race.
            path = self.names[self.slot]
            self.slot = (self.slot + 1) % len(self.names)
            with open(path, "wb") as f:
                f.write(np.ascontiguousarray(img).tobytes())
            payload = base64.standard_b64encode(path.encode()).decode()
            return f"\x1b_Ga=T,f=24,t=t,s={w},v={h},{placement};{payload}{APC_END}"

        data = base64.standard_b64encode(zlib.compress(np.ascontiguousarray(img).tobytes(), 1))
        chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)] or [b""]
        out = []
        for i, chunk in enumerate(chunks):
            more = 1 if i + 1 < len(chunks) else 0
            head = (f"a=T,f=24,o=z,s={w},v={h},{placement},m={more}" if i == 0
                    else f"m={more}")
            out.append(f"\x1b_G{head};{chunk.decode()}{APC_END}")
        return "".join(out)

    def close(self):
        for path in self.names:
            try:
                os.remove(path)
            except OSError:
                pass


class PixelView:
    """One window's worth of plot1979 in pixels."""

    def __init__(self, mapping, extent, rates, projector, orders, cols, rows,
                 cell_px, transport, seed=0, infall=0.0, gamma=0.6, grain=4,
                 density=2.5, floor=0.0):
        self.cols, self.rows = cols, rows
        self.transport = transport
        scale = 1 if transport.mode == "file" else 2
        grain = max(1, grain // scale)

        # A coarse grid decides how likely each patch of screen is to show a
        # dot; dots are then painted at their own pixel. The canvas is made a
        # whole number of grid cells so the two share exact proportions.
        px_w = int(cols * cell_px[0]) // scale
        px_h = int(rows * cell_px[1]) // scale
        self.grid_w, self.grid_h = max(8, px_w // grain), max(8, px_h // grain)
        self.px_w, self.px_h = self.grid_w * grain, self.grid_h * grain
        self.dot = 1 if scale == 2 else 2

        # Pixels are square, so the sample shape is too.
        saved = spin.CELL_ASPECT
        spin.CELL_ASPECT = 1.0
        try:
            self.parcels = spin.Parcels(mapping["radii"],
                                        count=int(self.grid_w * self.grid_h * density),
                                        seed=seed, infall=infall, clumps=0, spread="log")
            self.field = spin.DotField(mapping, self.parcels, self.grid_w, self.grid_h,
                                       extent, rates, orders, gamma=gamma,
                                       projector=projector, floor=floor)
            ext_x, ext_y = self.field.ext_x, self.field.ext_y
            self.lines = spin.Isolines(mapping, self.px_w, self.px_h, ext_x, ext_y,
                                       samples=int(8 * max(self.px_w, self.px_h)))
        finally:
            spin.CELL_ASPECT = saved
        self.ext_x, self.ext_y = ext_x, ext_y
        self.projector = projector
        self.mapping = mapping
        self.critical = float(mapping["bh"].critical_b)
        self.background = None
        self.cell_rgb = None
        self.look = None

    # ------------------------------------------------------------- still parts

    def restyle(self, palette, glow_palette, bloom, mask, vignette, scanlines,
                ink, paper, hole):
        """Rebuild what does not move: dot colours, glow, mask and vignette."""
        look = (palette, glow_palette, round(bloom, 3), mask, vignette, scanlines)
        if look == self.look:
            return
        self.look = look
        target = self.field.target

        if palette == "ink":
            rgb = np.broadcast_to(np.array(ink, np.float32), (*target.shape, 3)).copy()
            glow_tint = np.array(ink, np.float32)
        else:
            stops = cells.palette(palette)
            tone = 0.4 + 0.6 * np.clip(target, 0.0, 1.0) ** 0.5
            rgb = cells._ramp(tone, stops).astype(np.float32)
            glow_tint = np.array(stops[len(stops) * 2 // 3], np.float32)

        grain = self.px_w // self.grid_w
        bg = np.empty((self.px_h, self.px_w, 3), np.float32)
        bg[:] = paper
        if bloom > 0:
            from scipy.ndimage import gaussian_filter, zoom

            spill = gaussian_filter(target.astype(np.float32), sigma=2.2)
            spill = zoom(spill, grain, order=1)[:self.px_h, :self.px_w]
            spill = np.pad(spill, ((0, self.px_h - spill.shape[0]),
                                   (0, self.px_w - spill.shape[1])), mode="edge")
            if glow_palette == "match":
                tint = glow_tint[None, None, :]
            else:
                reach = spill / max(float(spill.max()), 1e-6)
                tint = cells._ramp(reach, cells.palette(glow_palette)).astype(np.float32)
            bg += tint * (0.6 * bloom * spill)[..., None]

        yy, xx = np.mgrid[0:self.px_h, 0:self.px_w].astype(np.float32)
        if mask:
            # The shadow, exactly round. Its edge is blended over a pixel by
            # distance, so it is smooth at the display's own resolution.
            cx, cy = (self.px_w - 1) / 2.0, (self.px_h - 1) / 2.0
            rx = self.critical / self.ext_x * (self.px_w - 1) / 2.0
            ry = self.critical / self.ext_y * (self.px_h - 1) / 2.0
            dist = np.hypot((xx - cx) / rx, (yy - cy) / ry) * min(rx, ry)
            cover = np.clip(min(rx, ry) - dist + 0.5, 0.0, 1.0)
            # The near side of the disk passes in front of the hole; its glow is
            # left alone. Softened, because its edge comes from samples.
            from scipy.ndimage import gaussian_filter, zoom

            front = gaussian_filter((self.field.front > 0).astype(np.float32), 1.0)
            front = zoom(front, grain, order=1)
            front = np.pad(front, ((0, max(0, self.px_h - front.shape[0])),
                                   (0, max(0, self.px_w - front.shape[1]))),
                           mode="edge")[:self.px_h, :self.px_w]
            cover = (cover * (1.0 - np.clip(front * 1.5, 0.0, 1.0)))[..., None]
            bg = bg * (1.0 - cover) + np.array(hole, np.float32) * cover
        if vignette:
            fx = xx / max(self.px_w - 1, 1) * 2 - 1
            fy = yy / max(self.px_h - 1, 1) * 2 - 1
            fade = (1.0 - 0.45 * np.clip((fx * fx + fy * fy) / 2.0, 0, 1))[..., None]
            bg *= fade
            rgb *= float(fade.mean())
        if scanlines:
            period = max(2, self.px_h // max(self.rows * 2, 1))
            bg[(np.arange(self.px_h) % period) >= period / 2] *= 0.72

        self.background = np.clip(bg, 0, 255).astype(np.uint8)
        self.cell_rgb = np.clip(rgb.reshape(-1, 3), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ frame

    def frame(self, t, rates, dots_on=True, overlay="off", line_rgb=(150, 205, 235),
              ink=(238, 230, 210)):
        img = self.background.copy()
        if dots_on:
            if self.projector is not None:
                self.projector.paint(self.parcels, t, rates, self.ext_x, self.ext_y,
                                     self.grid_w, self.grid_h, self.field.chance,
                                     self.cell_rgb, img, self.dot)
            else:
                self._paint_numpy(img, t, rates)
        if overlay != "off":
            mask = self.lines.frame(t, rates, flowing=overlay == "flowing")
            img[mask] = line_rgb if dots_on else ink
        return self.transport.escape(img, self.cols, self.rows)

    def _paint_numpy(self, img, t, rates):
        """The same painting without numba."""
        m = self.mapping
        radii, angles = m["radii"], m["angles"]
        p = self.parcels
        ring, _, angle = p.at(t, radii, None, rates)
        upper = np.minimum(ring + 1, len(radii) - 1)
        col = angle / (2 * np.pi) * (len(angles) - 1)
        lo = np.floor(col).astype(int) % len(angles)
        hi = (lo + 1) % len(angles)
        f = col - np.floor(col)
        for o in (0, 1):
            if o not in m["tables"]:
                continue
            tb = m["tables"][o][0]
            b = ((1 - p.jitter) * ((1 - f) * tb[ring, lo] + f * tb[ring, hi])
                 + p.jitter * ((1 - f) * tb[upper, lo] + f * tb[upper, hi]))
            good = np.isfinite(b)
            xf = (b[good] * np.sin(angle[good]) / self.ext_x + 1) / 2
            yf = (b[good] * np.cos(angle[good]) / self.ext_y + 1) / 2
            gc = (xf * (self.grid_w - 1)).astype(np.int64)
            gr = (yf * (self.grid_h - 1)).astype(np.int64)
            inside = (gc >= 0) & (gc < self.grid_w) & (gr >= 0) & (gr < self.grid_h)
            c = gr[inside] * self.grid_w + gc[inside]
            on = p.luck[good][inside] < self.field.chance[o, c]
            px = (xf[inside][on] * (self.px_w - 1)).astype(np.int64)
            py = (yf[inside][on] * (self.px_h - 1)).astype(np.int64)
            colours = self.cell_rgb[c[on]]
            for dy in range(self.dot):
                for dx in range(self.dot):
                    yy, xx = py + dy, px + dx
                    # A dot just past the left or top edge lands at -1, which
                    # numpy would read as the far edge. Drop it, as the
                    # compiled painter does.
                    ok = (yy >= 0) & (yy < self.px_h) & (xx >= 0) & (xx < self.px_w)
                    img[yy[ok], xx[ok]] = colours[ok]
