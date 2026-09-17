"""Events over the plot: a probe, a transmission, an observatory HUD, a warp jump.

These are theatre rather than predictions, but each is staged on the same
geometry as the disk, so what they do follows the lensing the dots show:

probe         A craft falls straight in from rest, in the plane of the disk. Its
              position is solved exactly for every frame, direct image and ghost
              image both, down to 3 M where the solver stops; inside that it
              holds its last image. Seen from far away it slows as it nears the
              horizon, reddens, dims and freezes - it never visibly crosses.
              Its redshift combines gravity with its own fall, taking the
              fall as directly away from the observer, so it is an upper bound;
              the stretching is drawn, not computed.
transmission  A message, one cuneiform sign per byte, spiralling inward with
              the gas at the Keplerian rate for its radius, projected through
              the same map, and swallowed at the photon ring. Sign U+12000+n is
              byte n, so the message can be read back.
hud           Readouts in seven-segment digits, a reticle locked to one parcel
              of gas with its redshift factor, a dotted photon ring and a sweep.
warp          The view falls towards the hole with streaks, flashes white, and
              comes out in an inverted "white hole" with the gas running
              backwards, and stays there until sent back the same way. That
              part is pure fiction.

Everything is drawn through a canvas, so one description serves both the
pixel picture and the braille one. Glyphs - signs, digits, reticle corners -
go on top as text.
"""

import time

import numpy as np

from luminet import black_hole_math as bhm

HUD = (120, 240, 170)
TWO_PI = 2.0 * np.pi

MESSAGES = [
    "hello from the photon ring",
    "time runs slow down here",
    "the light you see left long ago",
    "wish you were here",
    "do not wait for the signal",
    "6 M and falling",
]


def seven_segment(text):
    """Digits as the segmented digits at U+1FBF0; everything else unchanged."""
    return "".join(chr(0x1FBF0 + ord(c) - 48) if "0" <= c <= "9" else c for c in text)


def encode_message(text):
    return "".join(chr(0x12000 + b) for b in text.encode("utf-8"))


def decode_message(signs):
    return bytes(ord(c) - 0x12000 for c in signs).decode("utf-8", errors="replace")


# ------------------------------------------------------------------ canvases

class PixelCanvas:
    """The pixel picture, an (H, W, 3) uint8 image, drawn on in place."""

    kind = "pixels"

    def __init__(self, img, ext_x, ext_y, cols, rows):
        self.img, self.ext_x, self.ext_y = img, ext_x, ext_y
        self.cols, self.rows = cols, rows
        self.h, self.w = img.shape[:2]
        # Pixels to one braille dot's width, so marks keep their size across
        # full-resolution and half-resolution pictures.
        self.unit = max(1, int(round(self.w / max(cols * 2, 1))))

    def plot(self, xf, yf, rgb, alpha=1.0, size=1):
        xf, yf = np.atleast_1d(xf), np.atleast_1d(yf)
        n = xf.size
        if n == 0:
            return
        px = np.rint(xf * (self.w - 1)).astype(np.int64)
        py = np.rint(yf * (self.h - 1)).astype(np.int64)
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
        a = np.clip(np.broadcast_to(np.asarray(alpha, np.float32), (n,)), 0.0, 1.0)
        # Every pixel of every mark in one gather and one scatter.
        half = size // 2
        off = np.arange(size) - half
        yy = (py[:, None, None] + off[None, :, None]).repeat(size, axis=2).ravel()
        xx = (px[:, None, None] + off[None, None, :]).repeat(size, axis=1).ravel()
        which = np.repeat(np.arange(n), size * size)
        ok = (yy >= 0) & (yy < self.h) & (xx >= 0) & (xx < self.w) & (a[which] > 0)
        if not ok.any():
            return
        flat = self.img.reshape(-1, 3)
        at = yy[ok] * self.w + xx[ok]
        which = which[ok]
        k = a[which, None]
        flat[at] = (flat[at].astype(np.float32) * (1 - k) + rgb[which] * k).astype(np.uint8)

    def wash(self, rgb, amount):
        if amount <= 0:
            return
        k = float(min(amount, 1.0))
        out = self.img.astype(np.float32)
        out *= 1 - k
        out += np.asarray(rgb, np.float32) * k
        self.img[:] = out.astype(np.uint8)

    def invert(self):
        np.subtract(255, self.img, out=self.img)

    def sample(self, n, rng):
        # Dots are a pixel or two across, so every pixel is looked at: a
        # stride through the image steps straight over most of them.
        idx = np.flatnonzero(self.img.max(axis=2) > 110)
        if idx.size == 0:
            return np.empty(0), np.empty(0)
        pick = rng.choice(idx, size=min(n, idx.size), replace=False)
        r, c = np.divmod(pick, self.w)
        return c / max(self.w - 1, 1), r / max(self.h - 1, 1)

    def cell(self, xf, yf):
        return (int(yf * (self.h - 1)) * self.rows // self.h,
                int(xf * (self.w - 1)) * self.cols // self.w)

    def bg_at(self, row, col):
        return None          # text over the picture keeps the default background


class DotCanvas:
    """The braille picture: a dot mask, and one colour and background per cell."""

    kind = "dots"
    unit = 1

    def __init__(self, lit, colours, backgrounds, ext_x, ext_y, cols, rows):
        self.lit, self.colours, self.backgrounds = lit, colours, backgrounds
        self.ext_x, self.ext_y = ext_x, ext_y
        self.cols, self.rows = cols, rows
        self.h, self.w = lit.shape

    def plot(self, xf, yf, rgb, alpha=1.0, size=1):
        xf, yf = np.atleast_1d(xf), np.atleast_1d(yf)
        n = xf.size
        if n == 0:
            return
        dx = np.rint(xf * (self.w - 1)).astype(np.int64)
        dy = np.rint(yf * (self.h - 1)).astype(np.int64)
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
        a = np.clip(np.broadcast_to(np.asarray(alpha, np.float32), (n,)), 0.0, 1.0)
        # A dot is on or off, so faint marks light nothing; they still tint.
        ok = (dx >= 0) & (dx < self.w) & (dy >= 0) & (dy < self.h) & (a > 0.05)
        dx, dy, rgb, a = dx[ok], dy[ok], rgb[ok], a[ok]
        on = a >= 0.3
        self.lit[dy[on], dx[on]] = True
        if size >= 3:
            for ox, oy in ((1, 0), (0, 1), (1, 1)):
                yy, xx = dy[on] + oy, dx[on] + ox
                keep = (yy < self.h) & (xx < self.w)
                self.lit[yy[keep], xx[keep]] = True
        r = np.minimum(dy // 4, self.rows - 1)
        c = np.minimum(dx // 2, self.cols - 1)
        cur = self.colours[r, c]
        self.colours[r, c] = cur * (1 - a[:, None]) + rgb * a[:, None]

    def wash(self, rgb, amount):
        if amount <= 0:
            return
        k = float(min(amount, 1.0))
        tint = np.asarray(rgb, np.float32)
        self.colours *= 1 - k
        self.colours += tint * k
        self.backgrounds *= 1 - k
        self.backgrounds += tint * k

    def invert(self):
        np.subtract(255.0, self.colours, out=self.colours)
        np.subtract(255.0, self.backgrounds, out=self.backgrounds)

    def sample(self, n, rng):
        idx = np.flatnonzero(self.lit)
        if idx.size == 0:
            return np.empty(0), np.empty(0)
        pick = rng.choice(idx, size=min(n, idx.size), replace=False)
        r, c = np.divmod(pick, self.w)
        return c / max(self.w - 1, 1), r / max(self.h - 1, 1)

    def cell(self, xf, yf):
        return (int(yf * (self.h - 1)) // 4, int(xf * (self.w - 1)) // 2)

    def bg_at(self, row, col):
        return tuple(int(v) for v in np.clip(self.backgrounds[row, col], 0, 255))


# ------------------------------------------------------------------ geometry

def to_screen(canvas, b, alpha):
    """Sky position (impact parameter, angle) as fractions across the picture."""
    xf = (b * np.sin(alpha) / canvas.ext_x + 1.0) / 2.0
    yf = (b * np.cos(alpha) / canvas.ext_y + 1.0) / 2.0
    return xf, yf


def project(mapping, r, alpha, order):
    """Impact parameter and redshift factor for points on the disk, from the map.

    The same bilinear lookup the frame kernel makes, vectorised: fractional
    ring from the radius, fractional column from the angle. Radii outside the
    map, and places no light reaches, come back as NaN.
    """
    r = np.atleast_1d(np.asarray(r, float))
    alpha = np.atleast_1d(np.asarray(alpha, float)) % TWO_PI
    b_tab, z_tab = mapping["tables"][order]
    radii = np.asarray(mapping["radii"], float)
    nr, na = b_tab.shape
    fi = np.interp(np.log(np.maximum(r, 1e-9)), np.log(radii), np.arange(nr))
    rg = np.minimum(np.floor(fi).astype(np.int64), nr - 2)
    fr = fi - rg
    col = alpha / TWO_PI * (na - 1)
    lo = np.floor(col).astype(np.int64) % na
    hi = (lo + 1) % na
    ft = col - np.floor(col)

    def lerp(tab):
        return ((1 - fr) * ((1 - ft) * tab[rg, lo] + ft * tab[rg, hi])
                + fr * ((1 - ft) * tab[rg + 1, lo] + ft * tab[rg + 1, hi]))

    b, z = lerp(b_tab), lerp(z_tab)
    outside = (r < radii[0]) | (r > radii[-1])
    b[outside] = np.nan
    z[outside] = np.nan
    return b, z


def framed_radius(live):
    """The disk radius the view is framed on, as Live.framing takes it."""
    return max(float(live.settings["outer_edge"]) * 0.85, float(live.mapping["radii"][0]) * 1.5)


def m_per_second(mapping, rates):
    """How much coordinate time, in units of M, one second of animation stands for.

    The gas turns at its Keplerian rate scaled by the speed setting, so the same
    scaling keeps everything else in step with it.
    """
    mass = float(mapping["bh"].mass)
    r0 = float(mapping["radii"][0])
    return float(rates[0]) / np.sqrt(mass / r0 ** 3)


def temperature(g):
    """White through amber to deep red as light is shifted to g = 1/(1+z)."""
    g = np.clip(g, 0.0, 1.0)
    return np.stack([255 * np.ones_like(g), 60 + 195 * g ** 1.6, 30 + 225 * g ** 3.5], -1)


def shift_colour(z):
    """Blue-white for light shifted blue, amber for red, cream between."""
    t = np.clip((np.asarray(z, float) - 1.0) * 3.0, -1.0, 1.0)[..., None]
    neutral = np.array([230, 235, 215], np.float32)
    blue = np.array([150, 215, 255], np.float32)
    red = np.array([255, 160, 90], np.float32)
    return np.where(t < 0, neutral + (blue - neutral) * -t, neutral + (red - neutral) * t)


# --------------------------------------------------------------------- probe

class Probe:
    LOST_BELOW = 0.06          # observed brightness factor g at which the signal is gone

    def __init__(self, mapping, framed_radius, rng):
        m = float(mapping["bh"].mass)
        self.m = m
        self.r0 = float(np.clip(framed_radius * 0.7, 14 * m, 60 * m))
        self.r = self.r0 * (1 - 1e-3)
        self.alpha = float(rng.uniform(0, TWO_PI))
        self.age = 0.0
        self.images = {0: None, 1: None}
        self.lost_at = None

    def fall(self, dm):
        """Advance by dm of coordinate time, falling from rest at r0."""
        m, r0 = self.m, self.r0
        e = np.sqrt(1 - 2 * m / r0)
        left = dm
        while left > 0:
            r = self.r
            speed = (1 - 2 * m / r) * np.sqrt(max(2 * m / r - 2 * m / r0, 0.0)) / e
            step = min(left, 0.5 * m, 0.02 * (r - 2 * m) / max(speed, 1e-12))
            self.r = max(r - speed * step, 2 * m * (1 + 1e-12))
            left -= step
            if step < 1e-9:
                break

    def local_speed(self):
        m = self.m
        return np.sqrt(max((2 * m / self.r - 2 * m / self.r0) / (1 - 2 * m / self.r0), 0.0))

    def shift(self):
        """1 + z seen from far away: gravity, and the fall taken as straight away."""
        v = min(self.local_speed(), 1 - 1e-12)
        doppler = np.sqrt((1 + v) / (1 - v))
        gravity = 1 / np.sqrt(max(1 - 2 * self.m / self.r, 1e-24))
        return float(doppler * gravity)

    def locate(self, incl):
        for order in (0, 1):
            if self.r > 3.02 * self.m:
                b = bhm.solve_for_impact_parameter(self.r, incl, self.alpha, self.m, order)
                if b is not None and np.isfinite(b):
                    self.images[order] = float(b)


# -------------------------------------------------------------- transmission

class Transmission:
    FLIGHT = 9.0        # seconds from the edge of the view to the inner edge
    GAP = 0.32          # seconds between signs
    SWALLOW = 0.5

    def __init__(self, text, mapping, framed_radius, rng):
        self.text = text
        self.signs = encode_message(text)
        m = float(mapping["bh"].mass)
        self.r_in = float(mapping["radii"][0])
        self.r0 = float(np.clip(framed_radius * 0.8, 3 * self.r_in, mapping["radii"][-1]))
        self.m = m
        n = len(self.signs)
        self.launch = np.arange(n) * self.GAP
        self.alpha = np.full(n, float(rng.uniform(0, TWO_PI)))
        self.r = np.full(n, self.r0)
        self.age = 0.0
        self.swallowed_at = np.full(n, np.nan)
        self.received = 0

    def advance(self, dt, dm):
        self.age += dt
        flying = (self.age >= self.launch) & np.isnan(self.swallowed_at)
        k = np.log(self.r0 / self.r_in) / self.FLIGHT
        self.r[flying] *= np.exp(-k * dt)
        self.alpha[flying] += np.sqrt(self.m / self.r[flying] ** 3) * dm
        # Signs still waiting to launch ride round at the starting radius, so
        # the message leaves as a stream along one orbit rather than a clump.
        idle = self.age < self.launch
        self.alpha[idle] += np.sqrt(self.m / self.r0 ** 3) * dm
        arrived = flying & (self.r <= self.r_in)
        self.swallowed_at[arrived] = self.age
        self.received = int(np.sum(self.age - np.nan_to_num(self.swallowed_at, nan=1e9)
                                   >= self.SWALLOW))

    @property
    def done(self):
        return self.received >= len(self.signs)


# ----------------------------------------------------------------------- warp

class Warp:
    """One leg of a warp jump: out into the white hole, or back from it.

    Out, the view falls towards the hole with streaks, flashes, and comes out
    inverted with time running backwards, and stays there. Back is the same
    passage the other way. `hold`, if set, sends it back on its own after that
    many seconds, which is how an automatic event avoids stranding the view.
    """

    SPOOL, FLASH, EMERGE = 2.2, 2.5, 4.0
    CROSS = 2.35                     # the moment of the flash

    def __init__(self, back=False, hold=None):
        self.age = 0.0
        self.points = None
        self.back = back
        self.hold = hold

    def zoom(self):
        t = self.age
        if t < self.SPOOL:
            return 1.0 - 0.62 * (t / self.SPOOL) ** 2.2
        if t < self.FLASH:
            return 0.38
        if t < self.EMERGE:
            s = (t - self.FLASH) / (self.EMERGE - self.FLASH)
            return 0.38 + 0.62 * (1 - (1 - s) ** 3)
        return 1.0

    def inverted(self):
        return (self.age >= self.CROSS) != self.back

    def flash(self):
        t = self.age
        if self.SPOOL <= t < self.FLASH:
            return max(0.0, 1.0 - abs(t - self.CROSS) / 0.15)
        if t < self.SPOOL:
            return 0.25 * (t / self.SPOOL) ** 3
        return 0.0

    def direction(self):
        return -1.0 if self.inverted() else 1.0

    @property
    def settled(self):
        return self.age >= self.EMERGE

    @property
    def held(self):
        """Through and out the other side, waiting to be sent back."""
        return self.settled and not self.back

    @property
    def done(self):
        return self.settled and self.back


# -------------------------------------------------------------------- manager

class Effects:
    def __init__(self, seed=0, messages=None, events=True, event_minutes=(4, 10)):
        self.rng = np.random.default_rng(seed + 991)
        self.messages = list(messages) if messages else list(MESSAGES)
        self.hud = False
        self.invert = False             # the picture inverted: a white hole, as a look
        self.probe = None
        self.transmission = None
        self.warp = None
        self.events = events
        self.event_minutes = tuple(event_minutes)
        self.next_event = None
        self.readout = ("", 0.0)        # a line shown for a while after an event
        self.target = None
        self.observed = 0.0
        self.last_cells = set()

    # ---------------------------------------------------------------- control

    def active(self):
        return bool(self.hud or self.invert or self.probe or self.transmission or self.warp)

    def launch_probe(self, live):
        self.probe = Probe(live.mapping, framed_radius(live), self.rng)

    def launch_transmission(self, live):
        text = self.messages[int(self.rng.integers(len(self.messages)))]
        self.transmission = Transmission(text, live.mapping, framed_radius(live), self.rng)

    def launch_warp(self, live, hold=None):
        """Warp out, or back if already out. False while one is under way."""
        if self.warp is None:
            self.warp = Warp(hold=hold)
        elif self.warp.held:
            self.warp = Warp(back=True)
        else:
            return False
        return True

    def warped(self):
        return self.warp is not None and self.warp.held

    def zoom(self):
        return self.warp.zoom() if self.warp else 1.0

    def time_direction(self):
        return self.warp.direction() if self.warp else 1.0

    def step(self, live, now, dt):
        """Move everything on by dt seconds of wall time."""
        if self.events:
            if self.next_event is None:
                self.next_event = now + 60 * float(self.rng.uniform(*self.event_minutes))
            elif now >= self.next_event and not (self.probe or self.transmission
                                                   or (self.warp and not self.warp.held)):
                roll = self.rng.random()
                if roll < 0.45:
                    self.launch_probe(live)
                elif roll < 0.9:
                    self.launch_transmission(live)
                elif live.projector is not None and self.warp is None:
                    self.launch_warp(live, hold=30.0)
                self.next_event = now + 60 * float(self.rng.uniform(*self.event_minutes))
        if live.paused:
            dt = 0.0
        dm = dt * m_per_second(live.mapping, live.rates)
        self.observed += dt

        if self.probe is not None:
            p = self.probe
            p.age += dt
            if p.lost_at is None:
                p.fall(dm)
                p.locate(float(live.mapping["bh"].incl))
                if 1.0 / p.shift() < Probe.LOST_BELOW:
                    p.lost_at = p.age
                    self.readout = (f"PROBE  signal lost  last r {p.r / p.m:.4f} M  "
                                    f"1+z {p.shift():.1f}", now + 6.0)
            elif p.age - p.lost_at > 1.0:
                self.probe = None

        if self.transmission is not None:
            tr = self.transmission
            before = tr.received
            tr.advance(dt, dm)
            if tr.received != before or tr.done:
                got = tr.signs[:tr.received]
                self.readout = (f"RECEIVED  {got}  →  {decode_message(got)}",
                                now + (8.0 if tr.done else 3.0))
            if tr.done:
                self.transmission = None

        if self.warp is not None:
            w = self.warp
            w.age += dt
            if w.done:
                self.warp = None
            elif w.held and w.hold is not None and w.age - w.EMERGE > w.hold:
                self.warp = Warp(back=True)

    # --------------------------------------------------------------- painting

    def paint(self, canvas, live):
        """Draw this frame's marks into the picture, before it is sent."""
        self.canvas = canvas
        w = self.warp
        if w is not None and w.points is None:
            # Streak from where the light is, found before any inversion.
            w.points = canvas.sample(320, self.rng)
        flipped = self.invert != (w is not None and w.inverted())
        if flipped:
            canvas.invert()
        if self.hud:
            self._paint_hud(canvas, live)
        if self.probe is not None:
            self._paint_probe(canvas, live)
        if self.transmission is not None:
            self._paint_swallow(canvas, live)
        if w is not None:
            self._paint_warp(canvas, flipped)

    def _paint_hud(self, canvas, live):
        crit = float(live.mapping["bh"].critical_b)
        # The photon ring: the shadow's edge, dotted.
        a = np.linspace(0, TWO_PI, 180, endpoint=False)[::2]
        line = max(1, canvas.unit // 2)
        # Everything is gathered into one list of marks and drawn in one call.
        rhos, angles, alphas = [], [], []
        a = np.linspace(0, TWO_PI, 180, endpoint=False)[::2]
        rhos.append(np.full_like(a, crit * 1.02))
        angles.append(a)
        alphas.append(np.full_like(a, 0.8))
        # Ticks every thirty degrees, outside the ring.
        for k in range(12):
            rho = np.linspace(crit * 1.12, crit * (1.28 if k % 3 == 0 else 1.2), 6 * canvas.unit)
            rhos.append(rho)
            angles.append(np.full_like(rho, k * TWO_PI / 12))
            alphas.append(np.full_like(rho, 0.7))
        # The sweep, with a short fading wake.
        theta = self.observed / 5.0 * TWO_PI
        reach = max(canvas.ext_x, canvas.ext_y) * 1.5
        rho = np.linspace(crit * 1.3, reach, 160 * canvas.unit)
        wake = 10 if canvas.kind == "pixels" else 1
        for k in range(wake + 1):
            rhos.append(rho)
            angles.append(np.full_like(rho, theta - k * 0.005))
            alphas.append(np.full_like(rho, 0.75 * (1 - k / (wake + 1)) ** 2))
        xf, yf = to_screen(canvas, np.concatenate(rhos), np.concatenate(angles))
        canvas.plot(xf, yf, HUD, np.concatenate(alphas), size=line)

    def _paint_probe(self, canvas, live):
        p = self.probe
        g = 1.0 / p.shift()
        fade = 1.0 if p.lost_at is None else max(0.0, 1.0 - (p.age - p.lost_at))
        colour = temperature(np.array(g))
        # Tidal stretching, exaggerated to be seen: along the line to the hole.
        stretch = 1.0 + 30.0 * (2 * p.m / p.r) ** 3
        for order, weight in ((0, 1.0), (1, 0.45)):
            b = p.images[order]
            if b is None:
                continue
            show = min(1.0, 1.4 * g ** 2) * weight * fade
            if show <= 0.02:
                continue
            length = 0.35 * p.m * stretch
            rho = np.linspace(b - length / 2, b + length / 2, 12)
            xf, yf = to_screen(canvas, rho, np.full_like(rho, p.alpha))
            size = max(1, canvas.unit)
            canvas.plot(xf, yf, colour, show, size=size)
            # A beacon, blinking.
            if (p.age % 0.9) < 0.12:
                xf, yf = to_screen(canvas, np.array([b]), np.array([p.alpha]))
                canvas.plot(xf, yf, (255, 255, 255), show, size=max(3, 2 * canvas.unit))

    def _paint_swallow(self, canvas, live):
        tr = self.transmission
        crit = float(live.mapping["bh"].critical_b)
        since = tr.age - tr.swallowed_at
        flaring = np.flatnonzero(np.isfinite(since) & (since >= 0) & (since < tr.SWALLOW))
        for i in flaring:
            b, _ = project(live.mapping, tr.r_in * 1.001, tr.alpha[i], 0)
            angle = tr.alpha[i]
            if not np.isfinite(b[0]):
                continue
            # Where the sign was, then along the ring as a flare.
            s = since[i] / tr.SWALLOW
            span = 0.1 + 0.5 * s
            arc = np.linspace(angle - span, angle + span, 40 * canvas.unit)
            xf, yf = to_screen(canvas, crit * 1.01, arc)
            canvas.plot(xf, yf, (255, 250, 235), (1 - s) * 0.9, size=max(1, canvas.unit // 2))

    def _paint_warp(self, canvas, flipped):
        w = self.warp
        px, py = w.points
        z = w.zoom()
        if w.age < w.SPOOL:
            tail = min(1.0, z * 1.7)
            strength = (w.age / w.SPOOL) ** 1.5
        elif w.FLASH <= w.age < w.EMERGE:
            tail = z * 0.72
            strength = max(0.0, 1.0 - (w.age - w.FLASH) / 1.4)
        else:
            tail, strength = z, 0.0
        if strength > 0.02 and px.size:
            t = np.linspace(0.0, 1.0, 18 * canvas.unit)[None, :]
            head_x = 0.5 + (px[:, None] - 0.5) / z
            head_y = 0.5 + (py[:, None] - 0.5) / z
            tail_x = 0.5 + (px[:, None] - 0.5) / tail
            tail_y = 0.5 + (py[:, None] - 0.5) / tail
            xs = (tail_x + (head_x - tail_x) * t).ravel()
            ys = (tail_y + (head_y - tail_y) * t).ravel()
            alpha = np.broadcast_to(np.minimum(1.0, 1.6 * t * strength), (px.size, t.size)).ravel()
            streak = (30, 20, 0) if flipped else (225, 235, 255)
            canvas.plot(xs, ys, streak, alpha, size=max(1, canvas.unit // 3))
        canvas.wash((255, 255, 255), w.flash())

    # ---------------------------------------------------------------- overlay

    def overlay(self, live, clear):
        """Glyphs drawn over the picture: signs, readouts, the reticle.

        Over pixels the picture never overwrites text, so cells written last
        frame and not this one are blanked; over braille the next frame does it.
        """
        canvas = getattr(self, "canvas", None)
        out, used = [], set()
        rows, cols = live.rows, live.cols

        def put(row, col, text, rgb):
            if not (0 <= row < rows) or col < 0 or col >= cols:
                return
            text = text[:max(0, cols - col)]
            bg = canvas.bg_at(row, min(col, cols - 1)) if canvas is not None else None
            back = f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m" if bg else ""
            out.append(f"\033[{row + 1};{col + 1}H\033[0m"
                       f"\033[38;2;{int(rgb[0])};{int(rgb[1])};{int(rgb[2])}m{back}{text}")
            for k in range(len(text)):
                used.add((row, col + k))

        if canvas is not None and self.transmission is not None:
            self._overlay_signs(canvas, live, put)
        if canvas is not None and self.hud:
            self._overlay_hud(canvas, live, put)
        if self.probe is not None and self.probe.lost_at is None:
            p = self.probe
            g = 1.0 / p.shift()
            bars = int(round(8 * min(1.0, g ** 2)))
            put(rows - 2, 1, f"PROBE  r {p.r / p.m:7.3f} M  1+z {p.shift():6.2f}  "
                             f"signal {'▮' * bars}{'▯' * (8 - bars)}",
                tuple(int(v) for v in temperature(np.array(max(g, 0.3)))))
        text, until = self.readout
        if text and time.monotonic() < until:
            put(rows - 2, 1, text, HUD)

        if clear:
            for row, col in self.last_cells - used:
                out.insert(0, f"\033[{row + 1};{col + 1}H\033[0m ")
        self.last_cells = used
        return "".join(out) + "\033[0m"

    def _overlay_signs(self, canvas, live, put):
        tr = self.transmission
        flying = np.flatnonzero((tr.age >= tr.launch) & np.isnan(tr.swallowed_at))
        if flying.size == 0:
            return
        taken = set()
        for order, weight in ((0, 1.0), (1, 0.5)):
            b, z = project(live.mapping, tr.r[flying], tr.alpha[flying], order)
            xf, yf = to_screen(canvas, b, tr.alpha[flying])
            colours = shift_colour(z)
            for j, i in enumerate(flying):
                if not (np.isfinite(b[j]) and 0 <= xf[j] <= 1 and 0 <= yf[j] <= 1):
                    continue
                row, col = canvas.cell(xf[j], yf[j])
                if (row, col) in taken or (row, col - 1) in taken or (row, col + 1) in taken:
                    continue
                taken.add((row, col))
                rise = min(1.0, (tr.age - tr.launch[i]) / 0.6)
                put(row, col, tr.signs[i], colours[j] * (0.35 + 0.65 * rise) * weight)

    def _overlay_hud(self, canvas, live, put):
        s = live.settings
        mapping = live.mapping
        put(0, 1, f"INCL {seven_segment(f'{np.degrees(s['incl']):05.1f}')}°", HUD)
        put(1, 1, f"MASS {seven_segment(f'{s['mass']:4.2f}')} M", HUD)
        put(2, 1, f"DISK {seven_segment(f'{s['outer_edge'] / s['mass']:03.0f}')} M", HUD)
        clock = f"T+ {seven_segment(f'{self.observed:08.1f}')}"
        put(0, live.cols - len(clock) - 2, clock, HUD)

        # A parcel of gas to follow, replaced when it goes behind the hole or
        # off the edge, and every twenty seconds anyway.
        mass = float(mapping["bh"].mass)
        t = self.target
        if t is None or self.observed - t["since"] > 20.0 or t.get("lost", 0) > 1.0:
            r = float(self.rng.uniform(7.0, 13.0)) * mass
            t = self.target = {"r": r, "a0": float(self.rng.uniform(0, TWO_PI)),
                               "since": self.observed, "lost": 0.0}
        rate = float(np.interp(t["r"], mapping["radii"], live.rates))
        angle = t["a0"] + rate * live.clock
        b, z = project(mapping, t["r"], angle, 0)
        xf, yf = to_screen(canvas, b[0], angle)
        if not (np.isfinite(b[0]) and 0.02 < xf < 0.98 and 0.02 < yf < 0.98):
            t["lost"] = t.get("lost", 0.0) + 1.0 / max(live.o.fps, 1)
            return
        t["lost"] = 0.0
        row, col = canvas.cell(xf, yf)
        # Closing in on a new lock.
        settle = min(1.0, (self.observed - t["since"]) / 0.5)
        dx = int(round(2 + 6 * (1 - settle)))
        dy = int(round(1 + 3 * (1 - settle)))
        put(row - dy, col - dx, "┌", HUD)
        put(row - dy, col + dx, "┐", HUD)
        put(row + dy, col - dx, "└", HUD)
        put(row + dy, col + dx, "┘", HUD)
        label = f"r {seven_segment(f'{t['r'] / mass:4.1f}')}M 1+z {seven_segment(f'{z[0]:5.3f}')}"
        put(row - dy, col + dx + 2, label, HUD)
