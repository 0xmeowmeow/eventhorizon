"""The per-frame hot loop, compiled.

Almost all of a plot1979 frame is the same few lines of arithmetic applied to
every parcel: where it has got to in its orbit, what the lensing map says about
that position, how bright it is, and which cell it lands in. In numpy each of
those lines builds a temporary array the length of the whole gas, so a frame
spends its time allocating and walking memory rather than computing. Compiled,
the lines fuse into one pass per parcel, and the parcels are independent, so the
pass can be split across cores.

It runs on one thread, deliberately. Measured through the launcher at full
screen, what a laptop cares about is CPU time rather than how soon a frame is
ready, and spreading the loop over sixteen threads made each frame finish sooner
while costing more CPU in total than plain numpy - 112% of a core at 20fps,
against 99% for numpy and 57% for a single compiled thread. At 10fps the single
thread uses 30%.

This stays on the CPU on purpose too. A GPU would save little from a few
milliseconds of arithmetic, and keeping one awake all day is a poor trade for
something meant to sit on a laptop desktop.

Numba is optional. Without it, spin.dots falls back to the numpy version, which
computes the same thing: the two light exactly the same dots, with brightness
agreeing to about 1e-16. tests/test_fast.py holds them to that, comparing the
whole of dots() rather than a hand-copied reference - a copied reference once
agreed with a kernel that drew the picture upside down, because it had copied
the same sign.
"""

import numpy as np

try:
    import numba as nb

    available = True
except ImportError:  # pragma: no cover - numba is an optional extra
    nb = None
    available = False

# Which path frames are drawn with, for the status line. The fallback is silent
# by design, and silent turned out to be hard to notice: a harness once ran for
# a whole session on an interpreter without numba and measured the slow path.
PATH_NAME = "compiled" if available else "numpy"


if available:

    # Deliberately not fastmath. That flag lets the compiler assume values are
    # never NaN or infinite and drop the checks for them, and the lensing map
    # uses NaN to mean "no light reaches here".
    @nb.njit(cache=True)
    def _project(ring, jitter, angle0, radii, rates, b_tab, z_tab, has, phase,
                 mass, acc, ext_x, ext_y, width, height, out_idx, out_flux, out_z):
        n = ring.shape[0]
        nr = radii.shape[0]
        na = b_tab.shape[2]
        two_pi = 2.0 * np.pi
        s3 = np.sqrt(3.0)
        s6 = np.sqrt(6.0)
        for i in range(n):
            rg = ring[i]
            up = rg + 1 if rg + 1 < nr else nr - 1
            fr = jitter[i]
            ang = (angle0[i] + rates[rg] * phase) % two_pi

            col = ang / two_pi * (na - 1)
            floor_col = np.floor(col)
            lo = int(floor_col) % na
            hi = (lo + 1) % na
            ft = col - floor_col

            # Intrinsic flux of a Page-Thorne disk at this radius; the observed
            # flux divides it by the redshift factor to the fourth.
            r = (1.0 - fr) * radii[rg] + fr * radii[up]
            rr = r / mass
            sr = np.sqrt(rr)
            a = 3.0 * mass * acc / (8.0 * np.pi) / ((rr - 3.0) * rr ** 2.5)
            log_arg = (sr + s3) * (s6 - s3) / ((sr - s3) * (s6 + s3))
            intrinsic = a * (sr - s6 + (s3 / 2.0) * np.log(log_arg))

            sa = np.sin(ang)
            ca = np.cos(ang)
            for o in range(2):
                out_idx[i, o] = -1
                if not has[o]:
                    continue
                b = ((1.0 - fr) * ((1.0 - ft) * b_tab[o, rg, lo] + ft * b_tab[o, rg, hi])
                     + fr * ((1.0 - ft) * b_tab[o, up, lo] + ft * b_tab[o, up, hi]))
                z = ((1.0 - fr) * ((1.0 - ft) * z_tab[o, rg, lo] + ft * z_tab[o, rg, hi])
                     + fr * ((1.0 - ft) * z_tab[o, up, lo] + ft * z_tab[o, up, hi]))
                if not (np.isfinite(b) and np.isfinite(z)):
                    continue
                flux = intrinsic / z ** 4
                if o == 1:
                    flux *= 0.45
                if not np.isfinite(flux):
                    continue
                # Screen y is -b cos(angle), and rows count downwards from the
                # top, so the row is taken from -y: the signs cancel.
                ci = int((b * sa / ext_x + 1.0) / 2.0 * (width - 1))
                ri = int((b * ca / ext_y + 1.0) / 2.0 * (height - 1))
                if 0 <= ci < width and 0 <= ri < height:
                    out_idx[i, o] = ri * width + ci
                    out_flux[i, o] = flux
                    out_z[i, o] = z


    @nb.njit(cache=True)
    def _place(ring, jitter, angle0, luck, radii, rates, b_tab, has, phase,
               ext_x, ext_y, width, height, chance, lit):
        """Move every parcel and light it or not; no brightness is computed.

        How bright a cell should be does not change over time - a given cell
        always sees gas at the same place on the disk - so that is worked out
        once per map and window and passed in as `chance`. What is left per
        frame is where each parcel has got to, which is only its angle, one
        table lookup and a sine and cosine.
        """
        lit[:] = False
        n = ring.shape[0]
        nr = radii.shape[0]
        na = b_tab.shape[2]
        two_pi = 2.0 * np.pi
        for i in range(n):
            rg = ring[i]
            up = rg + 1 if rg + 1 < nr else nr - 1
            fr = jitter[i]
            ang = (angle0[i] + rates[rg] * phase) % two_pi
            col = ang / two_pi * (na - 1)
            floor_col = np.floor(col)
            lo = int(floor_col) % na
            hi = (lo + 1) % na
            ft = col - floor_col
            sa = np.sin(ang)
            ca = np.cos(ang)
            for o in range(2):
                if not has[o]:
                    continue
                b = ((1.0 - fr) * ((1.0 - ft) * b_tab[o, rg, lo] + ft * b_tab[o, rg, hi])
                     + fr * ((1.0 - ft) * b_tab[o, up, lo] + ft * b_tab[o, up, hi]))
                if not np.isfinite(b):
                    continue
                ci = int((b * sa / ext_x + 1.0) / 2.0 * (width - 1))
                ri = int((b * ca / ext_y + 1.0) / 2.0 * (height - 1))
                if 0 <= ci < width and 0 <= ri < height:
                    k = ri * width + ci
                    if luck[i] < chance[o, k]:
                        lit[k] = True


    @nb.njit(cache=True)
    def _paint(ring, jitter, angle0, luck, radii, rates, b_tab, has, phase,
               ext_x, ext_y, grid_w, grid_h, chance, cell_rgb, img, dot):
        """Paint this frame's dots straight into an RGB image, at pixel resolution.

        The decision to show a parcel is the same one _place makes, against a
        coarse grid's chance. What changes is where the dot goes: at the
        parcel's own position in pixels, not snapped to a character cell, so at
        full resolution each dot is a few pixels of ink.
        """
        px_h = img.shape[0]
        px_w = img.shape[1]
        n = ring.shape[0]
        nr = radii.shape[0]
        na = b_tab.shape[2]
        two_pi = 2.0 * np.pi
        for i in range(n):
            rg = ring[i]
            up = rg + 1 if rg + 1 < nr else nr - 1
            fr = jitter[i]
            ang = (angle0[i] + rates[rg] * phase) % two_pi
            col = ang / two_pi * (na - 1)
            floor_col = np.floor(col)
            lo = int(floor_col) % na
            hi = (lo + 1) % na
            ft = col - floor_col
            sa = np.sin(ang)
            ca = np.cos(ang)
            for o in range(2):
                if not has[o]:
                    continue
                b = ((1.0 - fr) * ((1.0 - ft) * b_tab[o, rg, lo] + ft * b_tab[o, rg, hi])
                     + fr * ((1.0 - ft) * b_tab[o, up, lo] + ft * b_tab[o, up, hi]))
                if not np.isfinite(b):
                    continue
                xf = (b * sa / ext_x + 1.0) / 2.0
                yf = (b * ca / ext_y + 1.0) / 2.0
                gc = int(xf * (grid_w - 1))
                gr = int(yf * (grid_h - 1))
                if gc < 0 or gc >= grid_w or gr < 0 or gr >= grid_h:
                    continue
                c = gr * grid_w + gc
                if luck[i] >= chance[o, c]:
                    continue
                px = int(xf * (px_w - 1))
                py = int(yf * (px_h - 1))
                for dy in range(dot):
                    for dx in range(dot):
                        yy = py + dy
                        xx = px + dx
                        if 0 <= yy < px_h and 0 <= xx < px_w:
                            img[yy, xx, 0] = cell_rgb[c, 0]
                            img[yy, xx, 1] = cell_rgb[c, 1]
                            img[yy, xx, 2] = cell_rgb[c, 2]


class Projector:
    """Holds the lensing map in the flat form the compiled loop reads.

    Stacking the tables and allocating the output once per map, rather than per
    frame, keeps the frame itself free of allocation.
    """

    def __init__(self, mapping, orders):
        tables = mapping["tables"]
        shape = next(iter(tables.values()))[0].shape
        self.b = np.full((2, *shape), np.nan)
        self.z = np.full((2, *shape), np.nan)
        self.has = np.zeros(2, dtype=np.bool_)
        for o in (0, 1):
            if o in orders and o in tables:
                self.b[o], self.z[o] = tables[o]
                self.has[o] = True
        self.radii = np.ascontiguousarray(mapping["radii"], dtype=np.float64)
        self.mass = float(mapping["bh"].mass)
        self.acc = float(mapping["bh"].acc)
        self._idx = None
        self._flux = None

    def __call__(self, parcels, phase, rates, ext_x, ext_y, width, height):
        """Cell index, observed flux and lifetime chance of every lit sample."""
        n = parcels.count
        if self._idx is None or self._idx.shape[0] != n:
            self._idx = np.empty((n, 2), dtype=np.int64)
            self._flux = np.zeros((n, 2))
            self._z = np.zeros((n, 2))
        _project(parcels.ring, parcels.jitter, parcels.angle0, self.radii,
                 np.ascontiguousarray(rates, dtype=np.float64), self.b, self.z,
                 self.has, float(phase), self.mass, self.acc, float(ext_x),
                 float(ext_y), int(width), int(height), self._idx, self._flux, self._z)
        flat = self._idx.ravel()
        keep = flat >= 0
        luck = np.repeat(parcels.luck, 2)
        return flat[keep], self._flux.ravel()[keep], luck[keep]

    def paint(self, parcels, phase, rates, ext_x, ext_y, grid_w, grid_h, chance,
              cell_rgb, img, dot):
        """Paint this frame's dots into img, at pixel resolution."""
        _paint(parcels.ring, parcels.jitter, parcels.angle0, parcels.luck, self.radii,
               np.ascontiguousarray(rates, dtype=np.float64), self.b, self.has,
               float(phase), float(ext_x), float(ext_y), int(grid_w), int(grid_h),
               chance, cell_rgb, img, int(dot))

    def place(self, parcels, phase, rates, ext_x, ext_y, width, height, chance, lit):
        """Light this frame's dots from a precomputed per-cell chance."""
        _place(parcels.ring, parcels.jitter, parcels.angle0, parcels.luck, self.radii,
               np.ascontiguousarray(rates, dtype=np.float64), self.b, self.has,
               float(phase), float(ext_x), float(ext_y), int(width), int(height),
               chance, lit)
