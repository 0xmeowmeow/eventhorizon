"""The line families from the luminet docs, drawn live over plot1979.

    radii       isoradials: where light from one ring of the disk lands. Direct
                and ghost images, at chosen radii.
    redshift    isoredshifts: lines of equal redshift factor, 1 + z.
    flux        isofluxlines: contours of equal brightness. The docs suggest
                spacing the levels logarithmically, and so do these.

Isoradials are read out of the lensing map ring by ring. The other two are
contours of per-cell maps the dot field already measures, so none of the three
costs a new solve.

Each can be coloured flat, by the redshift or brightness along the line, from
the dot palette, or one hue per line, and drawn in one of five styles:

    solid       as the docs draw them
    flowing     dashes travelling with the gas: at each ring's own orbital rate
                for isoradials, and at one rate about the centre for contours
    dotted      a fixed dotted line
    pulse       a wave of brightness passing from line to line, outermost first,
                so for isoradials it travels inwards
    sweep       one line moving continuously through the range - an isoradial
                falling from the outer edge to the inner, or a redshift or
                brightness level drifting across the disk - over the rest dimmed
"""

import numpy as np

from luminet import black_hole_math as bhmath
from luminet import cells

FAMILIES = ("radii", "redshift", "flux")
STYLES = ("solid", "flowing", "dotted", "pulse", "sweep")
COLOURS = ("blue", "ink", "palette", "redshift", "flux", "spectrum")

BLUE = np.array([150, 205, 235], np.float32)
INK = np.array([238, 230, 210], np.float32)
DIVERGING = [(70, 130, 255), (170, 205, 255), (245, 245, 245), (255, 170, 140), (230, 60, 40)]

# The levels drawn in the docs' own example.
DEFAULT_REDSHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
DEFAULT_FLUX_LEVELS = tuple(np.round(np.geomspace(0.003, 1.0, 8), 4))


def _smooth(grid, sigma=1.2):
    """Blur a per-cell map without letting empty sky bleed into it.

    The maps are measured from samples, and a contour faithfully follows every
    bit of that noise. A light blur, weighted so NaN cells count for nothing,
    steadies the lines without moving them off the light they describe.
    """
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(grid)
    if not finite.any():
        return grid
    values = gaussian_filter(np.where(finite, grid, 0.0), sigma)
    weight = gaussian_filter(finite.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(finite, values / np.maximum(weight, 1e-6), np.nan)


def _upsample(grid, width, height):
    """A per-cell map resampled onto the canvas; NaN stays NaN."""
    gh, gw = grid.shape
    if (gw, gh) == (width, height):
        return grid.astype(np.float32)
    from scipy.ndimage import zoom

    finite = np.isfinite(grid)
    filled = np.where(finite, grid, 0.0).astype(np.float32)
    zy, zx = height / gh, width / gw
    values = zoom(filled, (zy, zx), order=1)[:height, :width]
    weight = zoom(finite.astype(np.float32), (zy, zx), order=1)[:height, :width]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0.5, values / np.maximum(weight, 1e-6), np.nan)
    return out


def _crossings(field, level):
    """Canvas positions where field passes through level: a contour line."""
    above = field > level
    finite = np.isfinite(field)
    edge = np.zeros_like(above)
    both_x = finite[:, 1:] & finite[:, :-1]
    edge[:, 1:] |= (above[:, 1:] != above[:, :-1]) & both_x
    both_y = finite[1:, :] & finite[:-1, :]
    edge[1:, :] |= (above[1:, :] != above[:-1, :]) & both_y
    return np.flatnonzero(edge)


class LineSet:
    """Every line family, ready to draw at any moment."""

    def __init__(self, mapping, field, width, height, ext_x, ext_y,
                 radii=(6, 10, 15, 20), ghost=(6, 20, 50, 100),
                 redshifts=DEFAULT_REDSHIFTS, flux_levels=DEFAULT_FLUX_LEVELS):
        self.mapping = mapping
        self.width, self.height = width, height
        self.ext_x, self.ext_y = ext_x, ext_y
        bh = mapping["bh"]
        self.scale = float(getattr(field, "scale", 1.0))

        # Isoradials, outermost first so a pulse runs inwards.
        chosen = [(0, r) for r in radii] + [(1, r) for r in ghost]
        chosen.sort(key=lambda item: -item[1])
        self.rings = []
        for rank, (order, radius) in enumerate(chosen):
            line = self._ring_line(order, float(radius))
            if line is not None:
                self.rings.append({**line, "rank": rank, "order": order})
        self.direct_span = (min(radii), max(radii))

        # Contour maps on the canvas, and each level's line on them.
        self.z_map = _upsample(_smooth(field.redshift), width, height)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.flux_map = _upsample(_smooth(np.log10(field.flux_level)), width, height)
        self.redshift_levels = [1.0 + z for z in redshifts]
        self.flux_levels = [float(np.log10(level)) for level in flux_levels]
        self.contours = {
            "redshift": [(i, lvl, _crossings(self.z_map, lvl))
                         for i, lvl in enumerate(self.redshift_levels)],
            "flux": [(i, lvl, _crossings(self.flux_map, lvl))
                     for i, lvl in enumerate(self.flux_levels)],
        }

        # Angle about the centre of every canvas position, for dashes and dots
        # on contours, which have no orbit of their own to follow.
        ys, xs = np.divmod(np.arange(width * height), width)
        bx = (xs / max(width - 1, 1) * 2 - 1) * ext_x
        by = (ys / max(height - 1, 1) * 2 - 1) * ext_y
        self.theta = np.arctan2(bx, by).astype(np.float32) % (2 * np.pi)
        rates_ref = np.argmin(np.abs(mapping["radii"] - 10.0))
        self.contour_ring = int(rates_ref)
        self.bh = bh

    # ------------------------------------------------------------------ rings

    def _ring_line(self, order, radius):
        """One isoradial at a radius, which may fall between solved rings."""
        tables = self.mapping["tables"]
        if order not in tables:
            return None
        radii, angles = self.mapping["radii"], self.mapping["angles"]
        if radius < radii[0] or radius > radii[-1]:
            return None
        # Between two solved rings, blend them by where the radius sits in log
        # radius: rings may be spaced geometrically, and that is the even way.
        hi = int(np.searchsorted(radii, radius))
        hi = min(max(hi, 1), len(radii) - 1)
        lo = hi - 1
        span = np.log(radii[hi]) - np.log(radii[lo])
        f = 0.0 if span == 0 else (np.log(radius) - np.log(radii[lo])) / span
        b_t, z_t = tables[order]
        b_row = (1 - f) * b_t[lo] + f * b_t[hi]
        z_row = (1 - f) * z_t[lo] + f * z_t[hi]
        good = np.isfinite(b_row) & np.isfinite(z_row)
        if good.sum() < 4:
            return None
        na = len(angles)
        # Sample by the ring's length on screen, at about two samples a pixel.
        # A fixed count was ten times more than the largest ring needed and far
        # more than the small ones, and every extra sample is painted again when
        # lines are thickened.
        probe = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
        pb = np.interp(probe, angles[good], b_row[good], period=2 * np.pi)
        px = pb * np.sin(probe) / self.ext_x * (self.width - 1) / 2
        py = pb * np.cos(probe) / self.ext_y * (self.height - 1) / 2
        length = float(np.hypot(np.diff(px, append=px[:1]), np.diff(py, append=py[:1])).sum())
        dense = np.linspace(0.0, 2 * np.pi, int(np.clip(2.0 * length, 256, 20000)),
                            endpoint=False)
        nearest = np.rint(dense / (2 * np.pi) * (na - 1)).astype(int) % na
        b = np.interp(dense, angles[good], b_row[good], period=2 * np.pi)
        z = np.interp(dense, angles[good], z_row[good], period=2 * np.pi)
        valid = good[nearest]
        ci = ((b * np.sin(dense) / self.ext_x + 1) / 2 * (self.width - 1)).astype(np.int64)
        ri = ((b * np.cos(dense) / self.ext_y + 1) / 2 * (self.height - 1)).astype(np.int64)
        inside = valid & (ci >= 0) & (ci < self.width) & (ri >= 0) & (ri < self.height)
        bh = self.mapping["bh"]
        with np.errstate(invalid="ignore", divide="ignore"):
            flux = bhmath.calc_flux_observed(radius, bh.acc, bh.mass, z[inside])
        return {
            "radius": radius,
            "ring": lo if f < 0.5 else hi,
            "alpha": dense[inside],
            "idx": ri[inside] * self.width + ci[inside],
            "z": z[inside],
            "flux": np.log10(np.maximum(flux / max(self.scale, 1e-30), 1e-12)),
        }

    # ---------------------------------------------------------------- drawing

    def draw(self, t, rates, families, style="solid", colour="blue", palette=None,
             width=1, dashes=14, duty=0.55):
        """Canvas positions to light and their colours, for time t.

        The shape of every line is fixed for a given choice of families, colour
        and width; only which parts show, and how brightly, changes with time.
        So the lines are gathered, de-duplicated and thickened once into a
        geometry that remembers which sample each painted pixel came from, and
        a frame only switches samples on or off (flowing) or scales their
        brightness (pulse). Solid and dotted lines are the geometry itself.
        """
        palette = palette or cells.palette("inferno")
        if getattr(self, "rates", None) is None or len(self.rates) != len(rates) \
                or not np.array_equal(self.rates, rates):
            self.rates = np.array(rates, dtype=float)
            self._geo = None             # flowing speeds are baked into the geometry
        geo = self._geometry(families, style if style == "dotted" else "solid",
                             colour, palette, width)
        if geo is None:
            empty = (np.zeros(0, np.int64), np.zeros((0, 3), np.uint8))
            return empty if style != "sweep" else self._sweep_only(t, families, colour,
                                                                    palette, width)
        owner, idx, base = geo["owner"], geo["idx"], geo["rgb"]

        if style in ("solid", "dotted"):
            return idx, geo["rgb_u8"]
        if style == "flowing":
            keep = ((geo["angle"] - geo["rate"] * t) * dashes / (2 * np.pi)) % 1.0 < duty
            on = keep[owner]
            return idx[on], geo["rgb_u8"][on]
        if style == "pulse":
            period = 3.0
            phase = geo["rank"] / geo["count"] - (t % period) / period
            level = 0.18 + 0.82 * (0.5 + 0.5 * np.cos(2 * np.pi * phase)) ** 3
            return idx, np.clip(base * level[owner][:, None], 0, 255).astype(np.uint8)

        # sweep: the fixed lines dimmed, one moving line per family bright on top
        still_idx = idx
        still_rgb = np.clip(base * 0.22, 0, 255).astype(np.uint8)
        moving_idx, moving_rgb = self._sweep_only(t, families, colour, palette, width)
        return (np.concatenate([still_idx, moving_idx]),
                np.concatenate([still_rgb, moving_rgb]))

    def _geometry(self, families, style, colour, palette, width):
        key = (frozenset(families), style, colour, tuple(map(tuple, palette)), width)
        cached = getattr(self, "_geo", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        idx_parts, rgb_parts, angle_parts, rate_parts, rank_parts, count_parts = [], [], [], [], [], []
        n_rings = max(len(self.rings), 1)
        if "radii" in families:
            for line in self.rings:
                keep = self._style_mask(style, line["alpha"], 0.0, 0.0, 14, 0.55)
                k = int(keep.sum())
                idx_parts.append(line["idx"][keep])
                rgb_parts.append(self._colours(colour, line["z"][keep], line["flux"][keep],
                                               line["rank"], n_rings, palette))
                angle_parts.append(line["alpha"][keep])
                rate_parts.append(np.full(k, 0.0))
                rank_parts.append(np.full(k, float(line["rank"])))
                count_parts.append(np.full(k, float(n_rings)))
                rate_parts[-1][:] = self.mapping_rate(line["ring"])
        for family in ("redshift", "flux"):
            if family not in families:
                continue
            levels = self.contours[family]
            n = max(len(levels), 1)
            for rank, value, idx in levels:
                theta = self.theta[idx]
                keep = self._style_mask(style, theta, 0.0, 0.0, 14, 0.55)
                k = int(keep.sum())
                z = np.full(k, value if family == "redshift" else 1.0)
                f = np.full(k, value if family == "flux" else -1.0)
                idx_parts.append(idx[keep])
                rgb_parts.append(self._colours(colour, z, f, rank, n, palette))
                angle_parts.append(theta[keep])
                rate_parts.append(np.full(k, self.mapping_rate(self.contour_ring)))
                rank_parts.append(np.full(k, float(rank)))
                count_parts.append(np.full(k, float(n)))
        if not idx_parts:
            self._geo = (key, None)
            return None

        idx = np.concatenate(idx_parts)
        rgb = np.concatenate(rgb_parts)
        angle = np.concatenate(angle_parts)
        rate = np.concatenate(rate_parts)
        rank = np.concatenate(rank_parts)
        count = np.concatenate(count_parts)
        # Many samples land on one pixel; keep one sample per pixel.
        last = len(idx) - 1 - np.unique(idx[::-1], return_index=True)[1]
        idx, rgb, angle, rate, rank, count = (idx[last], rgb[last], angle[last],
                                               rate[last], rank[last], count[last])
        owner = np.arange(len(idx))
        if width > 1:
            idx, owner = self._thicken_owned(idx, width)
        geo = {"idx": idx, "owner": owner, "rgb": rgb[owner],
               "rgb_u8": np.clip(rgb[owner], 0, 255).astype(np.uint8),
               "angle": angle, "rate": rate, "rank": rank, "count": count}
        self._geo = (key, geo)
        return geo

    def mapping_rate(self, ring):
        rates = getattr(self, "rates", None)
        return float(rates[ring]) if rates is not None else 0.0

    def _sweep_only(self, t, families, colour, palette, width):
        parts = []
        if "radii" in families:
            parts.extend(self._sweep_ring(t, colour, palette))
        for family in ("redshift", "flux"):
            if family in families:
                parts.extend(self._sweep_contour(family, t, colour, palette))
        if not parts:
            return np.zeros(0, np.int64), np.zeros((0, 3), np.uint8)
        idx = np.concatenate([p[0] for p in parts])
        rgb = np.concatenate([p[1] * p[2] for p in parts])
        if width > 1:
            idx, owner = self._thicken_owned(idx, width)
            rgb = rgb[owner]
        return idx, np.clip(rgb, 0, 255).astype(np.uint8)

    def _style_mask(self, style, angle, rate, t, dashes, duty):
        if style == "flowing":
            return ((angle - rate * t) * dashes / (2 * np.pi)) % 1.0 < duty
        if style == "dotted":
            return (angle * 90 / (2 * np.pi)) % 1.0 < 0.34
        return np.ones(angle.shape, dtype=bool)

    @staticmethod
    def _pulse(style, rank, n, t, period=3.0):
        if style != "pulse":
            return 1.0
        # One crest passes every line in turn, outermost first.
        phase = rank / n - (t % period) / period
        return 0.18 + 0.82 * (0.5 + 0.5 * np.cos(2 * np.pi * phase)) ** 3

    def _colours(self, mode, z, flux, rank, n, palette):
        count = len(z)
        if mode == "ink":
            return np.broadcast_to(INK, (count, 3)).astype(np.float32)
        if mode == "palette":
            return np.broadcast_to(np.array(palette[len(palette) * 3 // 4], np.float32),
                                   (count, 3)).astype(np.float32)
        if mode == "redshift":
            return cells._ramp(np.clip((z - 1.0) / 0.8 + 0.5, 0, 1), DIVERGING).astype(np.float32)
        if mode == "flux":
            return cells._ramp(np.clip((flux + 3.0) / 3.0, 0, 1), palette).astype(np.float32)
        if mode == "spectrum":
            hue = rank / max(n, 1)
            rgb = 255 * np.array([0.5 + 0.5 * np.cos(2 * np.pi * (hue + k / 3.0))
                                  for k in (0, 1, 2)], np.float32)
            return np.broadcast_to(rgb, (count, 3)).astype(np.float32)
        return np.broadcast_to(BLUE, (count, 3)).astype(np.float32)

    def _sweep_ring(self, t, colour, palette, period=8.0):
        """An isoradial falling from the outer to the inner radius, then again."""
        lo, hi = self.direct_span
        u = (t % period) / period
        radius = float(np.exp(np.log(hi) + u * (np.log(lo) - np.log(hi))))
        out = []
        for order in (0, 1):
            line = self._ring_line(order, radius)
            if line is None:
                continue
            out.append((line["idx"], self._colours(colour, line["z"], line["flux"],
                                                   int(u * 10), 10, palette), 1.0))
        return out

    def _sweep_contour(self, family, t, colour, palette, period=8.0):
        """One level drifting through the family's range, back and forth."""
        levels = self.redshift_levels if family == "redshift" else self.flux_levels
        if len(levels) < 2:
            return []
        u = 0.5 - 0.5 * np.cos(2 * np.pi * (t % period) / period)
        value = levels[0] + u * (levels[-1] - levels[0])
        field = self.z_map if family == "redshift" else self.flux_map
        idx = _crossings(field, value)
        z = np.full(len(idx), value if family == "redshift" else 1.0)
        f = np.full(len(idx), value if family == "flux" else -1.0)
        return [(idx, self._colours(colour, z, f, int(u * 10), 10, palette), 1.0)]

    def _thicken_owned(self, idx, width):
        """Widen lines, returning each painted pixel and the sample it belongs to."""
        r = (width - 1) / 2.0
        k = int(np.ceil(r))
        rows, cols = np.divmod(idx, self.width)
        sample = np.arange(len(idx))
        out_i, out_o = [], []
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy > r * r + 0.5:
                    continue
                rr, cc = rows + dy, cols + dx
                ok = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
                out_i.append(rr[ok] * self.width + cc[ok])
                out_o.append(sample[ok])
        idx_all = np.concatenate(out_i)
        owner = np.concatenate(out_o)
        # A pixel reached from two samples is painted once.
        first = np.unique(idx_all, return_index=True)[1]
        return idx_all[first], owner[first]
