"""Put the disk in motion by moving the gas, not the picture.

The renderer is time-averaged and the disk is axially symmetric, so turning it
produces an identical image. Nothing about rotation is visible in a render
because there is nothing in it that is in one place rather than another.

What moves is the material. The lensing map is fixed: for a parcel of gas at
radius r and disk angle alpha, where its light lands on the detector, how far it
is shifted and how bright it looks are all static functions of (r, alpha). So
the map is solved once, and parcels are then carried around it, which costs
nothing per frame. That is also what makes the motion honest rather than
decorative: a parcel climbing over the top of the shadow is doing so because the
map says its light bends that way, and it brightens on the approaching side
because its own redshift factor says so.

Parcels orbit at the Schwarzschild rate for a circular orbit,

    omega = sqrt(M / r^3)

so the inner disk laps the outer many times over. That differential shear is the
thing you actually see when a disk rotates.

Looping
-------

A loop has to return every parcel to where it began. Orbital periods at
different radii are not commensurate, so the turns each ring makes per loop are
rounded to whole numbers. Every ring then closes exactly at the end of the loop,
at the cost of each ring's rate being slightly off true Keplerian. The ordering
is preserved - inner rings always turn faster than outer ones - and the ratio is
capped so the fastest ring still advances in readable steps rather than
strobing.

Infall is optional and costs the exact loop: parcels drift inwards and re-enter
at the outer edge, so the ensemble matches at the ends of the loop but the
individual parcels have swapped places.
"""

import numpy as np

from luminet import black_hole_math as bhmath
from luminet.black_hole import BlackHole


def lensing_map(settings, n_rings=48, n_angles=180, orders=(0, 1),
                on_progress=None, batches=10):
    """Solve where light from each (radius, angle) lands, once.

    Returns the radii and angles sampled, and per image order a pair of
    (n_rings, n_angles) tables: the impact parameter and the redshift factor.
    """
    bh = BlackHole(
        mass=float(settings.get("mass", 1.0)),
        incl=float(settings.get("incl", 1.4)),
        acc=float(settings.get("acc", 1.0)),
        outer_edge=float(settings.get("outer_edge", 40.0)),
        angular_resolution=n_angles,
        radial_resolution=n_rings,
    )
    radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, n_rings)
    # Solve a few rings at a time. calc_isoradials accumulates and skips radii it
    # already holds, so this costs almost nothing and lets a caller show real
    # progress rather than a guess at how long is left.
    chunks = np.array_split(radii, min(batches, n_rings)) if on_progress else [radii]
    for done, chunk in enumerate(chunks, start=1):
        bh.calc_isoradials(
            direct_r=chunk if 0 in orders else [],
            ghost_r=chunk if 1 in orders else [],
        )
        if on_progress:
            on_progress(done, len(chunks))

    angles = np.linspace(0, 2 * np.pi, n_angles)
    tables = {}
    for order in orders:
        b = np.full((n_rings, n_angles), np.nan)
        z = np.full((n_rings, n_angles), np.nan)
        for ir in bh.isoradials:
            if ir.order != order:
                continue
            row = int(np.argmin(np.abs(radii - ir.radius)))
            a = np.asarray(ir.angles, dtype=float)
            good = np.isfinite(a) & np.isfinite(np.asarray(ir.impact_parameters, dtype=float))
            if good.sum() < 4:
                continue
            order_by_angle = np.argsort(a[good])
            b[row] = np.interp(angles, a[good][order_by_angle],
                               np.asarray(ir.impact_parameters)[good][order_by_angle],
                               period=2 * np.pi)
            z[row] = np.interp(angles, a[good][order_by_angle],
                               np.asarray(ir.redshift_factors)[good][order_by_angle],
                               period=2 * np.pi)
        tables[order] = (b, z)

    return {"bh": bh, "radii": radii, "angles": angles, "tables": tables}


def ring_turns(radii, mass, frames, max_turns=None, min_frames_per_turn=10):
    """How many whole turns each ring makes in one loop.

    Whole turns are what make the loop close. The count is scaled so the
    outermost ring turns once and the rest follow the Keplerian ratio, then
    capped: a ring given fewer than about ten frames per turn reads as a strobe
    rather than as rotation.
    """
    omega = np.sqrt(mass / radii ** 3)
    ratio = omega / omega[-1]
    ceiling = max_turns or max(1, int(frames // min_frames_per_turn))
    turns = np.clip(np.rint(ratio), 1, ceiling)
    return turns.astype(int)


def true_rates(radii, mass, inner_orbits_per_second=0.12):
    """Angular speed per ring, in radians per second, unrounded.

    The looping version rounds each ring to a whole number of turns so that
    everything returns to its start. Nothing has to return when the frames are
    computed as they are shown, so the true Keplerian rates can be used instead.
    Their ratios are irrational, so no two rings ever come back into the same
    arrangement and the picture never repeats. That is the whole reason a live
    version looks different from a loop, rather than just longer.
    """
    omega = np.sqrt(mass / radii ** 3)
    return omega / omega[0] * inner_orbits_per_second * 2 * np.pi


class Parcels:
    """Gas, spread over the disk, carried around the map."""

    def __init__(self, radii, count=9000, seed=0, infall=0.0, clumps=5, depth=0.9,
                 spread="area"):
        rng = np.random.default_rng(seed)
        if spread == "log":
            # Evenly in log radius. Where each cell's brightness is measured and
            # then imposed, parcels only have to cover the picture, and a wide
            # disk sampled by area puts almost all of them far off its edges.
            weights = 1.0 / radii
        else:
            # Weight by radius so the disk is evenly covered by area, not ring.
            weights = radii.astype(float)
        weights = weights / weights.sum()
        self.ring = rng.choice(len(radii), size=count, p=weights)
        self.angle0 = rng.random(count) * 2 * np.pi
        self.jitter = rng.random(count)          # keeps parcels off the ring lines
        self.luck = rng.random(count)            # whether it shows, fixed for life
        self.infall = infall
        self.count = count

        # Gas spread evenly around the disk looks the same however far it has
        # turned, so a perfectly smooth disk shows no rotation at all. This is a
        # tracer, not a prediction: the model says nothing about clumping, and
        # the pattern is here only so the motion can be seen. It is attached to
        # the material by the parcel's starting angle, so it orbits with the gas
        # and the differential rates wind it into arms - that shearing is real,
        # even though the pattern being sheared was put there by hand.
        self.brightness = 1.0
        if clumps:
            self.brightness = 1.0 + depth * np.cos(clumps * self.angle0)

    def at(self, phase, radii, turns, rates=None):
        """Where every parcel is.

        With `rates` given, `phase` is elapsed seconds and the parcels turn at
        their true rate, which never repeats. Without it, `phase` runs 0 to 1
        through a loop and the rounded whole-turn counts bring everything back.
        """
        ring = self.ring
        if self.infall:
            # Drift inwards and re-enter at the outer edge.
            slide = self.infall * phase * len(radii)
            ring = (self.ring - slide).astype(int) % len(radii)

        r = radii[ring]
        # Sit between rings rather than on them, or the disk looks like a target.
        step = radii[1] - radii[0] if len(radii) > 1 else 0.0
        r = r + (self.jitter - 0.5) * step

        if rates is None:
            angle = self.angle0 + 2 * np.pi * turns[ring] * phase
        else:
            angle = self.angle0 + rates[ring] * phase
        return ring, r, angle % (2 * np.pi)


def _kernel(spread):
    """Offsets and weights for spreading a point over nearby cells."""
    if spread <= 0:
        return np.array([0]), np.array([0]), np.array([1.0])
    k = int(np.ceil(spread))
    sigma = spread / 2.0
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
    distance = np.hypot(dx, dy)
    # Round, not square: a box of cells with a gentle falloff leaves the
    # corners bright enough to read as a square rather than a glint.
    keep = distance <= spread
    weight = np.exp(-(distance[keep] ** 2) / (2 * sigma ** 2))
    return dy[keep], dx[keep], weight


def _splat(grid, rows, cols, values, spread):
    """Add light at a position, optionally spread over neighbouring cells.

    np.add.at was nearly half the cost of a frame: it is unbuffered and handles
    one element at a time. Flattening the target index and accumulating with a
    single bincount does the same sum in one vectorised pass, with every kernel
    offset folded into the same call rather than one call each.
    """
    height, width = grid.shape
    dy, dx, weight = _kernel(spread)
    r = rows[None, :] + dy[:, None]
    c = cols[None, :] + dx[:, None]
    v = values[None, :] * weight[:, None]
    ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
    index = (r[ok] * width + c[ok]).astype(np.int64)
    grid += np.bincount(index, weights=v[ok], minlength=height * width).reshape(height, width)


def frame(mapping, parcels, phase, width, height, extent, turns, orders=(0, 1),
          hotspots=None, hot_gain=10.0, hot_spread=1.3, gas_spread=1.4,
          rates=None):
    """One frame: the surface brightness seen in each direction.

    `hotspots` is a second, much smaller parcel set drawn bright and spread over
    a few cells. Smooth gas cannot show rotation: spread evenly it looks the
    same at every phase, and a clumped pattern shears and averages away, because
    each screen position stacks light from many radii turning at different
    rates. A few discrete bright spots survive that averaging, so they are what
    makes the motion legible. Real disks do flare in spots, but their placement
    here is arbitrary - a marker on the gas, not a prediction of where gas is
    bright.

    Gas is averaged within a cell rather than summed. Parcels are a sampling of
    a continuous disk, so summing them measures how many samples happened to
    land in a cell, which is projection density and not brightness. That is not
    a small error: the near side of the disk compresses an enormous amount of
    disk area into the few cells in front of the hole, so summing fills the
    shadow with light and drowns the Doppler beaming. Hotspots stay additive,
    since they are meant to be an excess on top of the gas.
    """
    radii, angles = mapping["radii"], mapping["angles"]
    bh = mapping["bh"]
    if isinstance(extent, (tuple, list)):
        extent_x, extent_y = fit_extent(extent[0], width, height, reach_y=extent[1])
    else:
        extent_x, extent_y = fit_extent(extent, width, height)

    gas = np.zeros((height, width))
    gas_n = np.zeros((height, width))
    hot = np.zeros((height, width))
    z_sum = np.zeros((height, width))
    z_weight = np.zeros((height, width))

    sets = [(parcels, 1.0, 0.0, False)]
    if hotspots is not None:
        sets.append((hotspots, hot_gain, hot_spread, True))

    for group, gain, spread, is_hot in sets:
        ring, r, angle = group.at(phase, radii, turns, rates)

        for order in orders:
            if order not in mapping["tables"]:
                continue
            b_table, z_table = mapping["tables"][order]

            # Look the parcel up in the map: nearest ring, interpolated in angle.
            col = angle / (2 * np.pi) * (len(angles) - 1)
            lo = np.floor(col).astype(int) % len(angles)
            hi = (lo + 1) % len(angles)
            t = col - np.floor(col)

            b = (1 - t) * b_table[ring, lo] + t * b_table[ring, hi]
            z = (1 - t) * z_table[ring, lo] + t * z_table[ring, hi]
            good = np.isfinite(b) & np.isfinite(z)
            if not good.any():
                continue

            flux = bhmath.calc_flux_observed(r[good], bh.acc, bh.mass, z[good]) * gain
            weights = group.brightness
            if np.ndim(weights):
                flux = flux * weights[good]
            # Ghost light has already been round the back and is much the
            # fainter of the two; the map does not carry that, so weight it here.
            if order == 1:
                flux = flux * 0.45

            x = b[good] * np.sin(angle[good])
            y = -b[good] * np.cos(angle[good])

            col_i = ((x / extent_x + 1) / 2 * (width - 1)).astype(int)
            row_i = ((-y / extent_y + 1) / 2 * (height - 1)).astype(int)
            inside = (col_i >= 0) & (col_i < width) & (row_i >= 0) & (row_i < height)
            inside &= np.isfinite(flux)
            if not inside.any():
                continue

            rows, columns, values = row_i[inside], col_i[inside], flux[inside]
            if is_hot:
                _splat(hot, rows, columns, values, spread)
            else:
                # Spread the gas over a cell or so, and spread what it is
                # divided by in exactly the same way, so this stays a weighted
                # mean. Without it, any cell that happened to catch no parcel
                # reads as a hole in the disk.
                _splat(gas, rows, columns, values, gas_spread)
                _splat(gas_n, rows, columns, np.ones_like(values), gas_spread)
            _splat(z_sum, rows, columns, z[good][inside] * values, 0)
            _splat(z_weight, rows, columns, values, 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        brightness = np.where(gas_n > 0, gas / np.where(gas_n > 0, gas_n, 1), 0.0) + hot
        z_mean = np.where(z_weight > 0, z_sum / np.where(z_weight > 0, z_weight, 1), np.nan)

    lit = brightness > 0
    return {
        "flux": np.where(lit, brightness, np.nan),
        "z": z_mean,
        "radius": np.full((height, width), np.nan),
        "order": np.zeros((height, width)),
        "mask": lit,
    }


# A half-block subpixel in kitty and Ghostty is 10 wide by 11 tall, so it is
# very slightly taller than it is wide. Everything else assumes square samples.
CELL_ASPECT = 10.0 / 11.0


def fit_extent(extent, width, height, cell_aspect=CELL_ASPECT, reach_y=None):
    """Physical half-width and half-height that keep the picture in proportion.

    A terminal window is whatever shape somebody dragged it to, and mapping a
    fixed region onto it regardless stretches the disk to fill, which is why a
    wide short window squashed it.

    Fitting a square region instead keeps the shape but wastes most of a wide
    window, because an inclined disk is far wider than it is tall. So the disk's
    own bounding box is what gets fitted: the scale is whichever axis runs out
    first, and the other simply shows more empty sky.
    """
    reach_y = extent if reach_y is None else max(reach_y, 1e-6)
    on_screen_w = width * cell_aspect
    on_screen_h = height

    # Scale so the box fits both ways, then let the roomier axis show more sky.
    extent_y = max(reach_y, extent * on_screen_h / on_screen_w)
    extent_x = extent_y * on_screen_w / on_screen_h
    return extent_x, extent_y


def dots(mapping, parcels, phase, width, height, extent, rates, orders=(0, 1),
         gamma=0.85, smooth=1.4, projector=None):
    """Light individual dots from the gas, for the 1979 look.

    The dots are the material, so they orbit and the disk visibly turns without
    any tracer pattern. Two things have to be right for that to look like a
    brightness map rather than a map of where the samples happened to fall.

    Where a dot sits. Parcels in one ring all share that ring's curve on screen,
    so however their radius is jittered they line up into streaks. Each parcel's
    position is interpolated between its ring and the next by its jitter, which
    spreads the dots continuously across the disk.

    How many dots a cell gets. The near side of the disk packs a great many
    parcels into a few cells, so keeping parcels in proportion to their
    brightness still over-fills those cells - the same mistake as summing light.
    Instead the brightness per cell is measured first, as a mean, and each cell
    is then lit with that probability however many parcels it holds: each parcel
    is kept with 1 - (1 - target) ** (1 / count). A parcel's chance is fixed for
    its life, so dots do not flicker from frame to frame.

    Returns (lit, brightness): which dots to light, and the 0..1 brightness
    of each dot's cell, which a palette can colour.
    """
    radii, angles = mapping["radii"], mapping["angles"]
    bh = mapping["bh"]
    if isinstance(extent, (tuple, list)):
        extent_x, extent_y = fit_extent(extent[0], width, height, reach_y=extent[1])
    else:
        extent_x, extent_y = fit_extent(extent, width, height)

    if projector is not None:
        # The compiled pass: same samples, same cells, one fused loop.
        idx, flux, luck = projector(parcels, phase, rates, extent_x, extent_y, width, height)
        return _light(idx, flux, luck, width, height, gamma, smooth)

    ring, r, angle = parcels.at(phase, radii, None, rates)
    upper = np.minimum(ring + 1, len(radii) - 1)
    between = parcels.jitter
    col = angle / (2 * np.pi) * (len(angles) - 1)
    lo = np.floor(col).astype(int) % len(angles)
    hi = (lo + 1) % len(angles)
    t = col - np.floor(col)
    r_between = (1 - between) * radii[ring] + between * radii[upper]

    idx_all, flux_all, luck_all = [], [], []
    for order in orders:
        if order not in mapping["tables"]:
            continue
        b_table, z_table = mapping["tables"][order]

        def lookup(table):
            here = (1 - t) * table[ring, lo] + t * table[ring, hi]
            there = (1 - t) * table[upper, lo] + t * table[upper, hi]
            return (1 - between) * here + between * there

        b, z = lookup(b_table), lookup(z_table)
        good = np.isfinite(b) & np.isfinite(z)
        if not good.any():
            continue
        flux = bhmath.calc_flux_observed(r_between[good], bh.acc, bh.mass, z[good])
        if order == 1:
            flux = flux * 0.45
        x = b[good] * np.sin(angle[good])
        y = -b[good] * np.cos(angle[good])
        ci = ((x / extent_x + 1) / 2 * (width - 1)).astype(np.int64)
        ri = ((-y / extent_y + 1) / 2 * (height - 1)).astype(np.int64)
        inside = (ci >= 0) & (ci < width) & (ri >= 0) & (ri < height) & np.isfinite(flux)
        idx_all.append(ri[inside] * width + ci[inside])
        flux_all.append(flux[inside])
        luck_all.append(parcels.luck[good][inside])

    if not idx_all:
        return np.zeros((height, width), dtype=bool), np.zeros((height, width))
    return _light(np.concatenate(idx_all), np.concatenate(flux_all),
                  np.concatenate(luck_all), width, height, gamma, smooth)


def _light(idx, flux, luck, width, height, gamma, smooth):
    """From every sample's cell and flux, decide which dots to light."""
    lit = np.zeros(height * width, dtype=bool)
    if idx.size == 0:
        return lit.reshape(height, width), np.zeros((height, width))

    count = np.bincount(idx, minlength=height * width)
    total = np.bincount(idx, weights=flux, minlength=height * width)
    mean = (total / np.maximum(count, 1)).reshape(height, width)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter

        # Smooth the brightness, then put back the zeros: blurring into the
        # shadow would scatter dots across the one place that must stay black.
        covered = gaussian_filter((count > 0).reshape(height, width).astype(float), smooth)
        mean = gaussian_filter(mean, smooth) / np.maximum(covered, 1e-6)
        mean[count.reshape(height, width) == 0] = 0.0
    mean = mean.ravel()

    positive = mean[mean > 0]
    scale = np.percentile(positive, 99.3) if positive.size else 1.0
    target = np.clip(mean / max(scale, 1e-30), 0.0, 1.0) ** gamma

    n = np.maximum(count[idx], 1)
    chance = 1.0 - (1.0 - target[idx]) ** (1.0 / n)
    lit[idx[luck < chance]] = True
    return lit.reshape(height, width), target.reshape(height, width)


class DotField:
    """The 1979 look, split into what changes each frame and what does not.

    Where a parcel is changes every frame. How likely a screen cell is to show a
    dot does not: a given cell always sees gas at the same place on the disk,
    with the same Doppler shift, so its brightness is a fixed property of the
    map and the window. Measuring that once, averaged over a spread of moments,
    leaves each frame nothing to do but move the parcels and look them up - and
    because it is averaged rather than re-measured, the brightness scale no
    longer wobbles slightly from frame to frame.

    Parcels are kept with the same rule as dots(): each carries a lifetime
    random number and lights its cell when that is under the cell's chance,
    1 - (1 - target) ** (1 / n), n being how many samples usually land there.
    """

    def __init__(self, mapping, parcels, width, height, extent, rates, orders=(0, 1),
                 gamma=0.6, smooth=1.4, projector=None, moments=12):
        from scipy.ndimage import gaussian_filter

        self.mapping, self.parcels = mapping, parcels
        self.width, self.height = width, height
        self.orders, self.projector = orders, projector
        if isinstance(extent, (tuple, list)):
            self.ext_x, self.ext_y = fit_extent(extent[0], width, height, reach_y=extent[1])
        else:
            self.ext_x, self.ext_y = fit_extent(extent, width, height)
        cells_n = width * height

        # Accumulate per order: the two images sit on top of each other, and a
        # sample from one should be judged against its own image's density.
        total = np.zeros((2, cells_n))
        count = np.zeros((2, cells_n))
        scales = []
        span = 2 * np.pi / max(float(np.max(rates)), 1e-9)
        for k in range(moments):
            t = span * k / moments
            moment_total = np.zeros(cells_n)
            moment_count = np.zeros(cells_n)
            for o, (idx, flux) in self._samples(t, rates):
                f = np.bincount(idx, weights=flux, minlength=cells_n)
                c = np.bincount(idx, minlength=cells_n)
                total[o] += f
                count[o] += c
                moment_total += f
                moment_count += c
            # The brightness scale is set the way a single frame sets it, from
            # that frame's own peaks. Taking it from the averaged field instead
            # smooths the peaks away, lowers the scale, and lights noticeably
            # more of the dim outer disk than the look this matches.
            scales.append(self._scale(moment_total, moment_count, smooth))
        count /= moments
        total /= moments

        # Brightness per cell, as dots() measures it: a mean over whatever lands
        # there, both images together, smoothed, the shadow kept black.
        both_n = count.sum(axis=0)
        mean = (total.sum(axis=0) / np.maximum(both_n, 1e-9)).reshape(height, width)
        if smooth > 0:
            covered = gaussian_filter((both_n > 0).reshape(height, width).astype(float), smooth)
            mean = gaussian_filter(mean, smooth) / np.maximum(covered, 1e-6)
            mean[both_n.reshape(height, width) == 0] = 0.0
        mean = mean.ravel()
        scale = float(np.mean(scales)) if scales else 1.0
        target = np.clip(mean / max(scale, 1e-30), 0.0, 1.0) ** gamma
        self.target = target.reshape(height, width)

        # A cell is lit if any of its samples draws under the chance, so each
        # sample's chance is set from how many usually arrive - but never from
        # fewer than one. Where the disk is thin a cell is often empty, and
        # dividing by a fractional count would top each lone sample up to make
        # good the frames it is absent from. Measured per frame, as dots() does,
        # an empty cell simply stays dark, so the faint outer disk looks faint;
        # topping it up lit a third more dots there than the look this matches.
        n = np.maximum(both_n, 1.0)
        self.chance = np.empty((2, cells_n))
        self.chance[:] = 1.0 - (1.0 - target[None, :]) ** (1.0 / n[None, :])
        self.lit = np.zeros(cells_n, dtype=np.bool_)

    def _scale(self, total, count, smooth):
        """The 99.3rd percentile of one moment's smoothed brightness."""
        from scipy.ndimage import gaussian_filter

        h, w = self.height, self.width
        mean = (total / np.maximum(count, 1)).reshape(h, w)
        if smooth > 0:
            covered = gaussian_filter((count > 0).reshape(h, w).astype(float), smooth)
            mean = gaussian_filter(mean, smooth) / np.maximum(covered, 1e-6)
            mean[count.reshape(h, w) == 0] = 0.0
        positive = mean[mean > 0]
        return np.percentile(positive, 99.3) if positive.size else 1.0

    def _samples(self, t, rates):
        """Cell and flux of every sample at time t, one pair per image order."""
        if self.projector is not None:
            # One projection covers both images; split them apart afterwards.
            self.projector(self.parcels, t, rates, self.ext_x, self.ext_y,
                           self.width, self.height)
            out = []
            for order in self.orders:
                where = self.projector._idx[:, order]
                ok = where >= 0
                out.append((order, (where[ok], self.projector._flux[:, order][ok])))
            return out
        return [(order, self._samples_numpy(t, rates, order)) for order in self.orders]

    def _samples_numpy(self, t, rates, order):
        mapping = self.mapping
        radii, angles, bh = mapping["radii"], mapping["angles"], mapping["bh"]
        p = self.parcels
        ring, _, angle = p.at(t, radii, None, rates)
        upper = np.minimum(ring + 1, len(radii) - 1)
        col = angle / (2 * np.pi) * (len(angles) - 1)
        lo = np.floor(col).astype(int) % len(angles)
        hi = (lo + 1) % len(angles)
        f = col - np.floor(col)
        between = p.jitter
        b_t, z_t = mapping["tables"][order]
        look = lambda tb: ((1 - between) * ((1 - f) * tb[ring, lo] + f * tb[ring, hi])
                           + between * ((1 - f) * tb[upper, lo] + f * tb[upper, hi]))
        b, z = look(b_t), look(z_t)
        r_b = (1 - between) * radii[ring] + between * radii[upper]
        good = np.isfinite(b) & np.isfinite(z)
        flux = bhmath.calc_flux_observed(r_b[good], bh.acc, bh.mass, z[good])
        if order == 1:
            flux = flux * 0.45
        ci = ((b[good] * np.sin(angle[good]) / self.ext_x + 1) / 2 * (self.width - 1)).astype(np.int64)
        ri = ((b[good] * np.cos(angle[good]) / self.ext_y + 1) / 2 * (self.height - 1)).astype(np.int64)
        inside = ((ci >= 0) & (ci < self.width) & (ri >= 0) & (ri < self.height)
                  & np.isfinite(flux))
        return ri[inside] * self.width + ci[inside], flux[inside]

    def frame(self, t, rates):
        """Which dots are lit at time t."""
        if self.projector is not None:
            self.projector.place(self.parcels, t, rates, self.ext_x, self.ext_y,
                                 self.width, self.height, self.chance, self.lit)
            return self.lit.reshape(self.height, self.width)
        self.lit[:] = False
        luck = self.parcels.luck
        for o in self.orders:
            mapping = self.mapping
            radii, angles = mapping["radii"], mapping["angles"]
            p = self.parcels
            ring, _, angle = p.at(t, radii, None, rates)
            upper = np.minimum(ring + 1, len(radii) - 1)
            col = angle / (2 * np.pi) * (len(angles) - 1)
            lo = np.floor(col).astype(int) % len(angles)
            hi = (lo + 1) % len(angles)
            f = col - np.floor(col)
            tb = mapping["tables"][o][0]
            b = ((1 - p.jitter) * ((1 - f) * tb[ring, lo] + f * tb[ring, hi])
                 + p.jitter * ((1 - f) * tb[upper, lo] + f * tb[upper, hi]))
            good = np.isfinite(b)
            ci = ((b[good] * np.sin(angle[good]) / self.ext_x + 1) / 2 * (self.width - 1)).astype(np.int64)
            ri = ((b[good] * np.cos(angle[good]) / self.ext_y + 1) / 2 * (self.height - 1)).astype(np.int64)
            inside = (ci >= 0) & (ci < self.width) & (ri >= 0) & (ri < self.height)
            k = ri[inside] * self.width + ci[inside]
            on = luck[good][inside] < self.chance[o, k]
            self.lit[k[on]] = True
        return self.lit.reshape(self.height, self.width)


class Isolines:
    """Isoradials traced on the dot grid, for drawing over the 1979 dots.

    An isoradial is where light from one ring of the disk lands on the screen.
    The lensing map already holds every ring's curve, so nothing is solved
    again: the chosen rings are read out of it, sampled densely enough to leave
    no gaps between dots, and kept as dot positions.

    At a fixed inclination the curves themselves never change, so what can move
    is along them. flowing=True draws each ring as dashes that travel at that
    ring's own orbital rate, so the inner rings visibly run faster than the
    outer ones - the same motion as the gas, carried by the lines.

    A curve is broken wherever the map has no light, rather than joined across
    the gap.
    """

    def __init__(self, mapping, width, height, ext_x, ext_y,
                 direct=(6, 10, 15, 20), ghost=(6, 20, 50, 100), samples=2400):
        radii, angles, tables = mapping["radii"], mapping["angles"], mapping["tables"]
        na = len(angles)
        dense = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
        nearest = np.rint(dense / (2 * np.pi) * (na - 1)).astype(int) % na
        self.width, self.height = width, height
        self.lines = []
        for order, chosen in ((0, direct), (1, ghost)):
            if order not in tables:
                continue
            for radius in chosen:
                ring = int(np.argmin(np.abs(radii - radius)))
                b = tables[order][0][ring]
                good = np.isfinite(b)
                if good.sum() < 4:
                    continue
                b_dense = np.interp(dense, angles[good], b[good], period=2 * np.pi)
                valid = good[nearest]
                ci = ((b_dense * np.sin(dense) / ext_x + 1) / 2 * (width - 1)).astype(np.int64)
                ri = ((b_dense * np.cos(dense) / ext_y + 1) / 2 * (height - 1)).astype(np.int64)
                inside = valid & (ci >= 0) & (ci < width) & (ri >= 0) & (ri < height)
                self.lines.append((ring, dense[inside], ri[inside] * width + ci[inside]))
        self.mask = np.zeros(width * height, dtype=bool)

    def frame(self, t, rates, flowing=False, dashes=14, duty=0.55):
        """Which dots the lines light at time t."""
        self.mask[:] = False
        for ring, alpha, cells_at in self.lines:
            if flowing:
                # A dash pattern fixed to the gas: the pattern at angle alpha
                # now is where the gas that was at alpha - rate * t has come to.
                phase = ((alpha - rates[ring] * t) * dashes / (2 * np.pi)) % 1.0
                self.mask[cells_at[phase < duty]] = True
            else:
                self.mask[cells_at] = True
        return self.mask.reshape(self.height, self.width)


def reach(mapping, orders=(0, 1), max_radius=None):
    """How far the image extends across and down, so every frame shares a scale.

    Returned as (x, y): an inclined disk is much wider than it is tall, and
    knowing both is what lets a wide window be filled rather than padded.
    """
    angles = mapping["angles"]
    rows = (slice(None) if max_radius is None
            else mapping["radii"] <= max_radius)
    rx = ry = 1.0
    for order in orders:
        if order not in mapping["tables"]:
            continue
        b = mapping["tables"][order][0][rows]
        if not np.isfinite(b).any():
            continue
        xs = np.abs(b * np.sin(angles)[None, :])
        ys = np.abs(b * np.cos(angles)[None, :])
        rx = max(rx, float(np.nanmax(xs)))
        ry = max(ry, float(np.nanmax(ys)))
    return rx * 1.04, ry * 1.06
