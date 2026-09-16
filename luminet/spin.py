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
         gamma=0.85, smooth=1.4):
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

    Returns a boolean (height, width) array of lit dots.
    """
    radii, angles = mapping["radii"], mapping["angles"]
    bh = mapping["bh"]
    if isinstance(extent, (tuple, list)):
        extent_x, extent_y = fit_extent(extent[0], width, height, reach_y=extent[1])
    else:
        extent_x, extent_y = fit_extent(extent, width, height)

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

    lit = np.zeros(height * width, dtype=bool)
    if not idx_all:
        return lit.reshape(height, width)
    idx = np.concatenate(idx_all)
    flux = np.concatenate(flux_all)
    luck = np.concatenate(luck_all)

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
    return lit.reshape(height, width)


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
