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


def lensing_map(settings, n_rings=48, n_angles=180, orders=(0, 1)):
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
    bh.calc_isoradials(
        direct_r=radii if 0 in orders else [],
        ghost_r=radii if 1 in orders else [],
    )

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


class Parcels:
    """Gas, spread over the disk, carried around the map."""

    def __init__(self, radii, count=9000, seed=0, infall=0.0, clumps=3, depth=0.75):
        rng = np.random.default_rng(seed)
        # Weight by radius so the disk is evenly covered by area, not by ring.
        weights = radii / radii.sum()
        self.ring = rng.choice(len(radii), size=count, p=weights)
        self.angle0 = rng.random(count) * 2 * np.pi
        self.jitter = rng.random(count)          # keeps parcels off the ring lines
        self.infall = infall
        self.count = count

        # Gas spread evenly around the disk looks the same however far it has
        # turned, so a perfectly smooth disk shows no rotation at all. This is a
        # tracer, not a prediction: the model says nothing about clumping, and
        # the pattern is here only so the motion can be seen. It is attached to
        # the material by the parcel's starting angle, so it orbits with the gas
        # and the differential rates shear it, as they would any real pattern.
        self.brightness = 1.0
        if clumps:
            self.brightness = 1.0 + depth * np.cos(clumps * self.angle0)

    def at(self, phase, radii, turns):
        """Where every parcel is, a fraction `phase` through the loop."""
        ring = self.ring
        if self.infall:
            # Drift inwards and re-enter at the outer edge. Parcels swap places
            # over one loop, so the ensemble repeats even though they do not.
            slide = self.infall * phase * len(radii)
            ring = (self.ring - slide).astype(int) % len(radii)

        r = radii[ring]
        # Sit between rings rather than on them, or the disk looks like a target.
        step = radii[1] - radii[0] if len(radii) > 1 else 0.0
        r = r + (self.jitter - 0.5) * step

        angle = (self.angle0 + 2 * np.pi * turns[ring] * phase) % (2 * np.pi)
        return ring, r, angle


def _splat(grid, rows, cols, values, spread):
    """Add light at a position, optionally spread over neighbouring cells."""
    if spread <= 0:
        np.add.at(grid, (rows, cols), values)
        return
    k = int(spread)
    height, width = grid.shape
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            falloff = np.exp(-(dx * dx + dy * dy) / (2 * (spread / 1.5) ** 2))
            r, c = rows + dy, cols + dx
            ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            np.add.at(grid, (r[ok], c[ok]), values[ok] * falloff)


def frame(mapping, parcels, phase, width, height, extent, turns, orders=(0, 1),
          hotspots=None, hot_gain=55.0, hot_spread=1.3):
    """One frame: accumulate parcel light into a grid.

    `hotspots` is a second, much smaller parcel set drawn bright and spread over
    a few cells. Smooth gas cannot show rotation: spread evenly it looks the
    same at every phase, and a clumped pattern shears and averages away, because
    each screen position stacks light from many radii turning at different
    rates. A few discrete bright spots survive that averaging, so they are what
    makes the motion legible. Real disks do flare in spots, but their placement
    here is arbitrary - a marker on the gas, not a prediction of where gas is
    bright.
    """
    radii, angles = mapping["radii"], mapping["angles"]
    bh = mapping["bh"]

    flux_grid = np.zeros((height, width))
    z_grid = np.zeros((height, width))
    weight = np.zeros((height, width))

    sets = [(parcels, 1.0, 0.0)]
    if hotspots is not None:
        sets.append((hotspots, hot_gain, hot_spread))

    for group, gain, spread in sets:
      ring, r, angle = group.at(phase, radii, turns)
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
        # Ghost light has already been round the back and is much the fainter
        # of the two; the map does not carry that, so weight it here.
        if order == 1:
            flux = flux * 0.45

        x = b[good] * np.sin(angle[good])
        y = -b[good] * np.cos(angle[good])

        col_i = ((x / extent + 1) / 2 * (width - 1)).astype(int)
        row_i = ((-y / extent + 1) / 2 * (height - 1)).astype(int)
        inside = (col_i >= 0) & (col_i < width) & (row_i >= 0) & (row_i < height)
        inside &= np.isfinite(flux)
        if not inside.any():
            continue

        _splat(flux_grid, row_i[inside], col_i[inside], flux[inside], spread)
        _splat(z_grid, row_i[inside], col_i[inside], z[inside] * flux[inside], spread)
        _splat(weight, row_i[inside], col_i[inside], flux[inside], spread)

    lit = weight > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        z_mean = np.where(lit, z_grid / np.where(lit, weight, 1), np.nan)

    return {
        "flux": np.where(lit, flux_grid, np.nan),
        "z": z_mean,
        "radius": np.full((height, width), np.nan),
        "order": np.zeros((height, width)),
        "mask": lit,
    }


def reach(mapping, orders=(0, 1)):
    """How far the image extends, so every frame shares one scale."""
    biggest = 1.0
    for order in orders:
        if order in mapping["tables"]:
            b = mapping["tables"][order][0]
            if np.isfinite(b).any():
                biggest = max(biggest, float(np.nanmax(b)))
    return biggest * 1.02
