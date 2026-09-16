"""The observer plane as data rather than as a picture.

A rendered PNG has already thrown the physics away: one grey value per pixel,
and no way to ask what that light was. This module keeps what the renderer
actually knows at each point instead, so a display can choose later what to show
without recomputing anything:

    flux    how bright it appears
    z       redshift factor 1 + z, Doppler plus the climb out of the well
    radius  which ring of gas the light left from
    order   0 direct, 1 for light that went around the back

Sampling is scattered, because the solver works along isoradials rather than on
a grid. Points are placed on a grid afterwards by nearest neighbour with a
distance cutoff, which keeps the shadow and the outer edge genuinely empty
rather than interpolating across them.
"""

import numpy as np

from luminet import black_hole_math as bhmath
from luminet.black_hole import BlackHole

CHANNELS = ["flux", "z", "radius"]


def compute(settings, n_radii=140, n_angles=240):
    """Sample the observer plane. Returns one dict of flat arrays per image order."""
    bh = BlackHole(
        mass=float(settings.get("mass", 1.0)),
        incl=float(settings.get("incl", 1.4)),
        acc=float(settings.get("acc", 1.0)),
        outer_edge=float(settings.get("outer_edge", 40.0)),
        angular_resolution=n_angles,
        radial_resolution=n_radii,
    )
    radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, n_radii)
    bh.calc_isoradials(direct_r=radii, ghost_r=radii)

    orders = {}
    for ir in bh.isoradials:
        angle = np.asarray(ir.angles, dtype=float)
        b = np.asarray(ir.impact_parameters, dtype=float)
        zf = np.asarray(ir.redshift_factors, dtype=float)
        flux = bhmath.calc_flux_observed(ir.radius, bh.acc, bh.mass, zf)

        good = np.isfinite(b) & np.isfinite(zf) & np.isfinite(flux)
        if not good.any():
            continue

        # Matplotlib draws these polar, with angle zero at the south of the
        # figure, so match that here and the terminal picture agrees with the
        # rendered one.
        bucket = orders.setdefault(ir.order, {k: [] for k in ("x", "y", *CHANNELS)})
        bucket["x"].append(b[good] * np.sin(angle[good]))
        bucket["y"].append(-b[good] * np.cos(angle[good]))
        bucket["flux"].append(flux[good])
        bucket["z"].append(zf[good])
        bucket["radius"].append(np.full(good.sum(), float(ir.radius)))

    return {order: {k: np.concatenate(v) for k, v in bucket.items()}
            for order, bucket in orders.items()}


def extent_of(orders):
    """How far the image reaches, so the grid can be squared on it."""
    reach = 1.0
    for points in orders.values():
        if len(points["x"]):
            reach = max(reach, float(np.abs(points["x"]).max()),
                        float(np.abs(points["y"]).max()))
    return reach * 1.02


def rasterise(orders, width, height, extent=None, slack=1.6):
    """Put the scattered samples on a width x height grid.

    The grid always covers a square region, so the picture keeps its shape even
    where the samples per axis differ. `slack` is how far a grid point may reach
    for a sample, in multiples of the grid spacing; beyond that the cell is
    empty, which is what keeps the shadow dark.
    """
    from scipy.spatial import cKDTree

    from luminet.spin import fit_extent

    extent = extent or extent_of(orders)
    # Keep the picture in proportion whatever shape the grid is.
    if isinstance(extent, (tuple, list)):
        extent_x, extent_y = fit_extent(extent[0], width, height, reach_y=extent[1])
    else:
        extent_x, extent_y = fit_extent(extent, width, height)
    xs = np.linspace(-extent_x, extent_x, width)
    ys = np.linspace(extent_y, -extent_y, height)  # first row is the top
    gx, gy = np.meshgrid(xs, ys)
    targets = np.column_stack([gx.ravel(), gy.ravel()])

    # A cell may reach about as far as the coarser axis, or dense sampling on
    # one axis would punch holes along the other.
    spacing = max(2 * extent_x / max(width - 1, 1), 2 * extent_y / max(height - 1, 1))
    cutoff = spacing * slack

    out = {k: np.full((height, width), np.nan) for k in CHANNELS}
    out["order"] = np.full((height, width), np.nan)
    best_flux = np.full(height * width, -np.inf)

    for order in sorted(orders):
        points = orders[order]
        if not len(points["x"]):
            continue
        tree = cKDTree(np.column_stack([points["x"], points["y"]]))
        distance, index = tree.query(targets, distance_upper_bound=cutoff)
        hit = np.isfinite(distance)
        if not hit.any():
            continue

        # Where two orders land on the same cell, the brighter one is the one
        # you would actually see there.
        flux = points["flux"][index[hit]]
        wins = np.zeros_like(best_flux, dtype=bool)
        wins[hit] = flux > best_flux[hit]
        best_flux[wins] = points["flux"][index[wins]]

        for name in CHANNELS:
            out[name].reshape(-1)[wins] = points[name][index[wins]]
        out["order"].reshape(-1)[wins] = order

    out["mask"] = np.isfinite(out["flux"])
    return out


def normalise(grid, channel, lo=None, hi=None, percentile=97.0, gamma=1.0):
    """Scale one channel to 0..1, leaving empty cells as NaN.

    Flux spans orders of magnitude, and a handful of very bright cells set the
    maximum: with a bright spot in frame the whole disk sits near 1.7% of the
    scale and renders as black. Scaling to a high percentile instead and letting
    the brightest cells clip puts the disk back in the visible range, which is
    what a photograph of something this bright would do anyway.
    """
    values = grid[channel]
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.full_like(values, np.nan)

    if channel == "z":
        # Centre a diverging scale on z = 0, i.e. a redshift factor of 1.
        reach = max(abs(float(finite.min()) - 1.0), abs(float(finite.max()) - 1.0)) or 1.0
        lo, hi = 1.0 - reach, 1.0 + reach
    else:
        lo = float(finite.min()) if lo is None else lo
        if hi is None:
            hi = float(np.percentile(finite, percentile)) if percentile else float(finite.max())

    if hi == lo:
        return np.where(np.isfinite(values), 0.5, np.nan)
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return scaled ** gamma if gamma != 1.0 else scaled
