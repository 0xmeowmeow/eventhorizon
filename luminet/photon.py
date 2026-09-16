r"""Individual photons sampled from the accretion disk."""

import numpy as np

import luminet.black_hole_math as bhmath


class Photon:
    def __init__(self, radius, alpha, impact_parameter, z_factor=0.0, flux_o=0.0):
        self.radius = radius
        self.alpha = alpha
        self.impact_parameter = impact_parameter
        self.z_factor = z_factor
        """Redshift factor of the photon. This is usually calculated vectorized by the :class:`~luminet.black_hole.BlackHole` class"""
        self.flux_o = flux_o
        """Observed flux of the photon. This is usually calculated vectorized by the :class:`~luminet.black_hole.BlackHole` class"""

    def __repr__(self):
        return (
            f"Photon(radius={self.radius:.4f}, alpha={self.alpha:.4f}, "
            f"impact_parameter={self.impact_parameter:.4f}, "
            f"z_factor={self.z_factor:.4f}, flux_o={self.flux_o:.4e})"
        )


def sample_photon(min_r, max_r, incl, bh_mass, n, seed=None) -> Photon:
    r"""Sample a random photon from the accretion disk

    Each photon is a :class:`Photon` with the following properties:

    - ``radius``: radius of the photon on the accretion disk :math:`r`
    - ``alpha``: angle of the photon on the accretion disk :math:`\alpha`
    - ``impact_parameter``: impact parameter of the photon :math:`b`
    - ``z_factor``: redshift factor of the photon :math:`1+z`

    This function is used in :meth:`~luminet.black_hole.BlackHole.sample_photons` to sample
    photons on the accretion disk of a black hole in a parallellized manner.

    Attention:
        Photons are not sampled uniformly on the accretion disk, but biased towards the center.
        Black holes have more flux delta towards the center, and thus we need more precision there.
        This makes the triangulation with hollow mask in the center also very happy.

    Args:
        min_r: minimum radius of the accretion disk
        max_r: maximum radius of the accretion disk
        incl: inclination of the observer wrt the disk
        bh_mass: mass of the black hole
        n: order of the isoradial
        seed: seed for this photon's random number generator. Anything
            :func:`numpy.random.default_rng` accepts, typically a
            :class:`numpy.random.SeedSequence`. When this function is called in
            parallel, every call needs its own seed: see
            :meth:`~luminet.black_hole.BlackHole.sample_photons`. Defaults to
            :py:data:`None`, which draws fresh entropy from the OS.

    Returns:
        :class:`Photon`: A single photon sampled from the accretion disk.
    """
    rng = np.random.default_rng(seed)

    alpha = rng.random() * 2 * np.pi

    # Bias sampling towards circle center (even sampling would be sqrt(random))
    r = np.float64(min_r + (max_r - min_r) * rng.random())
    b = bhmath.solve_for_impact_parameter(r, incl, alpha, bh_mass, n)
    if b is np.nan:
        raise ValueError(
            f"b is nan for r={r}, alpha={alpha}, incl={incl}, M={bh_mass}, n={n}"
        )

    return Photon(radius=r, alpha=alpha, impact_parameter=b)
