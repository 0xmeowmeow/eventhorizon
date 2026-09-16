from luminet.black_hole import BlackHole


def test_photons(n=1000):
    """
    Test if black hole can sample individual photons
    """

    bh = BlackHole(incl=1.3)
    photons, ghost_photons = bh.sample_photons(n)

    assert len(photons) == n
    assert len(ghost_photons) == n


def test_photons_are_independent_samples(n=64):
    """
    Test that sampled photons are actually distinct.

    Sampling is parallellized over a multiprocessing Pool. Workers forked from
    the parent inherit its global numpy RNG state, so a worker drawing from
    np.random directly would replay the very same numbers as every other worker
    and the sample would collapse onto a few repeated photons.
    """

    bh = BlackHole(incl=1.3)
    photons, ghost_photons = bh.sample_photons(n)

    for sample in (photons, ghost_photons):
        radii = {photon.radius for photon in sample}
        assert len(radii) == n, f"only {len(radii)} of {n} sampled photons are distinct"


def test_photon_sampling_is_reproducible(n=16):
    """
    Test that passing a seed makes sampling reproducible, and that omitting one
    does not.
    """

    bh = BlackHole(incl=1.3)

    seeded = [photon.radius for photon in bh.sample_photons(n, seed=42)[0]]
    seeded_again = [photon.radius for photon in bh.sample_photons(n, seed=42)[0]]
    unseeded = [photon.radius for photon in bh.sample_photons(n)[0]]

    assert seeded == seeded_again
    assert seeded != unseeded


def test_direct_and_ghost_photons_are_sampled_independently(n=16):
    """
    Test that the direct and ghost image draw from different random streams,
    rather than both replaying the first half of the seeded sequence.
    """

    bh = BlackHole(incl=1.3)
    photons, ghost_photons = bh.sample_photons(n, seed=42)

    assert [p.radius for p in photons] != [p.radius for p in ghost_photons]
