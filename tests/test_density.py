from luminet.black_hole import BlackHole
import numpy as np
from pathlib import Path
import pytest

def test_read_data():
    import luminet
    data_saturn = Path(luminet.__file__).parent / "density_profiles" / "saturn.npy"
    assert data_saturn.exists()
    data = np.load(data_saturn)
    assert data.shape == (821,)


def test_density_irs():
    import matplotlib.pyplot as plt
    from luminet.isoradial import Isoradial
    fig = plt.figure()
    ax = fig.gca()
    rs = [10, 20, 30]
    densities = [0.1, 0.5, 1]
    incl = 1.0
    for r, d in zip(rs, densities):
        ir = Isoradial(radius=r, incl=incl, bh_mass=1.0, order=0, density=d)
        ir.plot(ax=ax, color='k')
    plt.show()


@pytest.mark.parametrize("mass", [1.])
def test_density(mass, incl=1.2):

    import luminet
    data_saturn = Path(luminet.__file__).parent / "density_profiles" / "saturn.npy"
    density = np.load(data_saturn)

    bh = BlackHole(incl=incl, mass=mass, density=density)

    import matplotlib.pyplot as plt
    bh.plot(lw=0.2)
    assert len(bh.isoradials) == 821 * 2
    plt.show()
    return None


if __name__ == "__main__":
    test_density(1.)
