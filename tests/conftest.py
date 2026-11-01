import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vfmchoke import network, pvt, units as u, well  # noqa: E402


@pytest.fixture
def fluid():
    return pvt.Fluid(api=35.0, gas_sg=0.75, gor=150.0, wc=0.2, salinity=0.05)


def make_wells(g=2, w=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (g, w)
    fl = pvt.Fluid(api=rng.uniform(28, 40, shape), gas_sg=rng.uniform(0.65, 0.85, shape),
                   gor=rng.uniform(60, 250, shape), wc=rng.uniform(0.0, 0.5, shape),
                   salinity=rng.uniform(0.02, 0.08, shape))
    return well.Wells(p_res=rng.uniform(250e5, 320e5, shape), t_res=u.c_to_k(rng.uniform(80, 110, shape)),
                      pi=rng.uniform(5, 15, shape) * well.PI_FIELD, depth=rng.uniform(1800, 2500, shape),
                      d_tub=np.full(shape, 0.1), d_choke=np.full(shape, 0.025), cd=rng.uniform(0.75, 0.85, shape),
                      trim_gamma=rng.uniform(0.9, 1.4, shape), t_sea=np.full(shape, u.c_to_k(4.0)),
                      q_t=np.full(shape, 0.002), fluid=fl)


@pytest.fixture
def wells():
    return make_wells()


@pytest.fixture
def line():
    return network.Line(length=np.array([5000.0, 8000.0]), d=np.array([0.2, 0.2]), depth=np.array([100.0, 120.0]),
                        p_sep=np.array([15e5, 15e5]))
