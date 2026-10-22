"""The sweep's fixed point must satisfy branch power balance, not just settle."""

import numpy as np
import pytest

from powerzoo.envs.grid.cal_pf_dist import build_radial_topology, run_bfs_power_flow


def _two_bus():
    return build_radial_topology(2, np.array([0]), np.array([1]),
                                 np.array([0.1]), np.array([0.0]))


def test_two_bus_matches_analytic_high_voltage_solution():
    r = run_bfs_power_flow(_two_bus(), np.array([0.0, 2.0]), np.zeros(2))
    p_send = float(r['p_branch'][0])
    assert r['converged']
    np.testing.assert_allclose(r['v_mag'][1], (1 + np.sqrt(1 - 0.8)) / 2, atol=1e-5)
    assert abs(p_send - 2.0 - 0.1 * p_send ** 2) < 1e-5


@pytest.mark.parametrize("load", [2.6, 5.6, 20.0])
def test_two_bus_beyond_collapse_never_converges(load):
    r = run_bfs_power_flow(_two_bus(), np.array([0.0, load]), np.zeros(2))
    assert not r['converged']
    assert np.isfinite(r['v_mag']).all()
