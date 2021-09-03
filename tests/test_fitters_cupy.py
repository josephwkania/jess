#!/usr/bin/env python3
"""
Test for jess.fitters
"""

import pytest

cp = pytest.importorskip("cupy")

# pylint: disable=C0413
from jess.fitters_cupy import (  # isort:skip # noqa: E402
    poly_fitter,
)


def test_poly_fitter():
    """
    Make a polynomial with spikes and fit to it
    """
    # need cp.asarray on gpu version
    data = cp.polyval(cp.asarray([-0.01, 7, 100]), cp.linspace(0, 1023, 1024))
    data_dirty = data.copy()
    data_dirty[65] += 65
    data_dirty[20] += 20
    fit = poly_fitter(data_dirty)
    assert cp.allclose(data, fit)
