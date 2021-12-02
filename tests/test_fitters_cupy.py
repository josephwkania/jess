#!/usr/bin/env python3
"""
Test for jess.fitters
"""

import warnings

import pytest

cp = pytest.importorskip("cupy")

# pylint: disable=C0413
from jess.fitters_cupy import (  # isort:skip # noqa: E402
    arpls_fitter,
    poly_fitter,
)


def test_arpls_fitter():
    """
    Fit to some data with spikes
    ignore the spikes

    There can be some ringing on the
    ends, so just clip those parts.
    Seems to have a sweet spot for lam
    """
    data = cp.polyval(cp.asarray([-0.01, 7, 100]), cp.linspace(0, 1023, 1024))
    data_dirty = data.copy()
    data_dirty[30] += 30
    data_dirty[60] += 60
    clip = 17
    with warnings.catch_warnings(record=True):
        # continues to throw csr warning even when using csr format
        # ignore
        fit = arpls_fitter(data_dirty, lam=20)
    cp.testing.assert_allclose(data[clip:-clip], fit[clip:-clip], rtol=0.5)


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
