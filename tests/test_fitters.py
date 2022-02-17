#!/usr/bin/env python3
"""
Test for jess.fitters
"""

import numpy as np
import pytest

from jess.fitters import (
    arpls_fitter,
    bspline_fit,
    bspline_fitter,
    cheb_fitter,
    get_fitter,
    median_fitter,
    poly_fitter,
)


def test_get_fitter():
    """
    Make sure get_fitter returns all the fitters
    """
    assert arpls_fitter == get_fitter("arpls_fitter")
    assert bspline_fitter == get_fitter("bspline_fitter")
    assert cheb_fitter == get_fitter("cheb_fitter")
    assert median_fitter == get_fitter("median_fitter")
    assert poly_fitter == get_fitter("poly_fitter")
    with pytest.raises(ValueError):
        get_fitter("joe")


def test_arpls_fitter():
    """
    Fit to some data with spikes
    ignore the spikes

    There can be some ringing on the
    ends, so just clip those parts.
    Can also increase lam, but that makes it slow
    """
    data = np.polyval([-0.01, 7, 100], np.linspace(0, 1023, 1024))
    data_dirty = data.copy()
    data_dirty[30] += 30
    data_dirty[60] += 60
    clip = 17
    fit = arpls_fitter(data_dirty, lam=20)
    np.testing.assert_allclose(data[clip:-clip], fit[clip:-clip], rtol=0.5)


def test_bspline_fit():
    """
    Fit to some data with spikes,
    ignore the spikes
    """
    # random = np.random.normal(size=1024)
    clean = 20 * np.sin(np.linspace(0, 2 * np.pi, 1024))
    contaminated = clean.copy()
    contaminated[30] += 30
    contaminated[60] += 60
    np.testing.assert_allclose(clean, bspline_fit(contaminated), atol=0.15)


def test_bspline_fitter():
    """
    This should be the same as bspine_fit
    Fit to some data with spikes,
    ignore the spikes
    """
    # random = np.random.normal(size=1024)
    clean = 20 * np.sin(np.linspace(0, 2 * np.pi, 1024))
    contaminated = clean.copy()
    contaminated[30] += 30
    contaminated[60] += 60
    # need to use atol instead of rtol,
    # problem with relative and zero crossing?
    np.testing.assert_allclose(clean, bspline_fitter(contaminated), atol=0.15)


def test_cheb_fitter():
    """
    Make a Chebyshev polynomial with spikes and fit to it
    """
    data = np.polynomial.chebyshev.chebval(np.linspace(0, 1023, 1024), [-0.01, 7, 100])
    data_dirty = data.copy()
    data_dirty[65] += 65
    data_dirty[20] += 20
    fit = cheb_fitter(data_dirty)
    np.testing.assert_allclose(data, fit)


def test_median_fitter():
    """
    Make a polynomial with spikes and fit to it
    Have to clip the end because the is not info past them
    and the fit goes off
    """
    data = np.polyval([0.5, 100], np.linspace(0, 1023, 1024))
    data_dirty = data.copy()
    data_dirty[65] += 65
    data_dirty[512] += 120
    fit = median_fitter(data_dirty)
    clip = 3
    np.testing.assert_allclose(data[clip:-clip], fit[clip:-clip], atol=1)
    with pytest.raises(ValueError):
        median_fitter(data_dirty, interations=0)


def test_poly_fitter():
    """
    Make a polynomial with spikes and fit to it
    """
    data = np.polyval([-0.01, 7, 100], np.linspace(0, 1023, 1024))
    data_dirty = data.copy()
    data_dirty[65] += 65
    data_dirty[20] += 20
    fit = poly_fitter(data_dirty)
    np.testing.assert_allclose(data, fit)
