#!/usr/bin/env python3
"""
Tests for median_abs_deviation
Stolen from scipy.stats.tests.test_stats.py
and subject to the scipy license
"""
import warnings

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_equal,
)

cp = pytest.importorskip("cupy")

# pylint: disable=C0413
from jess.scipy_cupy.stats import (  # isort:skip # noqa: E402
    median_abs_deviation,
    median_abs_deviation_med,
)


class TestMedianAbsDeviation:
    def setup_class(self):
        self.dat_nan = cp.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                np.nan,
            ]
        )
        self.dat = cp.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                28.95,
            ]
        )

    def test_median_abs_deviation(self):
        assert_almost_equal(median_abs_deviation(self.dat, axis=None).get(), 0.355)
        dat = self.dat.reshape(6, 4)
        mad = median_abs_deviation(dat, axis=0)
        mad_expected = cp.asarray([0.435, 0.5, 0.45, 0.4])
        assert_array_almost_equal(mad.get(), mad_expected.get())

    def test_mad_nan_omit(self):
        mad = median_abs_deviation(self.dat_nan, nan_policy="omit")
        assert_almost_equal(mad.get(), 0.34)

    def test_axis_and_nan(self):
        x = cp.array([[1.0, 2.0, 3.0, 4.0, cp.nan], [1.0, 4.0, 5.0, 8.0, 9.0]])
        mad = median_abs_deviation(x, axis=1)
        assert_equal(mad.get(), np.array([np.nan, 3.0]))

    def test_nan_policy_omit_with_inf(self):
        z = cp.array([1, 3, 4, 6, 99, cp.nan, cp.inf])
        mad = median_abs_deviation(z, nan_policy="omit")
        assert_equal(mad.get(), 3.0)

    @pytest.mark.parametrize("axis", [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        x = np.zeros((3, 0, 4))
        mad = median_abs_deviation(cp.asarray(x), axis=axis)
        mad = cp.asarray(mad).get()
        assert_equal(mad, np.full_like(x.sum(axis=axis), fill_value=np.nan))

    @pytest.mark.parametrize(
        "nan_policy, expected",
        [
            ("omit", np.array([np.nan, 1.5, 1.5])),
            ("propagate", np.array([np.nan, np.nan, 1.5])),
        ],
    )
    def test_nan_policy_with_axis(self, nan_policy, expected):
        x = cp.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [1, 5, 3, 6, np.nan, np.nan],
                [5, 6, 7, 9, 9, 10],
            ]
        )
        mad = median_abs_deviation(x, nan_policy=nan_policy, axis=1)
        assert_equal(mad.get(), expected)

    @pytest.mark.parametrize("axis, expected", [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        x = cp.array([[1, 2, 4, 9, np.nan], [0, 1, 1, 1, 12], [-10, -10, -10, 20, 20]])
        mad = median_abs_deviation(x, center=cp.mean, nan_policy="omit", axis=axis)
        assert_allclose(mad.get(), expected, rtol=1e-15, atol=1e-15)

    def test_center_not_callable(self):
        with pytest.raises(TypeError, match="callable"):
            median_abs_deviation([1, 2, 3, 5], center=99)


class TestMedianAbsDeviationMed:
    """
    Test MedianAbsDeviation that also returns the
    central value
    """

    def setup_class(self):
        """
        For all the tests
        """
        self.dat_nan = cp.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                cp.nan,
            ]
        )
        self.dat = cp.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                28.95,
            ]
        )

    def test_median_abs_deviation(self):
        """
        None axis
        """
        mad, center = median_abs_deviation_med(self.dat, axis=None)
        np.testing.assert_almost_equal(mad, 0.355)
        np.testing.assert_almost_equal(center, cp.median(self.dat, axis=None))

        dat = self.dat.reshape(6, 4)
        mad, center = median_abs_deviation_med(dat, axis=0)
        mad_expected = np.asarray([0.435, 0.5, 0.45, 0.4])
        np.testing.assert_array_almost_equal(mad.get(), mad_expected)
        np.testing.assert_array_almost_equal(center.get(), cp.median(dat, axis=0).get())

    def test_mad_nan_omit(self):
        """
        Omit nans, not sure if my center policy is the same as MAD
        """
        mad, center = median_abs_deviation_med(self.dat_nan, nan_policy="omit")
        np.testing.assert_almost_equal(mad, 0.34)
        np.testing.assert_almost_equal(center, cp.nanmedian(center))

    @staticmethod
    def test_axis_and_nan():
        """
        tests axis 1 with nans
        """
        arr = cp.array([[1.0, 2.0, 3.0, 4.0, cp.nan], [1.0, 4.0, 5.0, 8.0, 9.0]])
        mad, center = median_abs_deviation_med(arr, axis=1)
        np.testing.assert_equal(mad.get(), np.array([cp.nan, 3.0]))
        np.testing.assert_equal(center.get(), cp.median(arr, axis=1).get())

    @staticmethod
    def test_nan_policy_omit_with_inf():
        """ "
        Test with nan and inf
        """
        arr = cp.array([1, 3, 4, 6, 99, np.nan, np.inf])
        mad, center = median_abs_deviation_med(arr, nan_policy="omit")
        np.testing.assert_equal(mad.get(), 3.0)
        np.testing.assert_equal(center.get(), cp.median(arr).get())

    @pytest.mark.parametrize("axis", [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        """
        zeros axis
        """
        arr = cp.zeros((3, 0, 4))
        mad, _ = median_abs_deviation_med(arr, axis=axis)
        mad = cp.asarray(mad).get()
        np.testing.assert_equal(
            mad, np.full_like(arr.get().sum(axis=axis), fill_value=np.nan)
        )

    @pytest.mark.parametrize(
        "nan_policy, expected",
        [
            ("omit", cp.array([np.nan, 1.5, 1.5])),
            ("propagate", cp.array([cp.nan, cp.nan, 1.5])),
        ],
    )
    def test_nan_policy_with_axis(self, nan_policy, expected):
        """
        tests with nans along eaxis
        """
        arr = cp.array(
            [
                [cp.nan, cp.nan, cp.nan, cp.nan, cp.nan, cp.nan],
                [1, 5, 3, 6, cp.nan, cp.nan],
                [5, 6, 7, 9, 9, 10],
            ]
        )
        with warnings.catch_warnings(record=True):
            # warning is expected, so catch it
            mad, _ = median_abs_deviation_med(arr, nan_policy=nan_policy, axis=1)
        np.testing.assert_equal(mad.get(), expected.get())

    @pytest.mark.parametrize("axis, expected", [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        """
        test with a differnent centering method
        """
        arr = cp.array(
            [[1, 2, 4, 9, cp.nan], [0, 1, 1, 1, 12], [-10, -10, -10, 20, 20]]
        )
        mad, center = median_abs_deviation_med(
            arr, center=cp.nanmean, nan_policy="omit", axis=axis
        )
        # print(mad)
        # print(expected)
        np.testing.assert_allclose(mad.get(), expected, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(center.get(), cp.nanmean(arr, axis=axis).get())

    @staticmethod
    def test_center_not_callable():
        """
        Center finder must be callable
        """
        with pytest.raises(TypeError, match="callable"):
            median_abs_deviation_med([1, 2, 3, 5], center=99)
