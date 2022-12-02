#!/usr/bin/env python3
"""
Tests for calculator.py
"""


import warnings

import numpy as np
import pytest
from scipy import ndimage, signal, stats
from your import Your

import jess.calculators as calc

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201

rs = np.random.RandomState(2021)


class TestAccumulate:
    """
    Accumulate a matrix on both axes
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.to_accumulate = np.asarray(
            [6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]]
        )

    def test_vert_accumulate(self):
        """
        Accumulate along axis=0
        """
        accumulates_vert = calc.accumulate(self.to_accumulate, factor=2, axis=0)
        assert np.array_equal(
            accumulates_vert, np.asarray([6 * [1], 6 * [5], 6 * [11]])
        )

    def test_hor_accumulate(self):
        """
        Accumulate on axis=1
        """
        accumulates_hor = calc.accumulate(self.to_accumulate, factor=2, axis=1)
        assert np.array_equal(
            accumulates_hor,
            np.asarray(
                [
                    3 * [2 * 0],
                    3 * [2 * 1],
                    3 * [2 * 2],
                    3 * [2 * 3],
                    3 * [2 * 5],
                    3 * [2 * 6],
                ]
            ),
        )

    def test_not_implemented(self):
        """
        Raise error for higher axis
        """
        with pytest.raises(NotImplementedError):
            calc.accumulate(self.to_accumulate, factor=2, axis=3)


class TestAutoCorrelate:
    """
    Test autocorrelate against np.correlate
    """

    def setup_class(self):
        """
        Random array to autocorrelate
        """
        self.rand = rs.normal(size=51 * 256).reshape(51, 256)

    @staticmethod
    def np_autocorrelate(data):
        """
        Auto correlation along a single array
        """
        data -= data.mean()
        correlation = np.correlate(data, data, mode="same")[len(data) // 2 :]
        return correlation / np.max(correlation)

    def test_array(self):
        """
        Test on a array
        """
        auto_corralate = calc.autocorrelate(self.rand[:, 0])
        assert np.allclose(auto_corralate, self.np_autocorrelate(self.rand[:, 0]))

    def test_2d_zero(self):
        """
        Test matrix along axis=0
        """
        auto_corralate = calc.autocorrelate(self.rand, axis=0)
        assert np.allclose(auto_corralate[:, 0], self.np_autocorrelate(self.rand[:, 0]))
        assert np.allclose(
            auto_corralate[:, 20], self.np_autocorrelate(self.rand[:, 20])
        )

    def test_2d_one(self):
        """
        Test matrix along axis=1
        """
        auto_corralate = calc.autocorrelate(self.rand, axis=1)
        assert np.allclose(auto_corralate[0, :], self.np_autocorrelate(self.rand[0, :]))
        assert np.allclose(
            auto_corralate[20, :], self.np_autocorrelate(self.rand[20, :])
        )

    def test_not_implemented(self):
        """
        Raise error for higher axis
        """
        with pytest.raises(NotImplementedError):
            calc.autocorrelate(self.rand, axis=3)


def test_find_largest_factor():
    """
    Test if find_largest factor retruns common powers of two
    """
    np.testing.assert_equal(calc.closest_larger_factor(4090, 16), 4096)
    np.testing.assert_equal(calc.closest_larger_factor(1020, 8), 1024)


class TestMedianAbsDeviation:
    """
    Test MedianAbsDeviation that also returns the
    central value
    """

    def setup_class(self):
        """
        For all the tests
        """
        self.dat_nan = np.array(
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
        self.dat = np.array(
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
        mad, center = calc.median_abs_deviation_med(self.dat, axis=None)
        np.testing.assert_almost_equal(mad, 0.355)
        np.testing.assert_almost_equal(center, np.median(self.dat, axis=None))

        dat = self.dat.reshape(6, 4)
        mad, center = calc.median_abs_deviation_med(dat, axis=0)
        mad_expected = np.asarray([0.435, 0.5, 0.45, 0.4])
        np.testing.assert_array_almost_equal(mad, mad_expected)
        np.testing.assert_array_almost_equal(center, np.median(dat, axis=0))

    def test_mad_nan_omit(self):
        """
        Omit nans, not sure if my center policy is the same as MAD
        """
        mad, center = calc.median_abs_deviation_med(self.dat_nan, nan_policy="omit")
        np.testing.assert_almost_equal(mad, 0.34)
        np.testing.assert_almost_equal(center, np.median(center))

    @staticmethod
    def test_axis_and_nan():
        """
        tests axis 1 with nans
        """
        arr = np.array([[1.0, 2.0, 3.0, 4.0, np.nan], [1.0, 4.0, 5.0, 8.0, 9.0]])
        mad, center = calc.median_abs_deviation_med(arr, axis=1)
        np.testing.assert_equal(mad, np.array([np.nan, 3.0]))
        np.testing.assert_equal(center, np.median(arr, axis=1))

    @staticmethod
    def test_nan_policy_omit_with_inf():
        """ "
        Test with nan and inf
        """
        arr = np.array([1, 3, 4, 6, 99, np.nan, np.inf])
        mad, center = calc.median_abs_deviation_med(arr, nan_policy="omit")
        np.testing.assert_equal(mad, 3.0)
        np.testing.assert_equal(center, np.median(arr))

    @pytest.mark.parametrize("axis", [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        """
        zeros axis
        """
        arr = np.zeros((3, 0, 4))
        mad, _ = calc.median_abs_deviation_med(arr, axis=axis)
        np.testing.assert_equal(
            mad, np.full_like(arr.sum(axis=axis), fill_value=np.nan)
        )

    @pytest.mark.parametrize(
        "nan_policy, expected",
        [
            ("omit", np.array([np.nan, 1.5, 1.5])),
            ("propagate", np.array([np.nan, np.nan, 1.5])),
        ],
    )
    def test_nan_policy_with_axis(self, nan_policy, expected):
        """
        tests with nans along eaxis
        """
        arr = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [1, 5, 3, 6, np.nan, np.nan],
                [5, 6, 7, 9, 9, 10],
            ]
        )
        with warnings.catch_warnings(record=True):
            # warning is expected, so catch it
            mad, _ = calc.median_abs_deviation_med(arr, nan_policy=nan_policy, axis=1)
        np.testing.assert_equal(mad, expected)

    @pytest.mark.parametrize("axis, expected", [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        """
        test with a differnent centering method
        """
        arr = np.array(
            [[1, 2, 4, 9, np.nan], [0, 1, 1, 1, 12], [-10, -10, -10, 20, 20]]
        )
        mad, center = calc.median_abs_deviation_med(
            arr, center=np.nanmean, nan_policy="omit", axis=axis
        )
        np.testing.assert_allclose(mad, expected, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(center, np.nanmean(arr, axis=axis))

    @staticmethod
    def test_center_not_callable():
        """
        Center finder must be callable
        """
        with pytest.raises(TypeError, match="callable"):
            calc.median_abs_deviation_med([1, 2, 3, 5], center=99)


class TestMean:
    """
    Take a mean along both axes
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.to_mean = np.asarray(
            [6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]]
        )

    def test_vert_mean(self):
        """
        Mean along axis=0
        """

        mean_vert = calc.mean(self.to_mean, factor=2, axis=0)
        assert np.array_equal(
            mean_vert, np.asarray([6 * [1 / 2], 6 * [5 / 2], 6 * [11 / 2]])
        )

    def test_hor_mean(self):
        """
        Mean along axis=1
        """

        mean_hor = calc.mean(self.to_mean, factor=2, axis=1)
        assert np.array_equal(
            mean_hor,
            np.asarray(
                [
                    3 * [2 * 0 / 2],
                    3 * [2 * 1 / 2],
                    3 * [2 * 2 / 2],
                    3 * [2 * 3 / 2],
                    3 * [2 * 5 / 2],
                    3 * [2 * 6 / 2],
                ]
            ),
        )

    def test_not_implemented(self):
        """
        Raise error for higher axis
        """
        with pytest.raises(NotImplementedError):
            calc.mean(self.to_mean, factor=2, axis=3)


class TestDecimate:
    """
    Make random data, decimate using scipy.signal.decimate and
    jess.calculator.mean
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.random = rs.normal(loc=10, size=512 * 512).reshape(512, 512)

    def test_decimate_singal(self):
        """
        Test decimate using scipy.singal.decimate backend
        """

        decimated = calc.decimate(self.random, time_factor=2.0, freq_factor=2.0)
        # random -= np.median(random, axis=0)
        test = signal.decimate(self.random, 2, axis=0)

        test -= np.median(test, axis=0)
        test = signal.decimate(test, 2, axis=1)
        test -= np.median(test, axis=0)
        assert np.allclose(test, decimated)

    def test_decimage_mean(self):
        """
        Test decimate using jess.calculators.mean backend
        """
        decimated = calc.decimate(
            self.random, time_factor=2, freq_factor=2, backend=calc.mean
        )
        # random -= np.median(random, axis=0)
        test = calc.mean(self.random, 2, axis=0)

        test -= np.median(test, axis=0)
        test = calc.mean(test, 2, axis=1)
        test -= np.median(test, axis=0)
        assert np.allclose(test, decimated)


class TestFlattener:
    """
    Test the flatteners
    """

    @staticmethod
    def test_flattner_median():
        """
        Flatten a 2D array with a trend
        """
        rands = rs.normal(size=512 * 256).reshape(512, 256)
        line = 5 + 10 * np.arange(512)
        rands_with_trend = rands + line[:, None]
        flattened = calc.flattner_median(rands_with_trend)

        rands -= np.median(rands, axis=1)[:, None]
        rands -= np.median(rands, axis=0)
        rands += 64
        assert np.allclose(rands, flattened)

    @staticmethod
    def test_flattner_median_smooth():
        """
        Flatten a 2D array with a trend

        not sure why the relative error need to be
        so high
        """
        rands = rs.normal(size=512 * 256).reshape(512, 256)
        line = 5 + 10 * np.arange(512)
        rands_with_trend = rands + line[:, None]
        flattened = calc.flattner_median(rands_with_trend, kernel_size=3)

        rands -= ndimage.median_filter(np.median(rands, axis=1), mode="mirror", size=3)[
            :, None
        ]
        rands -= ndimage.median_filter(np.median(rands, axis=0), mode="mirror", size=3)
        rands += 64
        assert np.allclose(rands, flattened, rtol=0.2)

    @staticmethod
    def test_flattner_mix():
        """
        Flatten a 2D array with a trend
        """
        rands = rs.normal(size=512 * 256).reshape(512, 256)
        line = 5 + 10 * np.arange(512)
        rands_with_trend = rands + line[:, None]
        flattened = calc.flattner_mix(rands_with_trend)

        rands -= np.median(rands, axis=1)[:, None]
        rands -= np.mean(rands, axis=0)
        rands += 64
        assert np.allclose(rands, flattened)

    @staticmethod
    def test_flattner_mix_smooth():
        """
        Flatten a 2D array with a trend

        Not sure why the relative error need to be so high
        """
        rands = rs.normal(size=512 * 256).reshape(512, 256)
        line = 5 + 10 * np.arange(512)
        rands_with_trend = rands + line[:, None]
        flattened = calc.flattner_mix(rands_with_trend, kernel_size=3)

        rands -= signal.medfilt(np.median(rands, axis=1), kernel_size=3)[:, None]
        rands -= signal.medfilt(np.mean(rands, axis=0), kernel_size=3)
        rands += 64
        np.testing.assert_allclose(rands, flattened, rtol=0.2)


def test_highpass_window():
    """
    Length 7 half blackman window
    """
    window_7 = 1 - np.blackman(2 * 7)[7:]
    np.array_equal(window_7, calc.highpass_window(7))


def test_guassian_noise_adder():
    """
    Add some standard deviations
    """
    stds = np.array([1, 1, 2, 2])
    combined = np.sqrt(np.sum(stds**2)) / 4
    assert combined == calc.guassian_noise_adder(stds)


class TestNoiseCalculator:
    """
    Test the ideal noise calculator on random
    Guassian noise.

    The relative tolerances seem a bit high
    """

    def setup_class(self):
        """
        Read data in, get standard deviations
        """
        data = np.load("tests/fake.npy")
        self.stds = np.std(data, axis=0)
        self.zero_dm_std = np.std(data.mean(axis=1))

        self.yr_object = Your("tests/fake.fil")

    def test_noise_default(self):
        """
        Test the default case
        """

        ideal_noises, std = calc.noise_calculator(self.yr_object, num_samples=4)

        np.testing.assert_allclose(self.stds, ideal_noises, rtol=0.15)
        print(self.zero_dm_std, std)
        assert np.isclose(self.zero_dm_std, std, rtol=0.05)

    def test_ideal_noise_no_detrend(self):
        """
        Test with no detrend
        """
        ideal_noises_no_detrend, std = calc.noise_calculator(
            self.yr_object, num_samples=4, detrend=False
        )
        np.testing.assert_allclose(self.stds, ideal_noises_no_detrend, rtol=0.15)
        assert np.isclose(self.zero_dm_std, std, rtol=0.05)

    def test_ideal_noise_no_kernel(self):
        """
        Test without median filter
        """
        ideal_noises_no_kernel, std = calc.noise_calculator(
            self.yr_object, num_samples=4, kernel_size=0
        )
        np.testing.assert_allclose(self.stds, ideal_noises_no_kernel, rtol=0.15)
        assert np.isclose(self.zero_dm_std, std, rtol=0.05)


class TestPreprocess:
    """
    Make random data, change mean/std and make
    sure preprocess removes it
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.random = rs.normal(size=512 * 512).reshape(512, 512)

    def test_hor_mean_std(self):
        """
        Test change in row of data
        """
        random_0 = self.random.copy() - self.random.mean(axis=0)
        random_0 /= np.std(random_0, axis=0, ddof=1)
        modified_0 = self.random.copy()
        modified_0[:, 12] += 12
        modified_0[:, 24] *= 24
        assert np.allclose(random_0, calc.preprocess(modified_0)[0])

    def test_hor_median_mad(self):
        """
        Test change in row of data
        """
        random_0 = self.random.copy() - np.median(self.random, axis=0)
        random_0 /= stats.median_abs_deviation(random_0, axis=0, scale="Normal")
        modified_0 = self.random.copy()
        modified_0[:, 12] += 12
        modified_0[:, 24] *= 24
        assert np.allclose(
            random_0,
            calc.preprocess(
                modified_0, central_value_calc="median", disperion_calc="mad"
            )[0],
        )

    def test_vert_mean_std(self):
        """
        Test change in column of data
        """
        random_1 = self.random - self.random.mean(axis=1)[:, None]
        random_1 /= np.std(random_1, axis=1, ddof=1)[:, None]
        modified_1 = self.random.copy()
        modified_1[12, :] += 12
        modified_1[24, :] *= 24
        assert np.allclose(random_1, calc.preprocess(modified_1)[1])

    def test_vert_median_mad(self):
        """
        Test change in column of data
        """
        random_1 = self.random - np.median(self.random, axis=1)[:, None]
        random_1 /= stats.median_abs_deviation(random_1, axis=1, scale="Normal")[
            :, None
        ]
        modified_1 = self.random.copy()
        modified_1[12, :] += 12
        modified_1[24, :] *= 24
        assert np.allclose(
            random_1,
            calc.preprocess(
                modified_1, central_value_calc="median", disperion_calc="mad"
            )[1],
        )

    def test_std_not_implmented(self):
        """
        Raise a error when dispersion calculator is
         not valid
        """
        with pytest.raises(NotImplementedError):
            calc.preprocess(self.random, disperion_calc="joe")

        with pytest.raises(NotImplementedError):
            calc.preprocess(self.random, central_value_calc="joe")


class TestEntropy:
    """
    Tests for entropy
    """

    def setup_class(self):
        """ "
        Holds random
        """
        self.random = rs.normal(size=512 * 512).reshape(512, 512)

    def test_entropy(self):
        """
        Calculate entropy of random data
        """
        entropies = np.zeros(512)
        for j in range(0, 512):
            _, counts = np.unique(self.random[j], return_counts=True)
            entropies[j] = stats.entropy(counts)

        assert np.array_equal(entropies, calc.shannon_entropy(self.random))

    def test_no_axisself(self):
        """
        Raise a error when dispersion calculator is
         not valid
        """
        with pytest.raises(ValueError):
            calc.shannon_entropy(self.random, axis=3)


class TestBalanceChansPerSubband:
    """
    Test Balance by trying dividable and not dividable subbands
    """

    @staticmethod
    def test_even():
        """
        Test evenly divisible
        """
        num_section, limits = calc.balance_chans_per_subband(2048, 512)
        assert num_section == 4
        np.testing.assert_array_equal(limits, np.array((0, 512, 1024, 1536, 2048)))

    @staticmethod
    def test_uneven():
        """
        Test unevenly divisible,
        should be the same as before
        """
        num_section, limits = calc.balance_chans_per_subband(2048, 500)
        assert num_section == 4
        np.testing.assert_array_equal(limits, np.array((0, 512, 1024, 1536, 2048)))

    @staticmethod
    def test_subband_larger_chans():
        """
        Test when number of number of chans < chan_per_subband
        put everything in the subband
        """
        num_chans = 500
        num_section, limits = calc.balance_chans_per_subband(num_chans, 2048)
        assert num_section == 1
        np.testing.assert_array_equal(limits, np.array((0, num_chans)))


class TestDivideRange:
    """
    Divide range into chunks, should be close to even
    """

    @staticmethod
    def test_even():
        """
        Test if array is evenly divided
        """
        assert np.array_equal(
            np.array([0, 512, 1024, 1536, 2048]), calc.divide_range(2048, 4)
        )

    @staticmethod
    def test_uneven():
        """
        test if array with remainder get divided nearly evenly

        Make it 2 shorter, the last two subarrays should be 1 smaller
        """
        assert np.array_equal(
            np.array([0, 512, 1024, 1535, 2046]), calc.divide_range(2046, 4)
        )


def test_to_dtype():
    """
    Create some random data and turn it into uint8
    """
    random = rs.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = np.around(random)
    random_8 = np.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert np.array_equal(random_8, calc.to_dtype(random, np.uint8))


class TestGetFlattenTo:
    """
    Test flatten to.
    """

    @staticmethod
    def test_8():
        """
        Test eight bit.
        """
        assert 64 == calc.get_flatten_to(8)

    @staticmethod
    def test_raise():
        """
        Below 4 bits is not implemented.
        """
        with pytest.raises(NotImplementedError):
            calc.get_flatten_to(2)
