#!/usr/bin/env python3
"""
Tests for calculator.py
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy

import jess.calculators as calc

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201


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


class TestDecimate:
    """
    Make random data, decimate using scipy.signal.decimate and
    jess.calculator.mean
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.random = np.random.normal(loc=10, size=512 * 512).reshape(512, 512)

    def test_decimate_singal(self):
        """
        Test decimate using scipy.singal.decimate backend
        """

        decimated = calc.decimate(self.random, time_factor=2, freq_factor=2)
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


def test_flattner_median():
    """
    Flatten a 2D array with a trend
    """
    rands = np.random.normal(size=512 * 256).reshape(512, 256)
    line = 5 + 10 * np.arange(512)
    rands_with_trend = rands + line[:, None]
    flattened = calc.flattner_median(rands_with_trend)
    rands -= np.median(rands, axis=0)
    rands -= np.median(rands, axis=1)[:, None]
    rands += 64
    assert np.allclose(rands, flattened, rtol=0.1)


def test_flattner_mix():
    """
    Flatten a 2D array with a trend
    """
    rands = np.random.normal(size=512 * 256).reshape(512, 256)
    line = 5 + 10 * np.arange(512)
    rands_with_trend = rands + line[:, None]
    flattened = calc.flattner_mix(rands_with_trend)
    rands -= np.median(rands, axis=0)
    rands -= np.mean(rands, axis=1)[:, None]
    rands += 64
    assert np.allclose(rands, flattened, rtol=0.1)


def test_highpass_window():
    """
    Length 7 half blackman window
    """
    window_7 = 1 - np.blackman(2 * 7)[7:]
    np.array_equal(window_7, calc.highpass_window(7))


class TestPreprocess:
    """
    Make random data, change mean/std and make
    sure preprocess removes it
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.random = np.random.normal(size=512 * 512).reshape(512, 512)

    def test_hor(self):
        """
        Test change in row of data
        """
        random_0 = self.random.copy() - self.random.mean(axis=0)
        random_0 /= np.std(random_0, axis=0, ddof=1)
        modified_0 = self.random.copy()
        modified_0[:, 12] += 12
        modified_0[:, 24] *= 24
        assert np.allclose(random_0, calc.preprocess(modified_0)[0])

    def test_vert(self):
        """
        Test change in column of data
        """
        random_1 = self.random - self.random.mean(axis=1)[:, None]
        random_1 /= np.std(random_1, axis=1, ddof=1)[:, None]
        modified_1 = self.random.copy()
        modified_1[12, :] += 12
        modified_1[24, :] *= 24
        assert np.allclose(random_1, calc.preprocess(modified_1)[1])


def test_entropy():
    """
    Calculate entropy of random data
    """
    random = np.random.normal(size=512 * 512).reshape(512, 512)
    entropies = np.zeros(512)
    for j in range(0, 512):
        _, counts = np.unique(random[j], return_counts=True)
        entropies[j] = entropy(counts)

    assert np.array_equal(entropies, calc.shannon_entropy(random))


class DivideRange:
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
    random = np.random.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = np.around(random)
    random_8 = np.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert np.array_equal(random_8, calc.to_dtype(random, np.uint8))
