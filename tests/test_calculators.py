#!/usr/bin/env python3
"""
Tests for calculator.py
"""

# import pytest
import numpy as np
from scipy import signal
from scipy.stats import entropy

import jess.calculators as calc


class TestAccumulate:
    """
    Accumulate a matrix on both axes
    """

    @staticmethod
    def test_vert_accumulate():
        """
        Accumulate along axis=0
        """
        to_accumulate = np.asarray(
            [6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]]
        )
        accumulates_vert = calc.accumulate(to_accumulate, factor=2, axis=0)
        assert np.array_equal(
            accumulates_vert, np.asarray([6 * [1], 6 * [5], 6 * [11]])
        )

    @staticmethod
    def test_hor_accumulate():
        """
        Accumulate on axis=1
        """
        to_accumulate = np.asarray(
            [6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]]
        )
        accumulates_hor = calc.accumulate(to_accumulate, factor=2, axis=1)
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

    @staticmethod
    def test_vert_mean():
        """
        Mean along axis=0
        """
        to_mean = np.asarray([6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]])
        mean_vert = calc.mean(to_mean, factor=2, axis=0)
        assert np.array_equal(
            mean_vert, np.asarray([6 * [1 / 2], 6 * [5 / 2], 6 * [11 / 2]])
        )

    @staticmethod
    def test_hor_mean():
        """
        Mean along axis=1
        """
        to_mean = np.asarray([6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]])
        mean_hor = calc.mean(to_mean, factor=2, axis=1)
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

    @staticmethod
    def test_decimate_singal():
        """
        Test decimate using scipy.singal.decimate backend
        """
        random = np.random.normal(loc=10, size=512 * 512).reshape(512, 512)
        decimated = calc.decimate(random, time_factor=2, freq_factor=2)
        # random -= np.median(random, axis=0)
        test = signal.decimate(random, 2, axis=0)

        test -= np.median(test, axis=0)
        test = signal.decimate(test, 2, axis=1)
        test -= np.median(test, axis=0)
        assert np.allclose(test, decimated)

    @staticmethod
    def test_decimage_mean():
        """
        Test decimate using jess.calculators.mean backend
        """
        random = np.random.normal(loc=10, size=512 * 512).reshape(512, 512)
        decimated = calc.decimate(
            random, time_factor=2, freq_factor=2, backend=calc.mean
        )
        # random -= np.median(random, axis=0)
        test = calc.mean(random, 2, axis=0)

        test -= np.median(test, axis=0)
        test = calc.mean(test, 2, axis=1)
        test -= np.median(test, axis=0)
        assert np.allclose(test, decimated)


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

    @staticmethod
    def test_hor():
        """
        Test change in row of data
        """
        random = np.random.normal(size=512 * 512).reshape(512, 512)

        random_0 = random.copy() - random.mean(axis=0)
        random_0 /= np.std(random_0, axis=0, ddof=1)
        modified_0 = random.copy()
        modified_0[:, 12] += 12
        modified_0[:, 24] *= 24
        assert np.allclose(random_0, calc.preprocess(modified_0)[0])

    @staticmethod
    def test_vert():
        """
        Test change in column of data
        """
        random = np.random.normal(size=512 * 512).reshape(512, 512)
        random_1 = random - random.mean(axis=1)[:, None]
        random_1 /= np.std(random_1, axis=1, ddof=1)[:, None]
        modified_1 = random.copy()
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


def test_to_dtype():
    """
    Create some random data and turn it into uint8
    """
    random = np.random.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = np.around(random)
    random_8 = np.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert np.array_equal(random_8, calc.to_dtype(random, np.uint8))
