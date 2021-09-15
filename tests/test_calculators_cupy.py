#!/usr/bin/env python3
"""
Tests for calculator_cupy.py
"""


import pytest

cp = pytest.importorskip("cupy")

# pylint: disable=C0413
import jess.calculators_cupy as calc  # isort:skip # noqa: E402


class TestMean:
    """
    Take a mean along both axes
    """

    def setup_class(self):
        """
        Holds shared array
        """
        self.to_mean = cp.asarray(
            [6 * [0], 6 * [1], 6 * [2], 6 * [3], 6 * [5], 6 * [6]]
        )

    def test_vert_mean(self):
        """
        Mean along axis=0
        """

        mean_vert = calc.mean(self.to_mean, factor=2, axis=0)
        assert cp.array_equal(
            mean_vert, cp.asarray([6 * [1 / 2], 6 * [5 / 2], 6 * [11 / 2]])
        )

    def test_hor_mean(self):
        """
        Mean along axis=1
        """

        mean_hor = calc.mean(self.to_mean, factor=2, axis=1)
        assert cp.array_equal(
            mean_hor,
            cp.asarray(
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
        self.random = cp.random.normal(loc=10, size=512 * 512).reshape(512, 512)

    def test_decimate(self):
        """
        Test decimate using jess.calculators_cupy.mean backend (only one available)
        """
        decimated = calc.decimate(self.random, time_factor=2, freq_factor=2)
        # random -= np.median(random, axis=0)
        test = calc.mean(self.random, 2, axis=0)

        test -= cp.median(test, axis=0)
        test = calc.mean(test, 2, axis=1)
        test -= cp.median(test, axis=0)
        assert cp.allclose(test, decimated)


def test_flattner_median():
    """
    Flatten a 2D array with a trend
    """
    rands = cp.random.normal(size=512 * 256).reshape(512, 256)
    line = 5 + 10 * cp.arange(512)
    rands_with_trend = rands + line[:, None]
    flattened = calc.flattner_median(rands_with_trend)
    rands -= cp.median(rands, axis=0)
    rands -= cp.median(rands, axis=1)[:, None]
    rands += 64
    assert cp.allclose(rands, flattened, rtol=0.1)


def test_flattner_mix():
    """
    Flatten a 2D array with a trend
    """
    rands = cp.random.normal(size=512 * 256).reshape(512, 256)
    line = 5 + 10 * cp.arange(512)
    rands_with_trend = rands + line[:, None]
    flattened = calc.flattner_mix(rands_with_trend)
    rands -= cp.median(rands, axis=0)
    rands -= cp.mean(rands, axis=1)[:, None]
    rands += 64
    assert cp.allclose(rands, flattened, rtol=0.1)


def test_to_dtype():
    """
    Create some random data and turn it into uint8
    """
    random = cp.random.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = cp.around(random)
    random_8 = cp.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert cp.array_equal(random_8, calc.to_dtype(random, cp.uint8))
