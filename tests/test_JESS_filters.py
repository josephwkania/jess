#!/usr/bin/env python3
"""
Test for JESS_filters
"""

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201

import numpy as np
from scipy import signal, stats
from your import Your

import jess.JESS_filters as Jf
from jess.calculators import shannon_entropy

# class TestRunFilter:
#     """
#     Tests for run_filter
#     """

#     def test_run_filter_anserson():
#         """
#         Run anderson
#         """


class TestCentralLimit:
    """
    Test for outliers
    """

    def setup_class(self):
        """
        Make the array with some outliers
        """
        self.rand = np.random.normal(size=1024 * 512).reshape(1024, 512)
        self.rand[200, 200] += 8
        self.rand[300, 300] -= 8
        self.rand[244, 244] += 30
        self.rand[333, 333] -= 30

        self.window = 4

    def test_central_limit_masker(self):
        """
        Flag point above and below
        """
        should_mask = np.zeros((1024, 512), dtype=bool)
        should_mask[200, 200] += 8
        should_mask[300, 300] -= 8
        should_mask[244, 244] += 30
        should_mask[333, 333] -= 30
        should_mask = np.repeat(should_mask, self.window, axis=0)
        mask = Jf.central_limit_masker(self.rand, window=self.window, sigma=6)

        assert np.array_equal(should_mask, mask)

    def test_central_limit_masker_no_lower(self):
        """
        Don't flag points below
        """
        should_mask = np.zeros((1024, 512), dtype=bool)
        should_mask[200, 200] += 8
        # should_mask[300, 300] -= 8
        should_mask[244, 244] += 30
        # should_mask[333, 333] -= 30
        should_mask = np.repeat(should_mask, self.window, axis=0)
        mask = Jf.central_limit_masker(
            self.rand, window=self.window, remove_lower=False, sigma=6
        )

        assert np.array_equal(should_mask, mask)


class TestStatTest:
    """
    Class to hold stat tests
    """

    def setup_class(self):
        """
        Use this file for all the tests.
        fake.npy and fake.fil have the same
        data
        """
        self.yr_file = Your("tests/fake.fil")
        self.data = np.load("tests/fake.npy")

        self.nsamps, self.nchans = self.data.shape
        self.window = 256
        self.kernel_size = 3

    def test_anderson(self):
        """
        Test Anderson Darling
        """
        anderson = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )
            for kchan in range(self.nchans):
                anderson[j, kchan] = stats.anderson(chunk[:, kchan]).statistic

        ad_calcululate = Jf.anderson_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(anderson, ad_calcululate)

    def test_dagostino(self):
        """
        Test D'Agostino
        """

        dagostino = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )
            dagostino[j, :] = stats.normaltest(chunk, axis=0).statistic

        dagostino_calculate = Jf.dagostino_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(dagostino, dagostino_calculate)

    def test_entropy(self):
        """
        Test shannon entropy
        """
        entropy = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )

            entropy[j, :] = shannon_entropy(chunk, axis=0)

        entropy_calculate = Jf.entropy_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(entropy, entropy_calculate)
