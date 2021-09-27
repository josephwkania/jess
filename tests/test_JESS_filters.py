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
from jess.calculators import autocorrelate, shannon_entropy

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
        mask = Jf.central_limit_masker(self.rand, window=self.window, sigma=7.5)

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
            self.rand, window=self.window, remove_lower=False, sigma=7.5
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

    def test_autocorrelation(self):
        """
        Test autocorrelation
        """

        autocorrelation = np.zeros(
            (self.nsamps // self.window, self.nchans), dtype=float
        )
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )
            autocorr_abs = np.abs(autocorrelate(chunk, axis=0))
            autocorrelation[j, :] = autocorr_abs.sum(axis=0)

        autocorrelate_calculate = Jf.autocorrelation_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(autocorrelation, autocorrelate_calculate)

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

    def test_jarque_bera(self):
        """
        Test Jarque Bera
        """
        jarque_bera = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )
            for kchan in range(self.nchans):
                jarque_bera[j, kchan] = stats.jarque_bera(chunk[:, kchan]).statistic

        jb_calcululate = Jf.jarque_bera_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(jarque_bera, jb_calcululate)

    def test_lilliefors(self):
        """
        Test Lilliefors
        """
        lilliefors = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )
            chunk -= np.median(chunk, axis=0)
            chunk = chunk / stats.median_abs_deviation(chunk, axis=0)
            for kchan in range(self.nchans):
                lilliefors[j, kchan] = stats.kstest(chunk[:, kchan], "norm").statistic

        lilliefors_calcululate = Jf.lilliefors_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(lilliefors, lilliefors_calcululate)

    def test_std_iqr(self):
        """
        Test Std-IQR
        """
        bandpass_kernel = 7
        std_iqr = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )

            std_iqr[j, :] = np.std(chunk, axis=0) - signal.medfilt(
                stats.iqr(chunk, axis=0, scale="normal"), kernel_size=bandpass_kernel
            )

        std_calcululate = Jf.std_iqr_calculate_values(
            self.yr_file,
            window=self.window,
            time_median_kernel=self.kernel_size,
            bandpass_kernel=bandpass_kernel,
        )
        assert np.array_equal(std_iqr, std_calcululate)

    def test_skew(self):
        """
        Test skew
        """
        skew = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )

            skew[j, :] = stats.skew(chunk, axis=0)

        skew_calculate = Jf.skew_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(skew, skew_calculate)

    def test_kurtosis(self):
        """
        Test kurtosis
        """
        kurtosis = np.zeros((self.nsamps // self.window, self.nchans), dtype=float)
        for j, jstart in enumerate(range(0, self.nsamps, self.window)):
            chunk = self.data[jstart : jstart + self.window]
            chunk = (
                chunk
                - signal.medfilt(np.mean(chunk, axis=1), kernel_size=self.kernel_size)[
                    :, None
                ]
            )

            kurtosis[j, :] = stats.kurtosis(chunk, axis=0)

        kurtosis_calculate = Jf.kurtosis_calculate_values(
            self.yr_file, window=self.window, time_median_kernel=self.kernel_size
        )
        assert np.array_equal(kurtosis, kurtosis_calculate)


class TestMadSpectra:
    """
    Remove outliers from random data
    """

    def setup_class(self):
        """
        Set up some random data, add spikes
        """
        self.fake = np.random.normal(loc=64, scale=5, size=256 * 32).reshape(256, 32)
        self.fake = np.around(self.fake).astype("uint8")
        self.fake_with_rfi = self.fake.copy()

        self.fake_with_rfi[200, 10] += 140
        self.fake_with_rfi[12, 12] += 140
        self.fake_with_rfi[0, 0] += 140

    def test_mad_spectra(self):
        """
        Test if clean data is close to original data
        """
        clean = Jf.mad_spectra(
            self.fake_with_rfi.copy(), chans_per_subband=16, sigma=15
        )

        assert np.allclose(self.fake, clean, rtol=0.20)

    def test_mad_mask(self):
        """
        Test if mask is correct
        """
        _, mask = Jf.mad_spectra(
            self.fake_with_rfi, chans_per_subband=16, sigma=15, return_mask=True
        )

        mask_true = np.zeros_like(mask, dtype=bool)
        mask_true[200, 10] = True
        mask_true[12, 12] = True
        mask_true[0, 0] = True
        print(mask)
        print(mask_true)
        assert np.array_equal(mask, mask_true)
