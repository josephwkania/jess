#!/usr/bin/env python3
"""
Tests for JESS_filters.py
"""
# pylint: disable=C0413,W0201

import numpy as np
import pytest

from jess.calculators import to_dtype

cp = pytest.importorskip("cupy")

import jess.JESS_filters_cupy as Jf  # isort:skip # noqa: E402

rs = cp.random.RandomState(1)


class TestMadSpectra:
    """
    Remove outliers from random data
    """

    def setup_class(self):
        """
        Set up some random data, add spikes
        """
        self.fake = rs.normal(loc=64, scale=5, size=256 * 32).reshape(256, 32)
        self.fake = cp.around(self.fake).astype("uint8")
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
        ).dynamic_spectra

        assert cp.allclose(self.fake, clean, rtol=0.20)

    def test_mad_mask(self):
        """
        Test if mask is correct.

        This mask is defined opposability as normal
        True = good data
        """
        mask = Jf.mad_spectra(self.fake_with_rfi, chans_per_subband=16, sigma=15).mask

        mask_true = cp.ones_like(mask, dtype=bool)
        mask_true[200, 10] = False
        mask_true[12, 12] = False
        mask_true[0, 0] = False

        assert cp.array_equal(mask, mask_true)


class TestFftMad:
    """
    Put sine wave into fake data, and then remove it
    """

    def setup_class(self):
        """
        Load data
        """
        self.fake = cp.load("tests/fake.npy")
        self.fake_with_rfi = self.fake.copy()

        sin = to_dtype(
            15 * np.sin(np.linspace(0, 12 * np.pi, self.fake.shape[0])) + 15, "uint8"
        )
        sin = cp.asarray(sin)
        self.average_power = sin.mean()
        self.mid = self.fake.shape[1] // 2
        self.fake_with_rfi[:, self.mid] += sin

        # remove power so no DC spike
        sin_fftd = np.fft.rfft(sin - self.average_power)
        sin_fftd_abs = cp.abs(sin_fftd)
        self.max_bin = cp.argmax(sin_fftd_abs)

    def test_power(self):
        """
        Test removing sine wave

        needded to up the sigma to 7, while cpu version
        works at 6
        """

        fake_clean = Jf.fft_mad(
            self.fake_with_rfi, chans_per_subband=32, sigma=7
        ).dynamic_spectra
        fake_clean[:, self.mid] = fake_clean[:, self.mid] - self.average_power

        assert cp.allclose(self.fake, fake_clean, rtol=0.05)

    def test_mask(self):
        """
        Test if mask is correct

        needded to up the sigma to 7, while cpu version
        works at 6
        """
        mask = Jf.fft_mad(self.fake_with_rfi, chans_per_subband=32, sigma=7).mask
        mask_true = cp.zeros_like(mask, dtype=bool)
        mask_true[self.max_bin, self.mid] = True

        assert cp.array_equal(mask, mask_true)

    def test_zero_channel(self):
        """
        Test if channel information gets removed
        """
        bad_chans = cp.asarray([15])
        fake_clean = Jf.fft_mad(
            self.fake_with_rfi, chans_per_subband=32, bad_chans=bad_chans
        ).dynamic_spectra
        assert cp.isclose(cp.std(fake_clean, axis=0)[bad_chans], 0)


class TestMadSpectraFlat:
    """
    Test Mad Spectra Flat on some random data
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi
        """
        fake = rs.normal(loc=64, scale=5, size=128 * 64).reshape(128, 64)
        self.fake = to_dtype(fake, "uint8")

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 40
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 40] += 75

    def test_power(self):
        """
        Test to see if impulse noise is removed
        """
        fake_clean = Jf.mad_spectra_flat(
            self.fake_with_rfi, chans_per_subband=32, sigma=7
        ).dynamic_spectra

        assert cp.allclose(fake_clean, self.fake, rtol=0.1)

    def test_mask(self):
        """
        Test if returned mask is correct
        """
        mask = Jf.mad_spectra_flat(
            self.fake_with_rfi, chans_per_subband=32, sigma=7
        ).mask
        mask_true = cp.zeros_like(mask, dtype=bool)
        mask_true[12, 12] = True
        mask_true[20, 20] = True
        mask_true[40, 40] = True

        assert cp.array_equal(mask, mask_true)


def test_zero_dm():
    """
    Test if the mean is removed.
    Add and remove equal amounts so total power
    stays the same
    """

    data = cp.load("tests/fake.npy")
    bandpass = data.mean(axis=0)
    data_flat = data - data.mean(axis=1)[:, None] + bandpass
    data_flat = to_dtype(data_flat, "uint8")

    data_with_rfi = data.copy()
    data_with_rfi[15] += 15
    data_with_rfi[17] -= 15
    data_with_rfi[20] += 20
    data_with_rfi[25] -= 20

    assert cp.array_equal(Jf.zero_dm(data_with_rfi).dynamic_spectra, data_flat)
