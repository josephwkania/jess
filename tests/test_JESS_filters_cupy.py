#!/usr/bin/env python3
"""
Tests for dedispersion.py
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
        )

        assert cp.allclose(self.fake, clean, rtol=0.20)

    def test_mad_mask(self):
        """
        Test if mask is correct.

        This mask is defined opposability as normal
        True = good data
        """
        _, mask = Jf.mad_spectra(
            self.fake_with_rfi, chans_per_subband=16, sigma=15, return_mask=True
        )

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

        fake_clean = Jf.fft_mad(self.fake_with_rfi, chans_per_subband=32, sigma=7)
        fake_clean[:, self.mid] = fake_clean[:, self.mid] - self.average_power

        assert cp.allclose(self.fake, fake_clean, rtol=0.05)

    def test_mask(self):
        """
        Test if mask is correct

        needded to up the sigma to 7, while cpu version
        works at 6
        """
        _, mask = Jf.fft_mad(
            self.fake_with_rfi, chans_per_subband=32, return_mask=True, sigma=7
        )
        mask_true = cp.zeros_like(mask, dtype=bool)
        mask_true[self.max_bin, self.mid] = True

        assert cp.array_equal(mask, mask_true)


class TestMadSpectraFlat:
    """ "
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
        )

        assert cp.allclose(fake_clean, self.fake, rtol=0.1)

    def test_mask(self):
        """
        Test if returned mask is correct
        """
        _, mask = Jf.mad_spectra_flat(
            self.fake_with_rfi, chans_per_subband=32, sigma=7, return_mask=True
        )
        mask_true = cp.zeros_like(mask, dtype=bool)
        mask_true[12, 12] = True
        mask_true[20, 20] = True
        mask_true[40, 40] = True

        assert cp.array_equal(mask, mask_true)
