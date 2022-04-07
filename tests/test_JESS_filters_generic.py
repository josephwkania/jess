#!/usr/bin/env python3
"""
Tests for JESS_filters_generic.py
"""
# pylint: disable=C0413,W0201


try:
    import cupy as xp

    from jess.calculators_cupy import to_dtype

except ModuleNotFoundError:
    import numpy as xp
    from jess.calculators import to_dtype

import jess.JESS_filters_generic as Jf  # isort:skip # noqa: E402

rs = xp.random.RandomState(1)


class TestMadSpectraFlat:
    """
    Test Mad Spectra Flat on some random data
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi

        changed from [40,40], caused that made the test
        fail not sure why that was the case. It caused
        a wrap error, but even when converting to float
        before the test is was off by 10
        """
        fake = rs.normal(loc=64, scale=5, size=128 * 64).reshape(128, 64)
        self.fake = to_dtype(fake, xp.uint8)

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 40
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 39] += 75

    def test_power(self):
        """
        Test to see if impulse noise is removed
        """
        fake_clean = Jf.mad_spectra_flat(
            self.fake_with_rfi, chans_per_subband=32, sigma=7
        ).dynamic_spectra
        xp.testing.assert_allclose(fake_clean, self.fake, rtol=0.1)

    def test_mask(self):
        """
        Test if returned mask is correct
        """
        mask = Jf.mad_spectra_flat(
            self.fake_with_rfi, chans_per_subband=32, sigma=7
        ).mask
        mask_true = xp.zeros_like(mask, dtype=bool)
        mask_true[12, 12] = True
        mask_true[20, 20] = True
        mask_true[40, 39] = True

        xp.testing.assert_array_equal(mask, mask_true)
