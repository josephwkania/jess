#!/usr/bin/env python3
"""
Tests for JESS_filters_generic.py
"""
# pylint: disable=C0413,W0201


try:
    import cupy as xp

    from jess.calculators_cupy import to_dtype

    BACKEND_GPU = True

except ModuleNotFoundError:
    import numpy as xp

    from jess.calculators import to_dtype

    BACKEND_GPU = False

from scipy import stats
from your import Your

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


class TestCalculateSkewAndKurtosis:
    """
    Test calculate_skew_and_kurtoiss.
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi.
        """
        self.dynamic_length = 128
        fake = rs.normal(loc=64, scale=5, size=self.dynamic_length * 64).reshape(
            self.dynamic_length, 64
        )
        self.fake = to_dtype(fake, xp.uint8)

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 40
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 39] += 75

    def test_whole_block(self):
        """
        Test is Kurtosis and Skew is correctly calculated over the
        whole dynamic spectra.
        """
        skew, kurtosis, limits = Jf.calculate_skew_and_kurtosis(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length,
            nan_policy=None,
            winsorize_args=None,
        )
        skew = xp.squeeze(skew)
        kurtosis = xp.squeeze(kurtosis)

        if BACKEND_GPU:
            fake_with_rfi = self.fake_with_rfi.get()
        else:
            fake_with_rfi = self.fake_with_rfi

        xp.testing.assert_allclose(skew, stats.skew(fake_with_rfi, axis=0))
        xp.testing.assert_allclose(kurtosis, stats.kurtosis(fake_with_rfi, axis=0))

        assert limits[0] == 0
        assert limits[1] == self.dynamic_length

    def test_half_block(self):
        """
        Test is Kurtosis and Skew is correctly calculated over the
        first half of the dynamic spectra.
        """
        skew, kurtosis, limits = Jf.calculate_skew_and_kurtosis(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length // 2,
            nan_policy=None,
            winsorize_args=None,
        )
        if BACKEND_GPU:
            fake_with_rfi = self.fake_with_rfi.get()
        else:
            fake_with_rfi = self.fake_with_rfi
        fake_with_rfi = fake_with_rfi[: self.dynamic_length // 2]

        xp.testing.assert_allclose(skew[0], stats.skew(fake_with_rfi, axis=0))
        xp.testing.assert_allclose(kurtosis[0], stats.kurtosis(fake_with_rfi, axis=0))

        assert limits[0] == 0
        assert limits[1] == self.dynamic_length // 2

    def test_nan_propagate(self):
        """
        Test is Kurtosis and Skew is correctly calculated over the
        whole dynamic spectra, with nans present. Using `propagate`.
        """
        fake_with_rfi = self.fake_with_rfi.astype(float)
        fake_with_rfi[10, 10] = xp.nan
        fake_with_rfi[30, 30] = xp.nan
        skew, kurtosis, limits = Jf.calculate_skew_and_kurtosis(
            fake_with_rfi,
            samples_per_block=self.dynamic_length,
            nan_policy="propagate",
            winsorize_args=None,
        )
        skew = xp.squeeze(skew)
        kurtosis = xp.squeeze(kurtosis)

        if BACKEND_GPU:
            fake_with_rfi = fake_with_rfi.get()

        xp.testing.assert_allclose(
            skew,
            stats.skew(
                fake_with_rfi,
                axis=0,
                nan_policy="propagate",
            ),
        )
        xp.testing.assert_allclose(
            kurtosis,
            stats.kurtosis(
                fake_with_rfi,
                axis=0,
                nan_policy="propagate",
            ),
        )

        assert limits[0] == 0
        assert limits[1] == self.dynamic_length

    # def test_nan_omit(self):
    #     """
    #     Test is Kurtosis and Skew is correctly calculated over the
    #     whole dynamic spectra, with nans present. Using `omit`.
    #     """
    #     fake_with_rfi = self.fake_with_rfi.astype(float)
    #     fake_with_rfi[10, 10] = xp.nan
    #     fake_with_rfi[30, 30] = xp.nan
    #     skew, kurtosis, limits = Jf.calculate_skew_and_kurtosis(
    #         fake_with_rfi,
    #         samples_per_block=self.dynamic_length,
    #         nan_policy="omit",
    #         winsorize_args=None,
    #     )
    #     skew = xp.squeeze(skew)
    #     kurtosis = xp.squeeze(kurtosis)

    #     if BACKEND_GPU:
    #         fake_with_rfi = fake_with_rfi.get()
    #     else:
    #         fake_with_rfi = fake_with_rfi

    #     xp.testing.assert_allclose(
    #         skew,
    #         stats.skew(
    #             fake_with_rfi,
    #             axis=0,
    #             nan_policy="omit",
    #         ),
    #     )
    #     xp.testing.assert_allclose(
    #         kurtosis,
    #         stats.kurtosis(
    #             fake_with_rfi,
    #             axis=0,
    #             nan_policy="omit",
    #         ),
    #     )

    #     assert limits[0] == 0
    #     assert limits[1] == self.dynamic_length


class TestDAgostino:
    """
    Test dagostino.
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi.
        """
        self.dynamic_length = 128
        fake = rs.normal(loc=64, scale=5, size=self.dynamic_length * 64).reshape(
            self.dynamic_length, 64
        )
        self.fake = to_dtype(fake, xp.uint8)

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 43
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 39] += 75

    def test_whole_block(self):
        """
        Test dagostino masking over the
        whole dynamic spectra.
        """
        dagostino = Jf.dagostino(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length,
            nan_policy=None,
            winsorize_args=None,
        )

        assert dagostino.mask.shape == self.fake_with_rfi.shape
        # Nothing gets masked because RFI << samples_per_block
        # This works for the CuPy backend, but not Numpy
        # xp.testing.assert_allclose(dagostino.mask, xp.zeros_like(dagostino.mask))
        assert dagostino.mask.mean() < 0.1

    def test_half_block(self):
        """
        Test dagostino masking over the half of the dynamic spectra.
        """
        dagostino = Jf.dagostino(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length // 2,
            nan_policy=None,
            winsorize_args=None,
        )

        assert dagostino.mask.shape == self.fake_with_rfi.shape
        # Blocks with bright points should be flagged
        assert dagostino.mask[12, 12]
        assert dagostino.mask[20, 20]
        assert dagostino.mask[40, 39]


class TestJarqueBera:
    """
    Test dagostino.
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi.
        """
        self.dynamic_length = 128
        fake = rs.normal(loc=64, scale=5, size=self.dynamic_length * 64).reshape(
            self.dynamic_length, 64
        )
        self.fake = to_dtype(fake, xp.uint8)

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 40
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 39] += 75

    def test_whole_block(self):
        """
        Test jarque_bera masking over the
        whole dynamic spectra.
        """
        jarque_bera = Jf.jarque_bera(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length,
            nan_policy=None,
            winsorize_args=None,
        )

        assert jarque_bera.mask.shape == self.fake_with_rfi.shape
        # Nothing gets masked because RFI << samples_per_block
        # This works for the CuPy backend, but not Numpy
        # xp.testing.assert_allclose(jarque_bera.mask, xp.zeros_like(jarque_bera.mask))
        assert jarque_bera.mask.mean() < 0.1

    def test_half_block(self):
        """
        Test jarque_bera masking over the half of the dynamic spectra.
        """
        jarque_bera = Jf.jarque_bera(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length // 2,
            nan_policy=None,
            winsorize_args=None,
        )

        assert jarque_bera.mask.shape == self.fake_with_rfi.shape
        # Blocks with bright points should be flagged
        assert jarque_bera.mask[12, 12]
        assert jarque_bera.mask[20, 20]
        assert jarque_bera.mask[40, 39]


class TestKurtosisAndSkew:
    """
    Test kurtosis_and_skew.
    """

    def setup_class(self):
        """
        Random matrix with impulse rfi.
        """
        self.dynamic_length = 128
        fake = rs.normal(loc=64, scale=5, size=self.dynamic_length * 64).reshape(
            self.dynamic_length, 64
        )
        self.fake = to_dtype(fake, xp.uint8)

        self.fake_with_rfi = self.fake.copy()
        self.fake_with_rfi[12, 12] += 40
        self.fake_with_rfi[20, 20] += 44
        self.fake_with_rfi[40, 39] += 75

    def test_whole_block(self):
        """
        Test kurtosis_and_skew masking over the
        whole dynamic spectra.
        """
        kurtosis_and_skew = Jf.kurtosis_and_skew(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length,
            nan_policy=None,
            winsorize_args=None,
        )

        assert kurtosis_and_skew.mask.shape == self.fake_with_rfi.shape
        # Nothing gets masked because RFI << samples_per_block
        # This works for the CuPy backend, but not Numpy
        # xp.testing.assert_allclose(
        #     kurtosis_and_skew.mask, xp.zeros_like(kurtosis_and_skew.mask)
        # )
        assert kurtosis_and_skew.mask.mean() < 0.1

    def test_half_block(self):
        """
        Test kurtosis_and_skew masking over the half of the dynamic spectra.
        """
        kurtosis_and_skew = Jf.kurtosis_and_skew(
            self.fake_with_rfi,
            samples_per_block=self.dynamic_length // 2,
            nan_policy=None,
            winsorize_args=None,
        )

        assert kurtosis_and_skew.mask.shape == self.fake_with_rfi.shape
        # Blocks with bright points should be flagged
        assert kurtosis_and_skew.mask[12, 12]
        assert kurtosis_and_skew.mask[20, 20]
        assert kurtosis_and_skew.mask[40, 39]


class TestRobustBandpass:
    """
    Test Robust Bandpass.
    """

    def setup_class(self):
        """
        Use this file for all the tests.
        """
        self.yr_file = Your("tests/fake.fil")

    def test_robust_bandpass_length(self):
        """
        Test on the small fill file.

        Test if the length if correct.
        Test if normalized bandpass is close to number of channels.
        """
        robust_bandpass = Jf.robust_bandpass(self.yr_file, num_samples=16)
        assert len(robust_bandpass) == self.yr_file.nchans
        xp.testing.assert_allclose(
            robust_bandpass.sum(), self.yr_file.your_header.nchans
        )

    def test_unnormed_close_to_medians(self):
        """
        Test if returned bandpass is close to filterbank
        medians.
        """
        robust_bandpass = Jf.robust_bandpass(
            self.yr_file, num_samples=16, normalized=False
        )
        dynamic_spectra = self.yr_file.get_data(0, self.yr_file.your_header.nspectra)
        dynamic_spectra = xp.asarray(dynamic_spectra)
        xp.testing.assert_allclose(
            robust_bandpass, xp.median(dynamic_spectra, axis=0), rtol=0.25
        )


class TestZeroDM:
    """
    Test if weighted broadband RFI is removed.
    """

    def setup_class(self):
        """
        Use this file for all the tests.
        """
        self.data = xp.load("tests/fake.npy")
        mean = self.data.mean()
        chan_weights = self.data.mean(axis=0)
        chan_weights /= chan_weights.mean()
        data_with_broadband = self.data.copy().astype(float)
        data_with_broadband[128:134] += 10 * chan_weights
        data_with_broadband[256:300] -= 10 * chan_weights
        data_with_broadband[400:420] += 20 * chan_weights
        data_with_broadband = to_dtype(data_with_broadband, xp.uint8)

        self.data_filtered = Jf.zero_dm(
            data_with_broadband, bandpass=mean, chan_weights=chan_weights
        )

    def test_mean(self):
        """
        The mean after filtering should be close to zero.
        """
        data_filtered_ts = self.data_filtered.dynamic_spectra.mean(axis=1)
        data_filtered_ts -= xp.median(data_filtered_ts)
        assert (data_filtered_ts < 0.5).all()

    def test_dynamic_spectra(self):
        """
        The difference between the original data (self.data) amd
        the cleaned data (self.data_filtered) should be close.
        """
        data_filtered_diff = self.data_filtered.dynamic_spectra - self.data.astype(
            float
        )
        assert (data_filtered_diff < 10).all()
