#!/usr/bin/env python3
"""
Tests for channel_masks
"""

import numpy as np
from scipy import stats

from jess.calculators import preprocess
from jess.channel_masks import stat_test

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201


class TestStatTest:
    """
    Tests for stat_test
    """

    def setup_class(self):
        """
        Holds shared array
        512 channels, 2k samples
        """
        self.rand = np.random.normal(size=512 * 2000).reshape(2000, 512)
        # add some features
        self.rand[:, 20] *= 20
        self.rand[:, 45] *= 45
        self.rand[:, 60][self.rand[:, 60] > 0.4] = 0

    def test_82_2(self):
        """
        82-2
        """
        top, bottom = np.quantile(self.rand, [0.98, 0.02], axis=0)
        assert np.all(top > bottom)
        diff = top - bottom
        assert len(diff) == 512
        assert np.array_equal(diff, stat_test(self.rand, "98-2"))

    def test_91_9(self):
        """
        91-9
        """
        top, bottom = np.quantile(self.rand, [0.91, 0.09], axis=0)
        assert np.all(top > bottom)
        diff = top - bottom
        assert len(diff) == 512
        assert np.array_equal(diff, stat_test(self.rand, "91-9"))

    def test_90_10(self):
        """
        90-10
        """
        top, bottom = np.quantile(self.rand, [0.90, 0.10], axis=0)
        assert np.all(top > bottom)
        diff = top - bottom
        assert len(diff) == 512
        assert np.array_equal(diff, stat_test(self.rand, "90-10"))

    def test_75_25(self):
        """
        75-25
        """
        top, bottom = np.quantile(self.rand, [0.75, 0.25], axis=0)
        assert np.all(top > bottom)
        diff = top - bottom
        assert len(diff) == 512
        assert np.array_equal(diff, stat_test(self.rand, "75-25"))

    def test_ad(self):
        """
        anderson darling
        """
        ad_stat = np.zeros(512)
        for j in range(0, 512):
            ad_stat[j], _, _ = stats.anderson(self.rand[:, j], dist="norm")
        assert np.array_equal(ad_stat, stat_test(self.rand, "anderson-darling"))

    def test_ang(self):
        """
        D'Angosinto
        """
        d_a = stats.normaltest(self.rand, axis=0).statistic
        assert len(d_a) == 512
        assert np.array_equal(d_a, stat_test(self.rand, "d'angostino"))

    def test_jb(self):
        """
        Jarque-Bera
        """
        j_b = np.zeros(512)
        for j in range(512):
            j_b[j] = stats.jarque_bera(self.rand[:, j]).statistic
        assert np.array_equal(j_b, stat_test(self.rand, "jarque-bera"))

    def test_kurtosis(self):
        """
        kurtosis
        """
        kurt = stats.kurtosis(self.rand, axis=0)
        assert len(kurt) == 512
        assert np.array_equal(kurt, stat_test(self.rand, "kurtosis"))

    def test_lilliefors(self):
        """
        Lilliefors
        """
        lillie = np.zeros(512)
        data_0, _ = preprocess(self.rand)
        for j in range(512):
            lillie[j], _ = stats.kstest(data_0[:, j], "norm")
        assert len(data_0[:, j]) == 2000
        assert np.array_equal(lillie, stat_test(self.rand.copy(), "lilliefors"))

    def test_mad(self):
        """
        MAD
        """
        mad = stats.median_abs_deviation(self.rand, axis=0)
        assert len(mad) == 512
        assert np.array_equal(mad, stat_test(self.rand, "mad"))

    def test_mean(self):
        """
        mean
        """
        mean = np.mean(self.rand, axis=0)
        assert len(mean) == 512
        assert np.array_equal(mean, stat_test(self.rand, "mean"))

    def test_midhing(self):
        """
        midhing
        """
        top_quant, bottom_quant = np.quantile(self.rand, [0.75, 0.25], axis=0)
        assert np.all(top_quant > bottom_quant)
        mid = (bottom_quant + top_quant) / 2.0
        assert len(mid) == 512
        assert np.array_equal(mid, stat_test(self.rand, "midhing"))
