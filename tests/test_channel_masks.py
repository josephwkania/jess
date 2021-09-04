#!/usr/bin/env python3
"""
Tests for channel_masks
"""

import numpy as np
from scipy import stats

from jess.calculators import preprocess, shannon_entropy
from jess.channel_masks import (
    channel_masker,
    dbscan_flagger,
    stat_test,
    z_score_flagger,
)

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
        assert len(self.rand[:, j]) == 2000
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

    def test_shannon_entropy(self):
        """
        Entropy
        """
        entropy = shannon_entropy(self.rand, axis=0)
        assert len(entropy) == 512
        assert np.array_equal(entropy, stat_test(self.rand, "shannon-entropy"))

    def test_shpiro_wilk(self):
        """
        Shapiro-wilk
        """
        s_w = np.zeros(512)
        for j in range(512):
            s_w[j] = stats.shapiro(self.rand[:, j]).statistic

        assert np.array_equal(s_w, stat_test(self.rand, "shapiro-wilk"))

    def test_skew(self):
        """ "
        Skew
        """
        skew = stats.skew(self.rand, axis=0)
        assert len(skew) == 512
        assert np.array_equal(skew, stat_test(self.rand, "skew"))

    def test_stand_dev(self):
        """
        Stand dev
        """
        std = np.std(self.rand, axis=0)
        assert len(std) == 512
        assert np.array_equal(std, stat_test(self.rand, "stand-dev"))

    def test_trimean(self):
        """
        trimean
        """
        top, middle, bottom = np.quantile(self.rand, [0.75, 0.50, 0.25], axis=0)
        # not sure why the below does not work
        # assert np.all(top > middle)
        assert np.all(middle > bottom)
        assert np.all(top > bottom)
        assert len(top) == 512
        tri = (top + 2 * middle + bottom) / 4
        assert np.array_equal(tri, stat_test(self.rand, "trimean"))


def test_dbscan_flagger():
    """
    Make some fake data, add spikes.
    See if spikes get flagged
    """
    rand = np.random.normal(size=512)
    rand[50] += 30
    rand[100] += 60
    rand[300] -= 40

    should_mask = np.zeros(512, dtype=bool)
    should_mask[50] = True
    should_mask[100] = True
    should_mask[300] = True

    mask = dbscan_flagger(rand, eps=0.8)

    assert np.array_equal(should_mask, mask)


def test_z_score_flagger():
    """
    Make some fake data, add spikes
    see if they get removed
    """
    rand = np.random.normal(size=512)
    rand[50] += 30
    rand[100] += 60
    rand[300] -= 40

    should_mask = np.zeros(512, dtype=bool)
    should_mask[50] = True
    should_mask[100] = True
    should_mask[300] = True

    mask = z_score_flagger(rand)

    assert np.array_equal(should_mask, mask)


def test_channel_masker():
    """
    Make a fake array and add spikes in std
    see if they get removed
    """
    rand = np.random.normal(size=512 * 200).reshape(200, 512)
    rand[:, 50] *= 30
    rand[:, 100] *= 60
    rand[:, 300] /= 40
    assert len(rand[:, 300]) == 200

    should_mask = np.zeros(512, dtype=bool)
    should_mask[50] = True
    should_mask[100] = True
    should_mask[300] = True

    # At three sigma some random channels will get flagged
    mask = channel_masker(rand, "stand-dev", sigma=3.5)
    assert np.array_equal(should_mask, mask)
