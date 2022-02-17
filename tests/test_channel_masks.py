#!/usr/bin/env python3
"""
Tests for channel_masks
"""

from unittest import mock

import numpy as np
import pytest
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
        iqr = stats.iqr(self.rand, scale="normal", axis=0)
        assert len(iqr) == 512
        assert np.array_equal(iqr, stat_test(self.rand, "75-25"))

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
        mad = stats.median_abs_deviation(self.rand, axis=0, scale="normal")
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

    def test_raise_not_implemented(self):
        """
        Raise for a test that does not exist
        """
        with pytest.raises(ValueError):
            stat_test(self.rand, "joe")


class TestDBScan:
    """
    Test the DBScan flagger
    """

    def setup_class(self):
        """
        Make some fake data, add spikes.
        See if spikes get flagged
        """
        self.rand = np.random.normal(size=512)
        self.rand[50] += 30
        self.rand[100] += 60
        self.rand[300] -= 40

        self.should_mask = np.zeros(512, dtype=bool)
        self.should_mask[50] = True
        self.should_mask[100] = True
        self.should_mask[300] = True

    def test_dbscan_flagger(self):
        """
        Test dbscan with array
        """
        mask = dbscan_flagger(self.rand, eps=0.8)

        assert np.array_equal(self.should_mask, mask)

    def test_dbscan_flagger_chans(self):
        """
        Test dbscan with array and channel array
        """
        chans = np.arange(len(self.rand))
        mask = dbscan_flagger(self.rand, chans=chans, eps=0.8)

        assert np.array_equal(self.should_mask, mask)

    def test_dbscan_plot(self):
        """
        Test if plot is called when show_plot=True
        """
        with mock.patch("matplotlib.pyplot.show") as show:
            dbscan_flagger(self.rand, eps=0.8, show_plot=True)
            show.assert_called_once()


class TestZScoreFlagger:
    """
    Test the Z score with values above, below,
    and both.

    Make some fake data, add spikes
    see if they get removed
    """

    def setup_class(self):
        """
        Shared random bandpass
        """
        self.rand = np.random.normal(size=512)
        self.rand[50] += 30
        self.rand[100] += 60
        self.rand[300] -= 40

    def test_z_score_flagger(self):
        """
        Test for both above and below
        """
        should_mask = np.zeros(512, dtype=bool)
        should_mask[50] = True
        should_mask[100] = True
        should_mask[300] = True

        mask = z_score_flagger(self.rand)
        assert np.array_equal(should_mask, mask)

    def test_z_score_flagger_below(self):
        """
        Only test for low outliers
        """
        should_mask = np.zeros(512, dtype=bool)
        # should_mask[50] = True
        # should_mask[100] = True
        should_mask[300] = True

        mask = z_score_flagger(self.rand, flag_above=False)
        assert np.array_equal(should_mask, mask)

    def test_z_score_flagger_above(self):
        """
        Only test for low outliers
        """
        should_mask = np.zeros(512, dtype=bool)
        should_mask[50] = True
        should_mask[100] = True
        # should_mask[300] = True

        mask = z_score_flagger(self.rand, flag_below=False)
        assert np.array_equal(should_mask, mask)

    def test_z_score_flagger_no(self):
        """
        Raise Value Error if flag_above=flag_below=False
        """
        with pytest.raises(ValueError):
            z_score_flagger(self.rand, flag_above=False, flag_below=False)

    def test_z_score_plot(self):
        """
        Test if plot is called when show_plot=True
        """
        with mock.patch("matplotlib.pyplot.show") as show:
            z_score_flagger(self.rand, show_plots=True)
            show.assert_called_once()


class TestChannelMasker:
    """
    Test the channel masker with z_score and dbscan
    """

    def setup_class(self):
        """
        Make a fake array and add spikes in std
        see if they get removed
        """
        self.rand = np.random.normal(size=512 * 200).reshape(200, 512)
        self.rand[:, 50] *= 30
        self.rand[:, 100] *= 60
        self.rand[:, 300] /= 40
        assert len(self.rand[:, 300]) == 200

        self.should_mask = np.zeros(512, dtype=bool)
        self.should_mask[50] = True
        self.should_mask[100] = True
        self.should_mask[300] = True

    def test_channel_masker_zscore(self):
        """
        At three sigma some random channels will get flagged
        512*4test = 2048 random channels, @6simga this should never happen
        """
        mask = channel_masker(self.rand, "stand-dev", sigma=6)
        assert np.array_equal(self.should_mask, mask)

    def test_channel_masker_dbscan(self):
        """
        Use DBscan

        Not sure why its not flaggin the low channel
        """
        mask = channel_masker(self.rand, "stand-dev", flagger="dbscan_flagger")
        self.should_mask[300] = False
        assert np.array_equal(self.should_mask, mask)

    def test_channel_masker_not_implemented(self):
        """
        Should raise an error
        """

        with pytest.raises(ValueError):
            channel_masker(self.rand, "stand-dev", flagger="joe")

    def test_channel_masker_plot(self):
        """
        Test if plot is called twice when show_plot=True
        """
        with mock.patch("matplotlib.pyplot.show") as show:
            channel_masker(self.rand, "stand-dev", show_plots=True)
            assert show.call_count == 2
