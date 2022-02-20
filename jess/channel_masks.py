#!/usr/bin/env python3
"""
Contains Utilities to make channel masks
"""

import logging
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from jess.calculators import preprocess, shannon_entropy
from jess.fitters import get_fitter

# Should change the masks to npt.NDArray[bool] in
# the future
# import numpy.typing as npt

logger = logging.getLogger()


def stat_test(data: np.ndarray, which_test: str) -> np.ndarray:
    """
    Runs the statistical tests
    Should have the same tests as rfi_viewer.py
    Test:
        Measures of scale:  98-2, 91-9, 90-10, 75-25, mad, stand-dev
        Gaussianity: anderson-darling, d'angostino, jarque-bera,
                     lilliefors, kurtosis, shapiro-wilk, skew
        Central Tendency: mean, midhing, trimean
        Information: shannopn-entropy

    MAD and IQR report the scaled version, to make comparison tests easier
    """
    which_test = which_test.lower()
    if which_test == "98-2":
        top_quant, bottom_quant = np.quantile(data, [0.98, 0.02], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "91-9":
        top_quant, bottom_quant = np.quantile(data, [0.91, 0.09], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "90-10":
        top_quant, bottom_quant = np.quantile(data, [0.90, 0.10], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "75-25":
        test = stats.iqr(data, axis=0, scale="normal")
    elif which_test == "anderson-darling":
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        # self.hor_test_p = np.zeros(num_samps)
        for ichan in range(0, num_freq):
            test[ichan], _, _ = stats.anderson(data[:, ichan], dist="norm")
    elif which_test == "d'angostino":
        test, _ = stats.normaltest(data, axis=0)
    elif which_test == "jarque-bera":
        _, num_freq = data.shape
        if num_freq < 2000:
            logging.warning(
                "Jarque-Bera requires > 2000 points, given %i channels", num_freq
            )
        test = np.zeros(num_freq)
        for ichan in range(0, num_freq):
            test[ichan], _ = stats.jarque_bera(
                data[:, ichan],
            )
    elif which_test == "kurtosis":
        test = stats.kurtosis(data, axis=0)
    elif which_test == "lilliefors":
        # I don't take into account the change of dof when calculating the p_value
        # The test stattic is the same as statsmodels lilliefors
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        data_0, _ = preprocess(data)
        for ichan in range(0, num_freq):
            test[ichan], _ = stats.kstest(data_0[:, ichan], "norm")
    elif which_test == "mad":
        test = stats.median_abs_deviation(data, axis=0, scale="normal")
    elif which_test == "mean":
        test = np.mean(data, axis=0)
    elif which_test == "midhing":
        top_quant, bottom_quant = np.quantile(data, [0.75, 0.25], axis=0)
        test = (bottom_quant + top_quant) / 2.0
    elif which_test == "shannon-entropy":
        test = shannon_entropy(data, axis=0)
    elif which_test == "shapiro-wilk":
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        # test_p = np.zeros(num_freq)
        for ichan in range(0, num_freq):
            test[ichan], _ = stats.shapiro(data[:, ichan])
    elif which_test == "skew":
        test = stats.skew(data, axis=0)
    elif which_test == "stand-dev":
        test = np.std(data, axis=0)
    elif which_test == "trimean":
        top_quant, middle_quant, bottom_quant = np.quantile(
            data, [0.75, 0.50, 0.25], axis=0
        )
        test = (top_quant + 2.0 * middle_quant + bottom_quant) / 4.0
    else:
        raise ValueError(f"You gave {which_test}, which is not avaliable.")

    assert len(test) == data.shape[1], "test fail to return a value for each channel"
    return test


def z_score_flagger(
    flat_bandpass: np.ndarray,
    flag_above: bool = True,
    flag_below: bool = True,
    sigma: float = 6.0,
    measure_of_scale: Callable = partial(stats.median_abs_deviation, scale="normal"),
    show_plots: bool = False,
) -> np.ndarray:
    """
    Flags points based on z-score

    args:
        flat_banpass - Results of some statistical test with the banpass effects removed

        flag_above - Flag values with a z-scores above the threshold

        flag_below - Flag values with a z-score below the threshold

        sigma - the standard deviation to flag points

        show_plots - Show diagnostic plots

    returns:
        Bool mask where True = bad data

    Example:
        yr = Your("some.fil")
        dynamic_spectra = yr.get_data(7000, 2 ** 17)
        test_values = stat_test(dynamic_spectra, test)
        fit = fitter(test_values, chans_per_fit=chans_per_fit)
        flat = test_values - fit
        mask = z_score_flagger(flat)
    """
    median = np.median(flat_bandpass)
    if median > 10:
        logging.warning(
            "Median after flatting is %.1f, this is high, check fit", median
        )

    stand_dev = measure_of_scale(flat_bandpass)
    if stand_dev < 0.001:
        logging.warning("Standard Dev: %.4f,", stand_dev)

    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.title("Channels to Flag")
        plt.plot(flat_bandpass)
        plt.axhline(median + sigma * stand_dev, color="r", label="Top Threshold")
        plt.axhline(median - sigma * stand_dev, color="r", label="Bottom Threshold")
        plt.xlabel("Channels")
        plt.ylabel("Test values")
        plt.legend()
        plt.show()

    logging.debug(
        "After Flattening - Median: %.3f, Standard Dev: %.3f", median, stand_dev
    )

    # flag based on z value
    if flag_above and flag_below:
        mask = np.abs(flat_bandpass - median) > sigma * stand_dev
    elif flag_above and not flag_below:
        mask = flat_bandpass - median > sigma * stand_dev
    elif flag_below and not flag_above:
        mask = flat_bandpass - median < -sigma * stand_dev
    else:
        raise ValueError("You must flag above or below, you set both to false")

    return mask


def dbscan_flagger(
    test_values: np.ndarray,
    chans: np.ndarray = None,
    eps: float = 0.3,
    min_clust_frac: float = 0.14,
    show_plot: bool = False,
) -> np.ndarray:
    """
    Use DBScan to look for outliers

    args:
        test_values: Values from some test for each channel,
                     expects bandpass effects to be subtracted off

        chans: list of chan numbers

        eps: DBSCAN eps

        min_clust_fraction: minimum fraction of channels for a DBSCAN cluster

        show_plot: show diagnostic plot of cluster

    return:
        Bool mask, True=Bad channel

    Example:
        yr = Your("some.fil")
        dynamic_spectra = yr.get_data(7000, 2 ** 17)
        test_values = stat_test(dynamic_spectra, test)
        fit = fitter(test_values, chans_per_fit=chans_per_fit)
        flat = test_values - fit
        mask = dbscan_flagger(flat)
    """
    num_data = len(test_values)
    if chans is None:
        chans = np.array(range(0, num_data), dtype=int)
    else:
        l_t_v = len(test_values)
        l_c = len(chans)
        assert l_t_v == l_c, f"len(test_values)={l_t_v}!=len(chans)={l_c}"
    positions_vec = [[chan, value] for chan, value in zip(chans, test_values)]

    scaler = StandardScaler()
    normed_vec = scaler.fit_transform(positions_vec)
    # Preprocess the data, keep scaler so we can undo this transform later

    min_samples = int(min_clust_frac * num_data)
    logging.debug("Running DBSCAN with eps=%.4f, min_samples=%i", eps, min_samples)
    db_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(normed_vec)
    core_samples_mask = np.zeros_like(db_obj.labels_, dtype=bool)
    core_samples_mask[db_obj.core_sample_indices_] = True
    labels = db_obj.labels_

    unique_labels = set(labels)
    num_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    num_noise_ = list(labels).count(-1)

    logging.debug(
        "DBSCAN found %i clusters and %i noise points", num_clusters_, num_noise_
    )

    assert (
        0 < num_clusters_ < 5
    ), f"Number of clusters is {num_clusters_}, expecting a value [1, 4]"
    assert (
        num_noise_ < 0.4 * num_data
    ), "More that 40% the points are classified as noise, clustering failed"

    normed_vec = scaler.inverse_transform(normed_vec)
    # transform back so we can see the data again
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.figure(figsize=(20, 10))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        x_y = normed_vec[class_member_mask & core_samples_mask]
        plt.plot(
            x_y[:, 0],
            x_y[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markersize=7,
            label="Clustered",
        )

        x_y = normed_vec[class_member_mask & ~core_samples_mask]
        plt.plot(
            x_y[:, 0],
            x_y[:, 1],
            "+",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
            label="noise",
        )

    chans_to_mask = np.array(x_y[:, 0], dtype=int)
    mask = np.zeros_like(test_values, dtype=bool)
    mask[chans_to_mask] = True
    if show_plot:
        plt.title(f"Clusters: {num_clusters_}, Noise: {num_noise_}")
        plt.xlabel("Channel #")
        plt.ylabel("Test Value")
        plt.legend()
        plt.show()
    return mask


def channel_masker(
    dynamic_spectra: np.ndarray,
    test: str,
    sigma: float = 4.0,
    fitter: str = "median_fitter",
    chans_per_fit: int = 30,
    flagger: str = "z_score_flagger",
    flag_above: bool = True,
    flag_below: bool = True,
    eps: float = 0.9,
    min_clust_frac: float = 0.01,
    show_plots: bool = False,
) -> np.ndarray:
    """
    Does the given statistical test on a given data array.
    Then a curve is fitted to the resulting test-bandpass, this removes large effects
    of the receiver. Z-scores are calculated and channels that decide
    x-score*sigma are flagged.

    The mask of the outlying channels is then saved.

    One should be careful if there are only a few channels, z-score will be
    a bad way to look for outliers.


    args:
        file - dynamic spectra, the 2D chunk of data to process, time is on y-axis

        test - Statistical test you preform on each channel,
                option are from stat_test and are
                Measures of scale: [98-2, 91-9, 90-10, 75-25, mad, stand-dev]
                Guassianity: [anderson-darling, d'angostino, jarque-bera,
                             kurtosis, lilliefors, shapio-wilk, skew]
                Entropy: [shannon-entropy]
                Central Value: [mean, midhing, trimean]

                If you give a list of two tests, you will get the first one
                subtracted from the second. For example (stand-dev, mad)
                will be a test the difference between the two.

                You can your rfi_viewer.py to see how your data looks at each
                one of these tests

        sigma - This routine calculates z values over all channels, this is the
                sigma at which to flag channels

        start - Sample at which to start (default: beginning of file)

        nspectra - the number of spectra to process
                   (default: 65536, the default heimdall gulp)

        fitter - Fitter used to fit bandpass effects

        chans_per_fit - the number of channels per fitting point, see fitters.py
                        If zero, don't do bandpass subtraction.
                        If two tests are given, smooth the second test. (default: 30)

        flagger: The flagger to remove outlying points,
                 [z_score_flagger, dbscan_flagger]

        flag_upper - flag values above median+sigma*standard dev (Only z-score)

        flag_lower - flag values below median - sigma*standard dev (Only z-score)

        eps - dbscan eps (dbscan only)

        min_clust_frac - fraction of channels for the minimum cluster size (dbscan only)

        show_plot - show the fit and threshold plots, used to check if data is
                    well behaved (default: false)

    returns:
        mask - ndarray[bools], where `True` is a bad channel

    Example:
        yr_obj = Your('some_data.fil')
        dynamic_spectra = yr_obj.get_data(0, 65536)
        mask = channel_masker(dynamic_spectra, which_test="mean")
    """

    test = np.array(test, ndmin=1)
    test_values = stat_test(dynamic_spectra, test[0])
    # Save if we want to plot
    test_values_original = test_values.copy()

    fit = None
    if len(test) == 2 and chans_per_fit > 0:
        base_values = stat_test(dynamic_spectra, test[1])
        fitter_func = get_fitter(fitter)
        fit = fitter_func(base_values, chans_per_fit=chans_per_fit)
        test_values -= fit
    elif len(test) == 2:
        base_values = stat_test(dynamic_spectra, test[1])
        test_values -= base_values
    elif chans_per_fit > 0:
        fitter_func = get_fitter(fitter)
        fit = fitter_func(test_values, chans_per_fit=chans_per_fit)
        test_values -= fit

    if flagger == "z_score_flagger":
        logging.debug("Using z_score_flagger")
        mask = z_score_flagger(
            test_values,
            flag_above=flag_above,
            flag_below=flag_below,
            sigma=sigma,
            show_plots=show_plots,
        )
    elif flagger == "dbscan_flagger":
        logging.debug("Using dbscan_flagger")
        mask = dbscan_flagger(test_values, eps=eps, min_clust_frac=min_clust_frac)
    else:
        raise ValueError(f"You asked for {flagger}, which not available")

    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.title("Fit to test")
        plt.xlabel("Channels")
        plt.ylabel("Test Values")
        plt.plot(test_values, label="Flattend")
        plt.plot(test_values_original, color="r", label="Test Values")
        if fit is not None:
            plt.plot(fit, label="Fit")
        plt.legend()
        plt.show()

    logging.info("Masking %.2f%% of channels", mask.mean() * 100)
    return mask
