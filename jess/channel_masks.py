#!/usr/bin/env python3
"""
Contains Utilities to make channel masks
"""

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jess.calculators import preprocess, shannon_entropy
from jess.fitters import get_fitter

logger = logging.getLogger()


def stat_test(data: np.ndarray, which_test: str) -> np.ndarray:
    """
    Runs the statistical tests
    Should have the same tests as rfi_viewer.py
    """
    which_test = which_test.lower()
    if which_test == "98-2":
        top_quant, bottom_quant = np.quantile(data, [0.98, 0.02], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "91-9":
        top_quant, bottom_quant = np.quantile(data, [0.91, 0.09], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "90-10":
        top_quant, bottom_quant = np.quantile(data, [0.90, 0.01], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "75-25":
        test = stats.iqr(data, axis=0)
    elif which_test == "anderson-darling":
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        # self.hor_test_p = np.zeros(num_samps)
        for ichan in range(0, num_freq):
            test[ichan], _, _ = stats.anderson(data[:, ichan], dist="norm")
    elif which_test == "d'angostino":
        test, _ = stats.normaltest(data, axis=1)
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
        num_freq, _ = data.shape
        test = np.zeros(num_freq)
        data_0, _ = preprocess(data)
        for j in range(0, num_freq):
            test[j], _ = stats.kstest(data_0[j, :], "norm")
    elif which_test == "mad":
        test = stats.median_abs_deviation(data, axis=0)
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
        for k in range(0, num_freq):
            test, _ = stats.shapiro(data[:, k])
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

    return test


def channel_masker(
    dynamic_spectra: np.ndarray,
    test: str,
    sigma: float = 3.0,
    fitter: str = "median_fitter",
    chans_per_fit: int = 47,
    flag_above: bool = True,
    flag_below: bool = True,
    show_plots: bool = False,
) -> List[bool]:
    """
    Reads data from the given file, does the given statistical test.
    Then a curve is fitted to the resulting test-bandpass, this removes large effects
    of the receiver. Z-scores are calculated and channels that decide
    x-score*sigma are flagged.

    The mask of the outlying channels is then saved.

    One should be careful if there are only a few channels, z-score will be
    a bad way to look for outliers.


    args:
        file - dynamic spectra, the 2D chunck of data to process, time is on y-axis

        test - Statistical test you preform on each channel,
               option are from stat_test and are
               Measures of scale: [98-2, 91-9, 90-10, 75-25, mad, stand-dev]
               Guassianity: [anderson-darling, d'angostino, jurue-bera,
                            kurtosis, lilliefors, shapio-wilk, skew]
               Entropy: [shannon-entropy]
               Central Value: [mean, midhing, trimean]

               You can your rfi_viewer.py to see how your data looks at each
               one of these tests

        sigma - This rutinne calculates z values over all channels, this is the
                simga at which to flag channels

        start - Sample at which to start (default: begenning of file)

        nspectra - the number of spectra to process
                   (default: 65536, the default heimdall gulp)

        fitter - the fitter to you to remove bandpass effects (default: "median_fitter")

        chans_per_fit - the number of channls per fitting point, see fitters.py
                        (default: 50)

        flag_upper - flag values above median+sigma*standard dev (default: True)

        flag_lower - flag values below median - sigma*standard dev (default: True)

        show_plot - show the fit and threshold plots, used to check if data is
                    well behaved (default: false)

    returns:
        mask - List of bools, where True is a bad channel

    Example:
        yr_obj = Your('some_data.fil')
        dynamic_spectra = yr_obj.get_data(0, 65536)
        channel_masker(dynamic_spectra, which_test="mean")
    """
    fitter = get_fitter(fitter)

    test_values = stat_test(dynamic_spectra, test)

    fit = fitter(test_values, chans_per_fit=chans_per_fit)

    flat = test_values - fit

    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.title("Fit to test")
        plt.xlabel("Channels")
        plt.ylabel("Test Values")
        plt.plot(flat, label="Flattend")
        plt.plot(test_values, color="r", label="Test Values")
        plt.plot(fit, label="Fit")
        plt.legend()
        plt.show()

    median = np.median(flat)
    if median > 10:
        logging.warning(
            "Median after flatting is %.1f, this is high, check fit", median
        )

    stand_dev = stats.median_abs_deviation(flat, scale="normal")
    if stand_dev < 0.001:
        logging.warning("Standard Dev: %.4f, recalcualting non-robustly", stand_dev)
        stand_dev = np.std(flat)
        logging.warning("Standard Dev: %.4f", stand_dev)

    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.title("Channels to Flag")
        plt.plot(flat)
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
        mask = np.abs(flat - median) > sigma * stand_dev
    elif flag_above:
        mask = flat - median > sigma * stand_dev
    elif flag_below:
        mask = flat - median < sigma * stand_dev
    else:
        raise ValueError("You must flag above or below, you set both to false")

    logging.info("Masking %.2f%% of channels", mask.mean() * 100)
    return mask
