#!/usr/bin/env python3
"""
A somewhat robust way to fit bandpasses
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger()


def bandpass_fitter(
    bandpass: float, poly_order: int = 20, mask_sigma: float = 6
) -> float:
    """
    Fits bandpasses by polyfitting the bandpass, looking for channels that
    are far from this fit, exluding these channels and refitting the bandpass

    Args:
        bandpass: the bandpass to fit

        polyorder: order of polynomial to fit

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass
    """
    channels = np.arange(0, len(bandpass))
    fit_values = np.polyfit(channels, bandpass, poly_order)  # fit a polynomial
    poly = np.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - poly(
        channels
    )  # find the differnece betweeen fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale="normal")
    logging.info(f"Standard Deviation of fit: {std_diff:.4}")
    mask = np.abs(diff - np.median(diff)) < mask_sigma * std_diff

    fit_values_clean = np.polyfit(
        channels[mask], bandpass[mask], poly_order
    )  # refit masking the outliers
    poly_clean = np.poly1d(fit_values_clean)
    best_fit_bandpass = poly_clean(channels)
    logger.info(
        f"chi^2: {stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]:.4}"
    )
    return best_fit_bandpass
