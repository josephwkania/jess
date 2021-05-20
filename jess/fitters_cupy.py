#!/usr/bin/env python3
"""
A somewhat robust way to fit bandpasses - cupy edition
"""

import logging

import cupy as cp
from jess.scipy_cupy.stats import median_abs_deviation_gpu
from scipy import stats

logger = logging.getLogger()


def bandpass_fitter(
    bandpass: list, poly_order: int = 20, mask_sigma: float = 6
) -> float:
    """
    Fits bandpasses by polyfitting the bandpass, looking for channels that
    are far from this fit, excluding these channels and refitting the bandpass

    Args:
        channels: list of channels

        bandpass: the bandpass to fit

        polyorder: order of polynomial to fit

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass
    """
    # xp = cp.get_array_module(bandpass)
    channels = cp.arange(0, len(bandpass))
    fit_values = cp.polyfit(channels, bandpass, poly_order)  # fit a polynomial
    poly = cp.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - poly(channels)
    # find the difference between fitted and real bandpass
    std_diff = median_abs_deviation_gpu(diff, scale="normal")
    logging.info(f"Standard Deviation of fit: {std_diff.get():.4}")

    if std_diff > 0.0:
        # if there is no variability in the channel, don't try to mask
        mask = cp.abs(diff - cp.median(diff)) < mask_sigma * std_diff  # .get()
        # cupy doesn't have ma analoge, above line might
        # cause warning, probably will not have a big effect
        fit_values_clean = cp.polyfit(
            channels[mask], bandpass[mask], poly_order
        )  # refit masking the outliers
        poly_clean = cp.poly1d(fit_values_clean)
        best_fit_bandpass = poly_clean(channels)
    else:
        best_fit_bandpass = poly(channels)
    logger.info(
        f"chi^2: {stats.chisquare(bandpass.get(), best_fit_bandpass.get(), poly_order)[0]:.4}"
    )
    return best_fit_bandpass
