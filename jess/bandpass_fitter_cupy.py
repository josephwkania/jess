#!/usr/bin/env python3

import logging

import cupy as cp
import numpy as np
from scipy import stats

from jess.scipy_cupy.stats import median_abs_deviation_gpu

logger = logging.getLogger()


def bandpass_fitter(
    channels: int, bandpass: float, poly_order: int = 20, mask_sigma: float = 6
) -> float:
    # xp = cp.get_array_module(bandpass)
    channels = np.arange(0, len(bandpass))
    fit_values = np.polyfit(channels, bandpass.get(), poly_order)  # fit a polynomial
    poly = np.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - cp.array(
        poly(channels)
    )  # find the differnece betweeen fitted and real bandpass
    std_diff = median_abs_deviation_gpu(diff, scale="normal")
    logging.info(f"Standard Deviation of fit: {std_diff.get():.4}")
    bandpass = bandpass.get()
    if std_diff > 0.0:  # if there is no variability in the channel, don't try to mask
        mask = (cp.abs(diff - cp.median(diff)) < mask_sigma * std_diff).get()

        fit_values_clean = np.polyfit(
            channels[mask], bandpass[mask], poly_order
        )  # refit masking the outliers
        poly_clean = np.poly1d(fit_values_clean)
        best_fit_bandpass = poly_clean(channels)
    else:
        best_fit_bandpass = poly(channels)
    logger.info(
        f"chi^2: {stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]:.4}"
    )
    return best_fit_bandpass
