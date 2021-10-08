#!/usr/bin/env python3
"""
A somewhat robust way to fit bandpasses - cupy edition
"""

import logging

import cupy as cp

from jess.scipy_cupy.stats import median_abs_deviation

# from scipy import stats


logger = logging.getLogger()


def poly_fitter(
    bandpass: cp.ndarray,
    channels: cp.ndarray = None,
    chans_per_fit: int = 200,
    mask_sigma: float = 6,
) -> cp.ndarray:
    """
    Fits bandpasses by polyfitting the bandpass, looking for channels that
    are far from this fit, excluding these channels and refitting the bandpass

    Args:
        bandpass: the bandpass (or any array) that you want to fit a polynomial to.

        channels: list of channel numbers, if None, will create a list starting at zero

        bandpass: the bandpass to fit

        polyorder: order of polynomial to fit

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = cp.asarray(yr.get_data(0, 8192))
        bandpass = cp.median(section, axis=0)
        fit = poly_fitter(bandpass)
        plt.plot(fit.get())
    """
    # xp = cp.get_array_module(bandpass)
    if channels is None:
        channels = cp.arange(0, len(bandpass))
    poly_order = len(bandpass) // chans_per_fit
    logging.debug("Fitting with a %i order polynomial", poly_order)
    if poly_order >= 8:
        raise RuntimeWarning(
            """Cupy polyfit will sometime catastrophically fail above 7th order!
        Consider using the numpy implementation"""
        )

    fit_values = cp.polyfit(channels, bandpass, poly_order)  # fit a polynomial
    poly = cp.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - poly(channels)
    # find the difference between fitted and real bandpass
    std_diff = median_abs_deviation(diff, scale="normal")
    logging.debug("Standard Deviation of fit: %.2f", std_diff.get())

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

    # fails because of new tolerance test
    # logger.debug(
    #     "chi^2: %.2f",
    #     stats.chisquare(bandpass.get(), best_fit_bandpass.get(), poly_order)[0],
    # )
    return best_fit_bandpass
