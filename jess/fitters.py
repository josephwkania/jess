#!/usr/bin/env python3
# cython: language_level=3
"""
A somewhat robust way to fit bandpasses

cheb_fitter() Fits Chebyshev polynomials, does to fits to be
robust.

bandpass_fitter() Fits polynomals twice to get a robust fit.

bspline_fitt() Fits bsplines using the Huber Regressor
as a loss function

bspline_fitter() Fits bsplines using the Huber Regressor
as a loss function, fits twice to be even more robust
"""

import logging

import numpy as np
from scipy import signal, stats
from scipy.interpolate import splev, splrep
from sklearn.base import TransformerMixin
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline

logger = logging.getLogger()


def get_fitter(fitter: str) -> object:
    """
    Get the fitter object for a given string

    Args:
        fitter: string with the selection of
        bspline_fitter, cheb_fitter, median_fitter, or poly_fitter

    return:
        corresponding fitter object
    """
    if fitter == "bspline_fitter":
        return bspline_fitter
    if fitter == "cheb_fitter":
        return cheb_fitter
    if fitter == "median_fitter":
        return median_fitter
    if fitter == "poly_fitter":
        return poly_fitter

    raise ValueError(f"You didn't give a valid fitter type! (Given {fitter})")


class SplineTransformer(TransformerMixin):
    """
    The bspline transformer class
    """

    def __init__(self, knots, degree=3, periodic=False):
        self.empty_splines = self.get_empty_bsplines(knots, degree, periodic=periodic)
        self.num_splines = len(self.empty_splines)

    @staticmethod
    def get_empty_bsplines(
        num_knots: int, num_chans: int, degree: int = 3, periodic: bool = False
    ) -> list:
        """
        Creates the empty bsplines that will be
        used then fit by the Huber Regressor
        """
        knots = np.linspace(0, num_chans, num_knots)
        # equally spaced knots

        zeros = np.zeros(num_knots)
        # this is a fake y to fit too

        knots, coeffs, degree = splrep(knots, zeros, k=degree, per=periodic)
        # pylint gives a 'unbalanced-tuple-unpacking' warning at the above
        # This seems erroneous

        empty_splines = []

        for sub_spline in range(num_knots):
            coeff = [
                1.0 if coeff_num == sub_spline else 0.0
                for coeff_num in range(len(coeffs))
            ]
            empty_splines.append((knots, coeff, degree))
        return empty_splines

    def fit(self, x, y=None):
        """
        This gives sklearn a fit function
        """
        return self

    def transform(self, x):
        """
        Transform the bsplines
        """
        num_samples, num_features = x.shape
        features = np.zeros((num_samples, num_features * self.num_splines))
        for i, spline in enumerate(self.empty_splines):
            i_start = i * num_features
            i_end = (i + 1) * num_features
            features[:, i_start:i_end] = splev(x, spline)
        return features


def bspline_fit(
    bandpass: np.ndarray,
    channels: np.ndarray = None,
    chans_per_knot: int = 100,
    max_inter=250,
) -> np.ndarray:
    """
    This fits a bsplines using the Huber regressor.

    The Huber Regressor is a robust fitting function.

    Inspired by https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4

    Args:
        Bandpass: the bandpass to fit

        chans_per_knot: number of channels per spline knot

        max_iter: The maximum number of iterations

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = bspline_fit(bandpass)
    """

    num_chans = len(bandpass)
    if channels is None:
        channels = np.arange(0, num_chans)

    channels = channels[:, np.newaxis]
    num_knots = num_chans // chans_per_knot

    empty_splines = SplineTransformer(num_knots, num_chans)
    model = make_pipeline(empty_splines, HuberRegressor(max_iter=max_inter))
    model.fit(channels, bandpass)

    fit = model.predict(channels)
    return fit


def bspline_fitter(
    bandpass: np.ndarray,
    channels: np.ndarray = None,
    chans_per_fit: int = 100,
    # mask_sigma: float = 6,
) -> np.ndarray:
    """
    This fits a bsplines using the Huber regressor.

    The Huber Regressor is a robust fitting function.

    This wraps bspline_fit, # running it twice to help it futher reject outliers
    (Not implemnted)

    Args:
        Bandpass: the bandpass to fit

        chans_per_fit: number of channels per spline knot

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = bspline_fitter(bandpass)

    Notes:
        I've attempted to make this even more robbust by running it once, flagging data
        and running it again. However I can't get model.predict() to work on the full
        channel set when it is trained with flagged channels.
    """
    if channels is None:
        channels = np.arange(0, len(bandpass))
    first_bspline_fit = bspline_fit(bandpass, channels, chans_per_knot=chans_per_fit)
    # fit the first spline
    # diff = bandpass - first_bspline_fit
    # find the difference between fitted and real bandpass
    # std_diff = stats.median_abs_deviation(diff, scale="normal")
    # logging.info("Standard Deviation of fit: %.4f", std_diff)
    # if std_diff > 0.0:
    #     # if there is no variability in the channel, don't try to mask
    #     mask = np.abs(diff - np.ma.median(diff)) < mask_sigma * std_diff

    #     second_bspline_fit = bspline_fit(bandpass[mask],
    #                                      channels[mask],
    #                                      chans_per_fit)
    #     # refit masking the outliers
    #     best_fit_bandpass = second_bspline_fit
    # else:
    #     best_fit_bandpass = first_bspline_fit
    best_fit_bandpass = first_bspline_fit
    logger.info(
        "chi^2: %.4f",
        stats.chisquare(
            bandpass, best_fit_bandpass, int(3 * chans_per_fit * len(bandpass))
        )[0],
    )
    return best_fit_bandpass


def cheb_fitter(
    bandpass: np.ndarray,
    channels: np.ndarray = None,
    chans_per_fit: int = 100,
    mask_sigma: float = 6,
) -> np.ndarray:
    """
    Fits bandpasses by Chebyshev fitting the bandpass, looking for channels that
    are far from this fit, excluding these channels and refitting the bandpass
    This works well for bandpasses with sine/cosine like features.

    Args:
        bandpass: the bandpass to fit

        chans_per_fit: number of channels for each polynomial order

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = cheb_fitter(bandpass)
    """
    if channels is None:
        channels = np.arange(0, len(bandpass))
    poly_order = len(bandpass) // chans_per_fit
    logging.debug("Fitting with a %i polynomial", poly_order)
    fit_values = np.polynomial.chebyshev.Chebyshev.fit(channels, bandpass, poly_order)
    # fit a polynomial
    diff = bandpass - fit_values(
        channels
    )  # find the difference between fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale="normal")
    logging.debug("Standard Deviation of fit %.4f: ", std_diff)
    if std_diff > 0.0:
        # if there is no variability in the channel, don't try to mask
        mask = np.abs(diff - np.ma.median(diff)) < mask_sigma * std_diff

        fit_values_clean = np.polynomial.chebyshev.Chebyshev.fit(
            channels[mask], bandpass[mask], poly_order
        )  # refit masking the outlier
        best_fit_bandpass = fit_values_clean(channels)
    else:
        best_fit_bandpass = fit_values(channels)

    logger.debug(
        "chi^2: %.4f", stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]
    )
    return best_fit_bandpass


def median_fitter(
    bandpass: np.ndarray,
    chans_per_fit: int = 19,
) -> np.ndarray:
    """
    Uses a median filter to fit for the bandpass shape

    Args:
        bandpass: the bandpass to fit

        chans_per_fit: humber of channels to run the median filter over

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = median_fitter(bandpass)
    """
    return signal.medfilt(bandpass, kernel_size=chans_per_fit)


def poly_fitter(
    bandpass: np.ndarray,
    channels: np.ndarray = None,
    chans_per_fit: int = 200,
    mask_sigma: float = 6,
) -> np.ndarray:
    """
    Fits bandpasses by polyfitting the bandpass, looking for channels that
    are far from this fit, excluding these channels and refitting the bandpass

    Args:
        bandpass: the bandpass to fit

        chans_per_fit: Number of channels per polynomial

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = poly_fitter(bandpass)
    """
    if channels is None:
        channels = np.arange(0, len(bandpass))
    poly_order = len(bandpass) // chans_per_fit
    logging.debug("Fitting with a %i polynomial", poly_order)
    fit_values = np.polyfit(channels, bandpass, poly_order)  # fit a polynomial
    poly = np.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - poly(
        channels
    )  # find the difference between fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale="normal")
    logging.debug("Standard Deviation of fit: %.4f", std_diff)
    if std_diff > 0.0:
        # if there is no variability in the channel, don't try to mask
        mask = np.abs(diff - np.ma.median(diff)) < mask_sigma * std_diff

        fit_values_clean = np.polyfit(
            channels[mask], bandpass[mask], poly_order
        )  # refit masking the outliers
        poly_clean = np.poly1d(fit_values_clean)
        best_fit_bandpass = poly_clean(channels)
    else:
        best_fit_bandpass = poly(channels)

    logger.debug(
        "chi^2: %.4f", stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]
    )
    return best_fit_bandpass
