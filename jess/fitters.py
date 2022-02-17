#!/usr/bin/env python3
# cython: language_level=3
"""
A somewhat robust way to fit bandpasses

cheb_fitter() Fits Chebyshev polynomials, does to fits to be
robust.

bandpass_fitter() Fits polynomials twice to get a robust fit.

bspline_fitt() Fits bsplines using the Huber Regressor
as a loss function

bspline_fitter() Fits bsplines using the Huber Regressor
as a loss function, fits twice to be even more robust
"""

import logging
from typing import Callable

import numpy as np
from scipy import ndimage, sparse, stats
from scipy.interpolate import splev, splrep
from scipy.linalg import cholesky
from sklearn.base import TransformerMixin
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline

logger = logging.getLogger()


def get_fitter(fitter: str) -> Callable:
    """
    Get the fitter object for a given string

    Args:
        fitter: string with the selection of
        bspline_fitter, cheb_fitter, median_fitter, or poly_fitter

    return:
        corresponding fitter object
    """
    fitter = fitter.casefold()
    if fitter == "arpls_fitter":
        return arpls_fitter
    if fitter == "bspline_fitter":
        return bspline_fitter
    if fitter == "cheb_fitter":
        return cheb_fitter
    if fitter == "median_fitter":
        return median_fitter
    if fitter == "poly_fitter":
        return poly_fitter

    raise ValueError(f"You didn't give a valid fitter type! (Given {fitter})")


def arpls_fitter(
    bandpass: np.ndarray,
    lam: float = 1e4,
    ratio: float = 0.05,
    itermax: int = 10,
    dtype: object = np.float32,
) -> np.ndarray:
    """
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Abstract

    Baseline correction methods based on penalized least squares are successfully
    applied to various spectral analyses. The methods change the weights iteratively
    by estimating a baseline. If a signal is below a previously fitted baseline,
    large weight is given. On the other hand, no weight or small weight is given
    when a signal is above a fitted baseline as it could be assumed to be a part
    of the peak. As noise is distributed above the baseline as well as below the
    baseline, however, it is desirable to give the same or similar weights in
    either case. For the purpose, we propose a new weighting scheme based on the
    generalized logistic function. The proposed method estimates the noise level
    iteratively and adjusts the weights correspondingly. According to the
    experimental results with simulated spectra and measured Raman spectra, the
    proposed method outperforms the existing methods for baseline correction and
    peak height estimation.

    This was first used for radio astronomy in
    Radio frequency interference mitigation based on the asymmetrically reweighted
    penalized least squares and SumThreshold method (2021)
    http://zmtt.bao.ac.cn/GPPS/RFI/


    Args:
        Bandpass: the bandpass to fit

        lam: parameter that can be adjusted by user. The larger lambda is,
             the smoother the resulting background, z

        ratio: weighting deviations: 0 < ratio < 1,
               smaller values allow less negative values

        itermax: number of iterations to perform

        dtype: data type to preform the matrix opterations

    Output:
        Fit to bandpass

    """
    input_length = len(bandpass)
    #  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    diff_mtx = sparse.eye(input_length, format="csc", dtype=dtype)
    diff_mtx = (
        diff_mtx[1:] - diff_mtx[:-1]
    )  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    diff_mtx = diff_mtx[1:] - diff_mtx[:-1]

    h_mtx = lam * diff_mtx.T * diff_mtx
    weights = np.ones(input_length, dtype=dtype)
    for j in range(itermax):
        logging.debug("loop #%i", j)
        weights_mtx = sparse.diags(
            weights, 0, shape=(input_length, input_length), dtype=dtype
        )
        wh_mtx = sparse.csc_matrix(weights_mtx + h_mtx)
        # CUDA likes csr here, doesn't seem to matter to scipy
        c_mtx = sparse.csr_matrix(cholesky(wh_mtx.todense()))
        fit = sparse.linalg.spsolve(
            c_mtx, sparse.linalg.spsolve(c_mtx.T, weights * bandpass)
        )
        diff = bandpass - fit
        diff_negative = diff[diff < 0]
        mean = np.mean(diff_negative)
        std = np.std(diff_negative)
        weights_iter = 1.0 / (1 + np.exp(2 * (diff - (2 * std - mean)) / std))
        if np.linalg.norm(weights - weights_iter) / np.linalg.norm(weights) < ratio:
            logging.debug("Reached end condition at iteration #%i", j)
            break
        weights = weights_iter
    return fit


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
    (Not implemented)

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
    # logger.info(
    #     "chi^2: %.4f",
    #     stats.chisquare(
    #         bandpass, best_fit_bandpass, int(3 * chans_per_fit * len(bandpass))
    #     )[0],
    # )
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

    # logger.debug(
    #    "chi^2: %.4f", stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]
    # )
    return best_fit_bandpass


def median_fitter(
    bandpass: np.ndarray,
    chans_per_fit: int = 19,
    interations: int = 1,
) -> np.ndarray:
    """
    Uses a median filter to fit for the bandpass shape

    Args:
        bandpass: ndarray to fit

        chans_per_fit: Number of channels to run the median filter over

        iterations: Number of iterations to run the median filter. Must be
                    greater that one.

    Returns:
        Fit to bandpass

    Notes:
        The idea to run this multiple times is from GSL.
        A recursive median filter might be worth investigating.
        See https://www.gnu.org/software/gsl/doc/html/filter.html

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = median_fitter(bandpass)
    """
    if interations < 1:
        raise ValueError(f"Must have at least one iteration, {interations=}")

    bandpass = bandpass.copy()
    for _ in range(interations):
        ndimage.median_filter(
            bandpass, size=chans_per_fit, mode="mirror", output=bandpass
        )
    return bandpass


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

        channels: list of channel numbers, if None, will create a list starting at zero

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
    diff = bandpass - poly(channels)
    # find the difference between fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale="normal")
    logging.debug("Standard Deviation of fit: %.2f", std_diff)

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

    # logger.debug(
    #    "chi^2: %.4f", stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]
    # )
    return best_fit_bandpass
