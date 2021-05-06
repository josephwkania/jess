#!/usr/bin/env python3
# cython: language_level=3
"""
A somewhat robust way to fit bandpasses

bandpass_fitter() Fits polynomals mutlipe times to
get a robust fit.

bspline_fitter() Fits bsplines using the Huber Regressor
as a loss function
"""

import logging

import numpy as np
from scipy import stats
from scipy.interpolate import splev, splrep
from sklearn.base import TransformerMixin
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline

logger = logging.getLogger()


def poly_fitter(
    bandpass: list, poly_order: int = 20, mask_sigma: float = 6
) -> float:
    """
    Fits bandpasses by polyfitting the bandpass, looking for channels that
    are far from this fit, excluding these channels and refitting the bandpass

    Args:
        bandpass: the bandpass to fit

        polyorder: order of polynomial to fit

        mask_sigma: standard deviation at which to mask outlying channels

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = fit_bspline_huber(bandpass)
    """
    channels = np.arange(0, len(bandpass))
    fit_values = np.polyfit(channels, bandpass, poly_order)  # fit a polynomial
    poly = np.poly1d(fit_values)  # get the values of the fitted bandpass
    diff = bandpass - poly(
        channels
    )  # find the difference between fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale="normal")
    logging.info(f"Standard Deviation of fit: {std_diff:.4}")
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

    logger.info(
        f"chi^2: {stats.chisquare(bandpass, best_fit_bandpass, poly_order)[0]:.4}"
    )
    return best_fit_bandpass


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


def bspline_fitter(bandpass: np.ndarray, chans_per_knot: int = 100) -> np.ndarray:
    """
    This fits a bsplines using the Huber regressor.

    The Huber Regressor is a robust fitting function.

    Inspired by https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4

    Args:
        Bandpass: the bandpass to fit

        chans_per_knot: number of channels per spline knot

    Returns:
        Fit to bandpass

    Example:
        yr = Your(input_file)
        data = yr.get_data(0, 8192)
        bandpass = np.median(section, axis=0)
        fit = bspline_fitter(bandpass)
    """
    num_chans = len(bandpass)
    channels = np.arange(0, num_chans)
    channels = channels[:, np.newaxis]
    num_knots = num_chans // chans_per_knot

    empty_splines = SplineTransformer(num_knots, num_chans)
    model = make_pipeline(empty_splines, HuberRegressor(max_iter=200))
    model.fit(channels, bandpass)

    fit = model.predict(channels)
    return fit
