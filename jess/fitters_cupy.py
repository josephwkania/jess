#!/usr/bin/env python3
"""
A somewhat robust way to fit bandpasses - cupy edition
"""

import logging

import cupy as cp
from cupy.linalg import cholesky
from cupyx.scipy import ndimage, sparse
from cupyx.scipy.sparse.linalg import spsolve

from jess.scipy_cupy.stats import median_abs_deviation

# from scipy import stats


logger = logging.getLogger()


def arpls_fitter(
    bandpass: cp.ndarray,
    lam: float = 3e4,
    ratio: float = 0.05,
    itermax: int = 10,
    dtype: object = cp.float32,
) -> cp.ndarray:
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

    Note:
        The Cholesky decomosition on cupy (9.5.0) is slightly off (at e-9 level)
        this causes the changes amount of smoothness. Upping lam to
        3e4 makes the outputs the same.
    """
    input_length = len(bandpass)
    #  diff_mtx = sparse.csc_matrix(np.diff(np.eye(N), 2))
    diff_mtx = sparse.eye(input_length, format="csr", dtype=dtype)
    diff_mtx = (
        diff_mtx[1:] - diff_mtx[:-1]
    )  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    diff_mtx = diff_mtx[1:] - diff_mtx[:-1]

    h_mtx = lam * diff_mtx.T * diff_mtx
    weights = cp.ones(input_length, dtype=dtype)
    for j in range(itermax):
        logging.debug("loop #%i", j)
        weights_mtx = sparse.diags(
            weights, 0, shape=(input_length, input_length), dtype=dtype
        )
        wh_mtx = sparse.csc_matrix(weights_mtx + h_mtx)
        # C = sparse_cp.csc_matrix(cp.asarray(cholesky(WH.todense().get())))
        # CUDA likes csr here
        c_mtx = sparse.csc_matrix(cholesky(wh_mtx.todense()))
        fit = spsolve(c_mtx, spsolve(c_mtx.T, weights * bandpass))
        diff = bandpass - fit
        diff_negative = diff[diff < 0]
        mean = cp.mean(diff_negative)
        std = cp.std(diff_negative)
        weights_iter = 1.0 / (1 + cp.exp(2 * (diff - (2 * std - mean)) / std))
        if cp.linalg.norm(weights - weights_iter) / cp.linalg.norm(weights) < ratio:
            logging.debug("Reached end condition at iteration #%i", j)
            break
        weights = weights_iter
    return fit


def median_fitter(
    bandpass: cp.ndarray,
    chans_per_fit: int = 19,
    interations: int = 1,
) -> cp.ndarray:
    """
    Uses a median filter to fit for the bandpass shape.
    Note: does inplace

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

    # bandpass = bandpass.copy()
    for _ in range(interations):
        ndimage.median_filter(
            bandpass, size=chans_per_fit, mode="mirror", output=bandpass
        )
    return bandpass


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
