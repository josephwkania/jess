#!/usr/bin/env python3
"""
The repository for all my filters
"""
import logging

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter as sg

from jess.fitters import poly_fitter


def anderson_darling_time(
    gulp: np.ndarray,
    critical_cut: float = 1e-22,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    UNDER CONSTRUCTION !!!
    Calculates the Anderson Darling test along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

        p_cut: blocks with a pvalue below this number get cut

        frame: number of time samples to calculate the kurtosis

        rerturn_values: return the test values

    Returns:

       Mask based on Critical value cut

       optional: return the test values for each block
    """
    frame = int(frame)
    test_values = np.zeros_like(gulp, dtype=np.float)
    p_values = np.zeros_like(gulp, dtype=np.float)
    mask = np.full_like(gulp, True, dtype=bool)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        for k in range(0, gulp.shape[1]):
            # Loop over the channels
            test_vec, p_vec = stats.kstest(
                (gulp[j : j + frame, k] - np.median(gulp[j : j + frame, k]))
                / np.std(gulp[j : j + frame, k]),
                "norm",
            )
            test_values[j : j + frame, k] = test_vec
            p_values[j : j + frame, k] = p_vec

    mask = p_values < critical_cut

    if return_values:
        return mask, p_values

    return mask


def dagostino_time(
    gulp: np.ndarray,
    p_cut: float = 0.001,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    Calculates the D’Agostino test along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

        p_cut: blocks with a pvalue below this number get cut

        frame: number of time samples to calculate the kurtosis

        rerturn_values: return the test values

    Returns:

       Mask based on bad iqr sections

       optional: return the test values for each block
    """
    frame = int(frame)
    test_values = np.zeros_like(gulp, dtype=np.float)
    p_values = np.zeros_like(gulp, dtype=np.float)
    mask = np.full_like(gulp, True, dtype=bool)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        test_vec, p_vec = stats.normaltest(gulp[j : j + frame], axis=0)
        test_values[j : j + frame, :] = test_vec
        p_values[j : j + frame, :] = p_vec

    mask = p_values < p_cut

    if return_values:
        return mask, p_values

    return mask


def iqr_time(
    gulp: np.ndarray, sigma: float = 6, frame: int = 128, return_values: bool = False
) -> np.ndarray:
    """
    Calculates the spectral Kurtosis along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

    `   sigma on to cut kurtosis values

        frame: number of time samples to calculate the kurtosis

        apply_mask: Apply the mask to the data, replacing bad values with zeros

    Returns:

       Mask based on bad iqr sections

       optional: apply mask as replace with zeros
    """
    frame = int(frame)
    iqr_values = np.zeros_like(gulp, dtype=np.float)
    mask = np.full_like(gulp, True, dtype=bool)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        iqr_vec = stats.iqr(gulp[j : j + frame], axis=0)
        iqr_values[j : j + frame, :] = iqr_vec

    # iqr_bandpass =
    iqr_fit = poly_fitter(iqr_values.mean(axis=0))
    iqr_flat = iqr_values - iqr_fit
    stds_iqr = stats.median_abs_deviation(iqr_flat, scale="normal")
    meds_iqr = np.median(iqr_flat)

    mask = iqr_flat - meds_iqr < sigma * stds_iqr

    if return_values:
        return mask, iqr_values

    return mask


def ks_time(
    gulp: np.ndarray,
    p_cut: float = 1e-22,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    UNDER CONSTRUCTION !!!
    Calculates the KS test along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

        p_cut: blocks with a pvalue below this number get cut

        frame: number of time samples to calculate the kurtosis

        rerturn_values: return the test values

    Returns:

       Mask based on p value cut

       optional: return the test values for each block
    """
    frame = int(frame)
    test_values = np.zeros_like(gulp, dtype=np.float)
    p_values = np.zeros_like(gulp, dtype=np.float)
    mask = np.full_like(gulp, True, dtype=bool)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        for k in range(0, gulp.shape[1]):
            # Loop over the channels
            test_vec, p_vec = stats.kstest(
                (gulp[j : j + frame, k] - np.median(gulp[j : j + frame, k]))
                / np.std(gulp[j : j + frame, k]),
                "norm",
            )
            test_values[j : j + frame, k] = test_vec
            p_values[j : j + frame, k] = p_vec

    mask = p_values < p_cut

    if return_values:
        return mask, p_values

    return mask


def kurtosis_time_thresh(
    gulp: np.ndarray,
    threshhold: float = 20,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    Calculates the spectral Kurtosis along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

    `   threshhold: abs threashold to filter kurtosis values

        frame: number of time samples to calculate the kurtosis

        return_mask: bool
    Returns:

       Dynamic Spectrum with bad Kurtosis values removed

       optional: masked values
    """
    frame = int(frame)
    kvalues = np.zeros_like(gulp)
    mask = np.zeros_like(gulp)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        kurtosis_vec = stats.kurtosis(gulp[j : j + frame], axis=0)
        kvalues[j : j + frame, :] = kurtosis_vec

    mask = kvalues < threshhold

    if return_values:
        return mask, kvalues

    return mask


def kurtosis_time(
    gulp: np.ndarray,
    p_cut: float = 0.001,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    Calculates the spectral Kurtosis along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

        p_cut: blocks with a pvalue below this number get cut

        frame: number of time samples to calculate the kurtosis

        return_mask: bool
    Returns:

       mask

       optional: pvalues for each of the blocks
    """
    frame = int(frame)
    kvalues = np.zeros_like(gulp, dtype=np.float)
    mask = np.zeros_like(gulp, dtype=bool)
    p_values = np.zeros_like(gulp, dtype=np.float)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        kurtosis_vec, p_vec = stats.kurtosistest(gulp[j : j + frame], axis=0)
        kvalues[j : j + frame, :] = kurtosis_vec
        p_values[j : j + frame, :] = p_vec

    mask = p_values < p_cut

    # stds_kurt = np.std(kvalues, axis=0)
    # meds_kurt = np.median(kvalues, axis=0)

    # mask = np.abs(kvalues - meds_kurt) < sigma * stds_kurt

    if return_values:
        return mask, p_values

    return mask


def mad_spectra(
    gulp: np.ndarray,
    frame: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
) -> np.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       gulp: a dynamic with time on the vertical axis,
       and freq on the horizontal

       frame (int): number of frequency samples to calculate MAD

       sigma (float): cutoff sigma

       chans_per_fit (int): polynomial/spline knots per channel to fit the bandpass

       fitter: which fitter to use

    Returns:

       Dynamic Spectrum with values clipped

    See:
        https://github.com/rohinijoshi06/mad-filter-gpu
    """
    frame = int(frame)
    data_type = gulp.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        fit = fitter(
            np.median(gulp[:, j : j + frame], axis=0), chans_per_fit=chans_per_fit
        )  # .astype(data_type)
        diff = gulp[:, j : j + frame] - fit
        cut = sigma * stats.median_abs_deviation(diff, axis=1, scale="Normal")
        medians = np.median(diff, axis=1)

        # thresh_top = np.tile(cut+medians, ( frame, 1)).T
        # thresh_bottom = np.tile(medians-cut, ( frame, 1)).T
        # mask = (thresh_bottom < diff) & (diff < thresh_top)
        # mask is where data is good

        thresh = np.tile(cut, (frame, 1)).T
        medians = np.tile(medians, (frame, 1)).T
        mask = np.abs(diff - medians) > thresh

        logging.info(f"mask: {mask.sum()}")

        try:  # sometimes this fails to converge, if happens use original fit
            masked_arr = np.ma.masked_array(gulp[:, j : j + frame], mask=mask)
            fit_clean = fitter(
                np.ma.median(masked_arr, axis=0),
                chans_per_fit=chans_per_fit,
            )
        except Exception as e:
            logging.warning(f"Failed to fit with Exception: {e}" ", using original fit")
            fit_clean = fit

        np.clip(fit_clean, min_value, max_value, out=fit_clean)
        # clip the values so they don't wrap
        # when converted to ints
        fit_clean = fit_clean.astype(data_type)
        # convert to dtype of the original
        gulp[:, j : j + frame] = np.where(
            mask, np.tile(fit_clean, (len(gulp), 1)), gulp[:, j : j + frame]
        )

    return gulp.astype(data_type)


def mad_time(
    gulp: np.ndarray, sigma: float = 6, frame: int = 128, return_values: bool = False
) -> np.ndarray:
    """
    Calculates the spectral Kurtosis along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

    `   sigma on to cut kurtosis values

        frame: number of time samples to calculate the kurtosis

        apply_mask: Apply the mask to the data, replacing bad values with zeros

    Returns:

       Mask based on bad iqr sections

       optional: apply mask as replace with zeros
    """
    frame = int(frame)
    test_values = np.zeros_like(gulp, dtype=np.float)
    mask = np.full_like(gulp, True, dtype=bool)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        test_vec = stats.median_abs_deviation(
            gulp[j : j + frame], axis=0, scale="normal"
        )
        test_values[j : j + frame, :] = test_vec

    # iqr_bandpass =
    # iqr_fit = poly_fitter(iqr_values.mean(axis=0))
    # iqr_flat = iqr_values - iqr_fit
    stds_test = stats.median_abs_deviation(test_values, scale="normal")
    meds_test = np.median(test_values)

    mask = test_values - meds_test < sigma * stds_test

    if return_values:
        return mask, test_values

    return mask


def sad_time(gulp, frame=128, window=65, sigma=3, clip=True):  # runs in time
    """
    Calculates Savgol Absolute Deviations along the time axis

    Args:
       frame: number of time samples to calculate the SAD

       sigma: cutoff sigma

    Returns:

       Dynamic Spectrum with values clipped
    """
    gulp = gulp.copy()
    frame = int(frame)
    data_type = gulp.dtype
    # savgol_array = sg(gulp, window, 2, axis=0)

    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        savgol_sub_array = sg(gulp[j : j + frame, :], window, 2, axis=0)
        cut = np.tile(
            np.array(
                1.4826
                * sigma
                * stats.median_absolute_deviation(
                    gulp[j : j + frame, :] - savgol_sub_array, axis=0
                )
            ),
            (frame, 1),
        )

        if clip:
            np.clip(
                gulp[j : j + frame, :],
                None,
                savgol_sub_array + cut,
                gulp[j : j + frame, :],
            )
        else:
            np.where(
                gulp[j : j + frame, :] > savgol_sub_array + cut,
                gulp[j : j + frame, :],
                0,
            )
    return gulp.astype(data_type)


def sad_spectra(gulp, frame=128, window=65, sigma=3, clip=True):
    """
    Calculates Savgol Absolute Deviations along the spectral axis

    Args:
       frame: number of time samples to calculate the SAD

       sigma: cutoff sigma

    Returns:

       Dynamic Spectrum with values clipped
    """
    gulp = gulp.copy()
    frame = int(frame)
    data_type = gulp.dtype
    savgol_array = sg(gulp, window, 2, axis=1)
    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        savgol_sub_array = sg(savgol_array[:, j : j + frame], window, 2, axis=1)
        print("test1")
        cut = np.tile(
            np.array(
                1.4826
                * sigma
                * stats.median_absolute_deviation(
                    gulp[:, j : j + frame] - savgol_sub_array, axis=1
                )
            ),
            (1, frame),
        )
        print("test2")
        if clip:
            np.clip(
                gulp[:, j : j + frame].T,
                None,
                savgol_sub_array.T + cut,
                gulp[:, j : j + frame].T,
            )
        else:
            np.where(
                gulp[:, j : j + frame].T > savgol_sub_array.T + cut,
                gulp[:, j : j + frame].T,
                0,
            )
    return gulp.astype(data_type)


def mad_time_cutter(gulp, frame=256, sigma=10):
    """
    Calculates Median Absolute Deviations along the time axis

    Args:
       frame: number of time samples to calculate the kurtosis

       sigma: cutoff sigma

    Returns:

       Dynamic Spectrum with values clipped
    """
    frame = int(frame)
    data_type = gulp.dtype
    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        cut = (
            1.4826
            * sigma
            * stats.median_absolute_deviation(gulp[j : j + frame, :], axis=0)
        )
        cut = np.transpose(cut)
        medians = np.median(gulp[j : j + frame, :], axis=0)
        np.clip(gulp[j : j + frame, :], None, medians + cut, gulp[j : j + frame, :])
    return gulp.as_type(data_type)


def skew_time(
    gulp: np.ndarray,
    p_cut: float = 0.001,
    frame: int = 128,
    return_values: bool = False,
) -> np.ndarray:
    """
    Calculates the spectral Skew on blocks along the time axis

    Args:
        gulp: the dynamic spectum to be analyzed

        p_cut: blocks with a pvalue below this number get cut

        frame: number of time samples to calculate the skew

        return_mask: bool
    Returns:

       mask

       optional: pvalues for each of the blocks
    """
    frame = int(frame)
    svalues = np.zeros_like(gulp, dtype=np.float)
    mask = np.zeros_like(gulp, dtype=bool)
    p_values = np.zeros_like(gulp, dtype=np.float)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        skew_vec, p_vec = stats.skewtest(gulp[j : j + frame], axis=0)
        svalues[j : j + frame, :] = skew_vec
        p_values[j : j + frame, :] = p_vec

    mask = p_values < p_cut

    # stds_kurt = np.std(kvalues, axis=0)
    # meds_kurt = np.median(kvalues, axis=0)

    # mask = np.abs(kvalues - meds_kurt) < sigma * stds_kurt

    if return_values:
        return mask, p_values

    return mask


def zero_dm(
    dynamic_spectra: np.ndarray, bandpass: np.ndarray = None, copy: bool = False
) -> np.ndarray:
    """
    Mask-safe zero-dm subtraction

    args:
        dynamic_spectra: The data you want to zero-dm, expects times samples
                         on the vertical axis. Accepts numpy.ma.arrays.

        bandpass - Use if a large file is broken up into pieces.
                   Be careful about how you use this with masks.

        copy: make a copy of the data instead of processing in place

    returns:
        dynamic spectra with a (more) uniform zero time series

    note:
        This should masked values. I am mainly conserned with bad data being spread out
        ny the filter, and this ignores masked values when calculating time series
        and bandpass

    example:
        yr = Your("some.fil")
        dynamic_spectra = yr.get_data(744000, 2 ** 14)

        mask = np.zeros(yr.your_header.nchans, dtype=bool)
        mask[0:100] = True # mask the first hundred channels

        dynamic_spectra = np.ma.array(dynamic_spectra,
                                        mask=np.broadcast_to(dynamic_spectra.shape))
        cleaned = zero_dm(dynamic_spectra)

    from:
        "An interference removal technique for radio pulsar searches" R.P Eatough 2009

    see:
        https://sourceforge.net/p/heimdall-astro/code/ci/master/tree/Pipeline/clean_filterbank_rfi.cu

        https://github.com/scottransom/presto/blob/de2cf58262190d35fb37dbebf8308a6e29d72adf/src/zerodm.c

        https://github.com/thepetabyteproject/your/blob/1f4b39326835e6bb87e0003318b433dc1455a137/your/writer.py#L232
    """
    if copy:
        dynamic_spectra = dynamic_spectra.copy()
    data_type = dynamic_spectra.dtype
    iinfo = np.iinfo(data_type)

    time_series = np.ma.mean(dynamic_spectra, axis=1)
    if bandpass is None:
        bandpass = np.ma.mean(dynamic_spectra, axis=0)  # .astype(data_type)

    np.clip(
        np.round(dynamic_spectra - time_series[:, None] + bandpass),
        iinfo.min,
        iinfo.max,
        out=dynamic_spectra,
    )
    return dynamic_spectra.astype(data_type)
