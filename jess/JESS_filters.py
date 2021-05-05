#!/usr/bin/env python3

import logging
from typing import Union

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter as sg

from jess.bandpass_fitter import bandpass_fitter


def tkurtosis(gulp, frame=128, return_mask=False):
    """
    Calculates the spectral Kurtosis along the time axis

    Args:

        gulp: the dynamic spectum to be analyzed

        frame: number of time samples to calculate the kurtosis

        return_mask: bool
    Returns:

       Dynamic Spectrum with bad Kurtosis values removed

       optional: masked values
    """
    frame = int(frame)
    sigma_sk = 2.0 / np.sqrt(frame)
    kvalues = np.zeros_like(gulp)
    mask = np.zeros_like(gulp)
    for j in np.arange(0, len(gulp) - frame + 1, frame):
        kurtosis_vec = stats.kurtosis(gulp[j : j + frame], axis=0)
        kvalues[j : j + frame, :] = kurtosis_vec

    mask = kvalues > 1 + frame * sigma_sk

    if return_mask:
        return np.where(~mask, gulp, 0), mask
    else:
        return np.where(~mask, gulp, 0)


def tmad(gulp, frame=256, sigma=10):
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


def spectral_mad(
    gulp: np.ndarray, frame: int = 256, sigma: float = 3, poly_order: int = 5
) -> np.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       gulp: a dynamic with time on the vertical axis,
       and freq on the horizontal

       frame (int): number of frequency samples to calculate MAD

       sigma (float): cutoff sigma

       poly_order (int): polynomial order to fit the bandpass

    Returns:

       Dynamic Spectrum with values clipped
    """
    frame = int(frame)
    data_type = gulp.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        fit = bandpass_fitter(
            np.median(gulp[:, j : j + frame], axis=0), poly_order=5
        )  # .astype(data_type)
        diff = gulp[:, j : j + frame] - fit
        cut = sigma * stats.median_abs_deviation(diff, axis=1, scale="Normal")
        medians = np.median(diff, axis=1)

        # thresh_top = np.tile(cut+medians, ( frame, 1)).T
        # thresh_bottom = np.tile(medians-cut, ( frame, 1)).T
        # mask = (thresh_bottom < diff) & (diff < thresh_top) #mask is where data is good

        thresh = np.tile(cut, (frame, 1)).T
        medians = np.tile(medians, (frame, 1)).T
        mask = np.abs(diff - medians) < thresh

        logging.info(f"mask: {mask.sum()}")

        try:  # sometimes this fails to converge, if happens use origial fit
            fit_clean = bandpass_fitter(
                np.nanmedian(np.where(mask, gulp[:, j : j + frame], np.nan), axis=0),
                poly_order=poly_order,
            )
        except Exception as e:
            logging.warning(
                f"Failed to fit with Exception: {e}, using original fit"
            )
            fit_clean = fit

        np.clip(
            fit_clean, min_value, max_value, out=fit_clean
        )  # clip the values so they don't wrap when converted to ints
        fit_clean = fit_clean.astype(data_type)  # convert to dtype of the original
        gulp[:, j : j + frame] = np.where(
            mask, gulp[:, j : j + frame], np.tile(fit_clean, (len(gulp), 1))
        )

    return gulp.astype(data_type)


def time_sad(gulp, frame=128, window=65, sigma=3, clip=True):  # runs in time
    gulp = gulp.copy()
    """
    Calculates Savgol Absolute Deviations along the time axis

    Args:
       frame: number of time samples to calculate the SAD

       sigma: cutoff sigma

    Returns:
     
       Dynamic Spectrum with values clipped
    """
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


def spec_sad(gulp, frame=128, window=65, sigma=3, clip=True):
    gulp = gulp.copy()
    """
    Calculates Savgol Absolute Deviations along the spectral axis

    Args:
       frame: number of time samples to calculate the SAD

       sigma: cutoff sigma
    
    Returns:
     
       Dynamic Spectrum with values clipped
    """
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
