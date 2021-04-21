#!/usr/bin/env python3

import logging
from typing import Union

import cupy as cp
import numpy as np

from jess.scipy_cupy.stats import median_abs_deviation
from jess.utils.bandpass_fitter_gpu import bandpass_fitter


def spectral_mad(
    gulp: Union[int, float], frame: int = 256, sigma: float = 3, poly_order: int = 5
) -> Union[int, float]:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       gulp: a dynamic with time on the vertical axis, and freq on the horizonal

       frame (int): number of frequency samples to calculate the MAD

       sigma (float): cutoff sigma

       poly_order (int): polynomial order to fit for the bandpass

    Returns:

       Dynamic Spectrum with values clipped
    """
    frame = int(frame)
    data_type = gulp.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        fit = cp.array(
            bandpass_fitter_gpu(cp.median(gulp[:, j : j + frame], axis=0), poly_order=5)
        )  # .astype(data_type)
        # fit = cp.array(bandpass_fitter(cp.median(gulp[:,j:j+frame],axis=0).get(), poly_order=5))
        diff = gulp[:, j : j + frame] - fit
        cut = sigma * median_abs_deviation_gpu(diff, axis=1, scale="Normal")

        medians = cp.median(diff, axis=1)
        # threash_top = cp.tile(cut+medians, ( frame, 1)).T
        # threash_bottom = cp.tile(medians-cut, ( frame, 1)).T
        # mask = (threash_bottom < diff) & (diff < threash_top) #mask is where data is good

        threash = cp.tile(cut, (frame, 1)).T
        medians = cp.tile(medians, (frame, 1)).T
        mask = cp.abs(diff - medians) < threash
        logging.info(f"mask: {mask.sum()}")

        try:  # sometimes this fails to converge, if happens use origial fit
            fit_clean = bandpass_fitter_gpu(
                cp.array(
                    np.nanmedian(
                        cp.where(mask, gulp[:, j : j + frame], np.nan).get(), axis=0
                    )
                ),
                poly_order=poly_order,
            )
            # cp.nanmedian exists in cupy 9.0, should update this when released
        except Exception:
            logging.warning(
                f"Failed to fit with Exception: {Exception}, using original fit"
            )
            fit_clean = fit

        cp.clip(
            fit_clean, min_value, max_value, out=fit_clean
        )  # clip the values so they don't wrap when converted to ints
        fit_clean = fit_clean.astype(data_type)  # convert to dtype of the original
        gulp[:, j : j + frame] = cp.where(
            mask, gulp[:, j : j + frame], cp.tile(fit_clean, (len(gulp), 1))
        )

    return (gulp.get()).astype(data_type)
