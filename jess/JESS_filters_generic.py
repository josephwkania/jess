#!/usr/bin/env python3

"""
Generic (cupy/numpy) filters.

This is a test of writing generic backend filters.
It relies on the user giving the correct ndarray.

If this is successful, we should merge JESS_fiters and
JESS_fiters_cupy into here.
"""

import logging
from typing import NamedTuple, Tuple, Union

try:
    import cupy as xp

    from .fitters_cupy import median_fitter

except ModuleNotFoundError:
    import numpy as xp
    from .fitters import median_fitter

from .calculators import balance_chans_per_subband
from .scipy_cupy.stats import combined, iqr_med


class FilterMaskResult(NamedTuple):
    """
    dynamic_spectra - Dynamic Spectra with RFI filtered
    mask - Boolean mask
    percent_masked - The percent masked
    """

    dynamic_spectra: xp.ndarray
    mask: xp.ndarray
    percent_masked: xp.float64


def kurtosis_and_skew(
    dynamic_spectra: xp.ndarray,
    samples_per_block: int = 4096,
    sigma: float = 4,
    detrend: Union[Tuple, None] = (median_fitter, 20),
    winsorize_args: Union[Tuple, None] = (5, 40),
    nan_policy: Union[str, None] = None,
) -> xp.ndarray:
    """
    Gaussainity test using Kurtosis and Skew. We calculate Kurtosis and skew along
    the time axis in blocks of `samples_per_block`. This is balanced if the number
    of samples is not evenly divisible. We then use the central limit theorem to
    flag outlying samples in Kurtosis and Skew individually. These masks are then
    added together.

    Args:
        dynamic_spectra - Section spectra time on the vertical axis

        samples_per_block - Time samples for each channel block

        detrend - Detrend Kurtosis and Skew values (fitter, chans_per_fit).
                  If `None`, no detrend

        winsorize_args - Winsorize the second moments. See scipy_cupy.stats.winsorize
                         If `None`, no winorization.

        nan_policy - How to propagate nans. If None, does not check for nans.

    Returns:
        bool Mask with True=bad data

    Notes:
        Flagging based on
        https://www.worldscientific.com/doi/10.1142/S225117171940004X
    """
    num_cols, limits = balance_chans_per_subband(
        dynamic_spectra.shape[0], samples_per_block
    )
    skew = xp.zeros((num_cols, dynamic_spectra.shape[1]), dtype=xp.float64)
    kurtosis = xp.zeros_like(skew)
    for jcol in range(num_cols):
        column = xp.index_exp[limits[jcol] : limits[jcol + 1]]
        skew[jcol], kurtosis[jcol] = combined(
            dynamic_spectra[column],
            axis=0,
            nan_policy=nan_policy,
            winsorize_args=winsorize_args,
        )

    if detrend is not None:
        skew -= detrend[0](xp.median(skew, axis=0), chans_per_fit=detrend[1])
        kurtosis -= median_fitter(xp.median(kurtosis, axis=0))
    skew_scale, skew_mid = iqr_med(
        skew, scale="normal", axis=None, nan_policy=nan_policy
    )
    kurt_scale, kurt_mid = iqr_med(
        kurtosis, scale="normal", axis=None, nan_policy=nan_policy
    )
    skew_mask = skew - skew_mid > sigma * skew_scale
    kurt_mask = kurtosis - kurt_mid > sigma * kurt_scale
    mask = skew_mask + kurt_mask
    mask_percent = 100 * mask.mean()
    logging.debug(
        "skew_mask:%.2f kurtosis_mask:%.2f, mask:%.2f",
        100 * skew_mask.mean(),
        100 * kurt_mask.mean(),
        mask_percent,
    )
    # repeat needs a list
    repeats = xp.diff(limits).tolist()
    mask = xp.repeat(mask, repeats=repeats, axis=0)
    return FilterMaskResult(xp.array(xp.nan), mask, mask_percent)
