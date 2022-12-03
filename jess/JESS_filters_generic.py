#!/usr/bin/env python3

"""
Generic (cupy/numpy) filters.

This is a test of writing generic backend filters.
It relies on the user giving the correct ndarray.

If this is successful, we should merge JESS_filters and
JESS_filters_cupy into here.
"""

import logging
from typing import NamedTuple, Tuple, Union

import numpy as np

try:
    import os

    import cupy as xp

    try:
        if int(os.environ["CUDA_VISIBLE_DEVICES"]) < 0:
            raise RuntimeError
    except KeyError:
        pass
    from cupyx.scipy import ndimage

    from .calculators_cupy import flattner_median, flattner_mix, to_dtype
    from .fitters_cupy import median_fitter

except (ModuleNotFoundError, RuntimeError):
    xp = np
    from .fitters import median_fitter
    from scipy import ndimage
    from .calculators import flattner_median, to_dtype, flattner_mix

from .calculators import balance_chans_per_subband
from .scipy_cupy.stats import combined, iqr_med, median_abs_deviation_med


class FilterMaskResult(NamedTuple):
    """
    dynamic_spectra - Dynamic Spectra with RFI filtered
    mask - Boolean mask
    percent_masked - The percent masked
    """

    dynamic_spectra: xp.ndarray
    mask: xp.ndarray
    percent_masked: xp.float64


def calculate_skew_and_kurtosis(
    dynamic_spectra: xp.ndarray,
    samples_per_block: int,
    nan_policy: Union[str, None],
    winsorize_args: Union[Tuple, None],
) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Calculate the Kurtosis and Skew over a dynamic spectra using per-channel time blocks
    the size of `samples_per_block`.

    Args:
        dynamic_spectra - The spectra we want the values

        samples_per_block - Time samples for each channel block

        nan_policy - How to propagate nans. If None, does not check for nans.

        winsorize_args - Winsorize the second moments. See scipy_cupy.stats.winsorize
                         If `None`, no winorization.

    Return:
        skew, kurtosis, block limits
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
    return skew, kurtosis, limits


def _calculate_skew_z(skew: xp.ndarray, spb: int) -> xp.ndarray:
    """
    Calculate the skew z score

    Args:
        skew - skew values

        spd - samples per block

    Returns:
        skew z score
    """
    if spb < 8:
        raise ValueError(
            f"skewtest is not valid with less than 8 samples; {int(spb)} given."
        )
    if spb < 20:
        logging.warning("kurtosistest only valid for n>=20 n=%i given", int(spb))

    skew_cor = skew * xp.sqrt(((spb + 1) * (spb + 3)) / (6.0 * (spb - 2)))
    beta2 = (
        3.0
        * (spb**2 + 27 * spb - 70)
        * (spb + 1)
        * (spb + 3)
        / ((spb - 2.0) * (spb + 5) * (spb + 7) * (spb + 9))
    )
    weight_2 = -1 + xp.sqrt(2 * (beta2 - 1))
    delta = 1 / xp.sqrt(0.5 * xp.log(weight_2))
    alpha = xp.sqrt(2.0 / (weight_2 - 1))
    skew_cor = xp.where(skew_cor == 0, 1, skew_cor)
    return delta * xp.log(skew_cor / alpha + xp.sqrt((skew_cor / alpha) ** 2 + 1))


def _calculate_kurtosis_z(kurtosis: xp.ndarray, spb: int) -> xp.ndarray:
    """
    Calculate the kurtosis z score

    Args:
        kurtosis - kurtosis values

        spd - samples per block

    Returns:
        kurtosis z score
    """
    # calculate_skew_and_kurtosis returns fisher=True
    # so we want the differnce
    fisher_corr = 3.0 * ((spb - 1) / (spb + 1) - 1)
    varb2 = (
        24.0
        * spb
        * (spb - 2)
        * (spb - 3)
        / ((spb + 1) * (spb + 1.0) * (spb + 3) * (spb + 5))
    )  # [1]_ Eq. 1
    excess_kurt = (kurtosis - fisher_corr) / xp.sqrt(varb2)  # [1]_ Eq. 4
    # [1]_ Eq. 2:
    sqrtbeta1 = (
        6.0
        * (spb * spb - 5 * spb + 2)
        / ((spb + 7) * (spb + 9))
        * xp.sqrt((6.0 * (spb + 3) * (spb + 5)) / (spb * (spb - 2) * (spb - 3)))
    )
    # [1]_ Eq. 3:
    num_corr = 6.0 + 8.0 / sqrtbeta1 * (
        2.0 / sqrtbeta1 + xp.sqrt(1 + 4.0 / (sqrtbeta1**2))
    )
    term1 = 1 - 2 / (9.0 * num_corr)
    denom = 1 + excess_kurt * xp.sqrt(2 / (num_corr - 4.0))
    term2 = xp.sign(denom) * xp.where(
        denom == 0.0, xp.nan, xp.power((1 - 2.0 / num_corr) / xp.abs(denom), 1 / 3.0)
    )
    if xp.any(denom == 0):
        logging.warning(
            "Test statistic not defined in some cases due to division by "
            "zero. Return nan in that case..."
        )

    return (term1 - term2) / xp.sqrt(2 / (9.0 * num_corr))


def dagostino(
    dynamic_spectra: xp.ndarray,
    samples_per_block: int = 4096,
    sigma: float = 4,
    detrend: Union[Tuple, None] = (median_fitter, 40),
    winsorize_args: Union[Tuple, None] = (5, 40),
    nan_policy: Union[str, None] = None,
) -> xp.ndarray:
    """
    D'agostino Gaussianity test, this uses a combination of Kurtosis and Skew,
    taking into account the number of samples in each block
    We calculate the D'Agostino along the time axis in blocks of `samples_per_block`.
    This is balanced if the number of samples is not evenly divisible.
    The D'Agostino K^2 statstic is tailed.
    This makes our Gaussian outlier flagging remove more data than expected.
    To combate this we take the squareroot of the K^2 Statistic, this makes the
    distrabution more Gaussian and the flagging work better.

    Args:
        dynamic_spectra - Section spectra time on the vertical axis

        samples_per_block - Time samples for each channel block

        sigma - Sigma at which to flag

        detrend - Detrend Kurtosis and Skew values (fitter, chans_per_fit).
                  If `None`, no detrend

        winsorize_args - Winsorize the second moments. See scipy_cupy.stats.winsorize
                         If `None`, no winorization.

        nan_policy - How to propagate nans. If None, does not check for nans.

        alternative - Defines the alternative hypothesis.
            The following options are available (default 'two-sided'):
            - 'two-sided': the kurtosis of the distribution underlying the sample
            is different from that of the normal distribution
            - 'less': the kurtosis of the distribution underlying the sample
            is less than that of the normal distribution
            - 'greater': the kurtosis of the distribution underlying the sample
            is greater than that of the normal distribution


    Returns:
        FilterMaskResult with mask True=bad data, and percent masked
    """
    skew, kurtosis, limits = calculate_skew_and_kurtosis(
        dynamic_spectra,
        samples_per_block=samples_per_block,
        nan_policy=nan_policy,
        winsorize_args=winsorize_args,
    )
    skew_z = _calculate_skew_z(skew=skew, spb=samples_per_block)
    kurtosis_z = _calculate_kurtosis_z(kurtosis=kurtosis, spb=samples_per_block)

    da_2 = skew_z * skew_z + kurtosis_z * kurtosis_z
    # map to chi distrabution
    xp.sqrt(da_2, out=da_2)

    # should return the same as scipy.stats.normaltest
    # return NormaltestResult(da_2, distributions.chi2.sf(da_2, 2))

    if detrend is not None:
        da_2 -= detrend[0](xp.median(da_2, axis=0), chans_per_fit=detrend[1])

    da_scale, da_mid = iqr_med(da_2, scale="normal", axis=None, nan_policy=nan_policy)
    mask = da_2 - da_mid > sigma * da_scale

    mask_percent = 100 * mask.mean()
    logging.debug(
        "mask:%.2f",
        mask_percent,
    )
    # repeat needs a list
    repeats = xp.diff(limits).tolist()
    mask = xp.repeat(mask, repeats=repeats, axis=0)
    return FilterMaskResult(xp.array(xp.nan), mask, mask_percent)


def jarque_bera(
    dynamic_spectra: xp.ndarray,
    samples_per_block: int = 4096,
    sigma: float = 4,
    detrend: Union[Tuple, None] = (median_fitter, 40),
    winsorize_args: Union[Tuple, None] = (5, 40),
    nan_policy: Union[str, None] = None,
) -> FilterMaskResult:
    """
    Jarque-Bera Gaussianity test, this uses a combination of Kurtosis and Skew.
    We calculate Jarque-Bera along the time axis in blocks of `samples_per_block`.
    This is balanced if the number of samples is not evenly divisible.
    The Jarque-Bera statstic is Chi-Squared distributed with two degrees of freedom.
    This makes our Gaussian outlier flagging remove more data than expected.
    To combate this we take the squareroot of the Jarque Statistic, this makes the
    distrabution more Gaussian and the flagging work better.


    Args:
        dynamic_spectra - Section spectra time on the vertical axis

        samples_per_block - Time samples for each channel block

        detrend - Detrend Kurtosis and Skew values (fitter, chans_per_fit).
                  If `None`, no detrend

        winsorize_args - Winsorize the second moments. See scipy_cupy.stats.winsorize
                         If `None`, no winorization.

        nan_policy - How to propagate nans. If None, does not check for nans.

    Returns:
        FilterMaskResult with mask True=bad data, and percent masked
    """
    skew, kurtosis, limits = calculate_skew_and_kurtosis(
        dynamic_spectra,
        samples_per_block=samples_per_block,
        nan_policy=nan_policy,
        winsorize_args=winsorize_args,
    )

    # already calculate the excess kurtosis, so we don't need to
    # subtract 3
    j_b = samples_per_block / 6 * (skew**2 + kurtosis**2 / 4)
    xp.sqrt(j_b, out=j_b)

    if detrend is not None:
        j_b -= detrend[0](xp.median(j_b, axis=0), chans_per_fit=detrend[1])

    jb_scale, jb_mid = iqr_med(j_b, scale="normal", axis=None, nan_policy=nan_policy)
    mask = j_b - jb_mid > sigma * jb_scale

    mask_percent = 100 * mask.mean()
    logging.debug(
        "mask:%.2f",
        mask_percent,
    )
    # repeat needs a list
    repeats = xp.diff(limits).tolist()
    mask = xp.repeat(mask, repeats=repeats, axis=0)
    return FilterMaskResult(xp.array(xp.nan), mask, mask_percent)


def kurtosis_and_skew(
    dynamic_spectra: xp.ndarray,
    samples_per_block: int = 4096,
    sigma: float = 4,
    detrend: Union[Tuple, None] = (median_fitter, 40),
    winsorize_args: Union[Tuple, None] = (5, 40),
    nan_policy: Union[str, None] = None,
) -> FilterMaskResult:
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
    skew, kurtosis, limits = calculate_skew_and_kurtosis(
        dynamic_spectra,
        samples_per_block=samples_per_block,
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


def mad_spectra_flat(
    dynamic_spectra: xp.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 4,
    flatten_to: int = 64,
    time_median_size: int = 0,
    mask_chans: bool = False,
    return_same_dtype: bool = True,
    no_time_detrend: bool = False,
) -> xp.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels). This flattens the
    data by subtracting the rolling median of median of time and frequencies.
    It then calculates the Median Absolute Deviation for every frame channels.
    Outliers are removed based on the assumption of Gaussian data. The dynamic
    spectra is then detrended again, masking the outliers. This process is then
    repeated again. The data is returned centerned around flatten_to with removed
    points set as flatten_to.

    Args:
       dynamic_spectra: a dynamic spectra with time on the vertical axis,
                        and freq on the horizontal

       chans_per_subband: number of channels to calculate the MAD

       sigma: sigma which to reject outliers

       flatten_to: the median of the output data

       time_median_size: the length of the median filter to run in time

       mask_chan - Mask bad channels based on differing mean and median
                   seems to be a good test for non-linearity

       return_same_dtype: return the same data type as given

       no_time_detrend: Don't deterend in time, useful fo low dm
                         where pulse>%50 of the channel

    Returns:
       Dynamic Spectrum with values clipped

    See:
        https://github.com/rohinijoshi06/mad-filter-gpu

        Kendrick Smith's talks about CHIME FRB

    Note:
        This has better performance than spectral_mad, you should probably use this one.
    """

    data_type = dynamic_spectra.dtype

    if xp.issubdtype(data_type, xp.integer):
        info = xp.iinfo(data_type)
    else:
        info = xp.finfo(data_type)

    if not info.min < flatten_to < info.max:
        raise ValueError(
            f"""Can't flatten {data_type}, which has a range
            [{info.min}, {info.max}, to {flatten_to}"""
        )

    # I medfilt to try and stabalized the subtraction process against large RFI spikes
    # I choose 7 empirically
    flattened, ts0 = flattner_median(
        dynamic_spectra,
        flatten_to=flatten_to,
        kernel_size=7,
        return_time_series=True,
    )
    mask = xp.zeros_like(flattened, dtype=bool)

    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]

        mads, medians = median_abs_deviation_med(
            flattened[subband], axis=1, scale="Normal"
        )
        cut = sigma * mads

        if time_median_size > 1:
            logging.debug("Running median filter with size %i", time_median_size)
            ndimage.median_filter(cut, size=time_median_size, mode="mirror", output=cut)
            ndimage.median_filter(
                medians, size=time_median_size, mode="mirror", output=medians
            )

        mask[subband] += xp.abs(flattened[subband] - medians[:, None]) > cut[:, None]

        flattened[subband][mask[subband]] = xp.nan

    # want kernel size to be 1, so every channel get set,
    # now that we've removed the worst RFI
    flattened, ts1 = flattner_median(
        flattened, flatten_to=flatten_to, kernel_size=1, return_time_series=True
    )

    if mask_chans:
        means = xp.nanmean(flattened, axis=0)
        chan_noise, chan_mid = iqr_med(means, scale="normal", nan_policy="omit")
        chan_mask = xp.abs(means - chan_mid) > sigma * chan_noise
        mask += chan_mask
        logging.debug("%.2f%% Channels flagged", 100 * chan_mask.mean())

    # set the masked values to what we want to flatten to
    # not obvious why this has to be done, because nans should be ok
    # but it works better this way
    flattened[mask] = flatten_to

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        # Second iteration
        # flattened[:, j : j + frame] = flattner(
        #    flattened[:, j : j + frame], flatten_to=flatten_to, kernel_size=7
        # )

        mads, medians = median_abs_deviation_med(
            flattened[subband], axis=1, scale="Normal"
        )
        cut = sigma * mads

        if time_median_size > 1:
            ndimage.median_filter(cut, size=time_median_size, mode="mirror", output=cut)
            ndimage.median_filter(
                medians, size=time_median_size, mode="mirror", output=medians
            )

        mask_new = xp.abs(flattened[subband] - medians[:, None]) > cut[:, None]
        mask[subband] += mask_new
        flattened[subband][mask[subband]] = xp.nan

    # mean frequency subtraction makes sure there is smooth
    # transition between the blocks
    flattened, ts2 = flattner_mix(
        flattened, flatten_to=flatten_to, kernel_size=1, return_time_series=True
    )
    flattened[mask] = flatten_to

    if no_time_detrend:
        logging.debug("Adding time series back.")
        time_series = ts0 + ts1 + ts2
        # subtract off the median, this takes care of big
        # trends when flattening about spectra
        time_series = time_series - xp.median(time_series)
        flattened = flattened + time_series[:, None]

    percent_masked = mask.mean() * 100
    logging.debug("Masking %.2f %%", percent_masked)

    if return_same_dtype:
        flattened = to_dtype(flattened, dtype=data_type)

    return FilterMaskResult(flattened, mask, percent_masked)
