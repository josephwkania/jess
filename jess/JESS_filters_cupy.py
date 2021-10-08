#!/usr/bin/env python3
"""
This contains cupy versions of some of JESS_filters
 """
import logging

import cupy as cp
import numpy as np
from cupyx.scipy.signal import medfilt

from jess.calculators import balance_chans_per_subband
from jess.calculators_cupy import flattner_median, flattner_mix, to_dtype

# from jess.fitters import poly_fitter
from jess.fitters_cupy import poly_fitter
from jess.scipy_cupy.stats import median_abs_deviation


def fft_mad(
    dynamic_spectra: cp.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
    bad_chans: np.ndarray = None,
    return_same_dtype: bool = True,
    return_mask: bool = False,
) -> cp.ndarray:
    """
    Takes the real FFT of the dynamic spectra along the time axis
    (a FFT for each channel). Then take the absolute value, this
    gives the magnitude of the power @ each frequency.

    Then run the MAD filter along the freqency axis, this looks for
    outliers in the in the spectra. Narrow band RFI will only be
    in a few channels, add will be flagged.

    This mask is then used to flag the complex FFT by setting the
    flagged points to zero. The first row is excluded because this
    is the powers for each channel. This could be zero, but it has
    so effect, and keeping it at its current value keeps the
    bandpass smooth.

    The masked FFT is then inverse real fft back. Data is clip to
    min/max for the given input data type and returned as that
    data type.

    Args:
        dynamic_spectra: spectra block with time on the vertical axis,
                         and freq on the horizontal

        chans_per_subband: number of frequency samples to calculate MAD

        sigma: cutoff sigma

        chans_per_fit: polynomial/spline knots per channel to fit the bandpass

        fitter: which fitter to use, see jess.fitters for options

        bad_chans: list of bad channels - these have all information
                  removed except for the power

        return_same_dtype: return the same data type as given

        return_mask: return the bool mask of flagged frequencies

    Returns:
        Dynamic Spectrum with narrow band perodic RFI removed.

        (optional) bool mask of frequencies where bad=True

    See:

        For MAD
        https://github.com/rohinijoshi06/mad-filter-gpu

        For FFT cleaning
        https://arxiv.org/abs/2012.11630 & https://github.com/ymaan4/RFIClean

    Note:
        This provides a 1% difference in masks from the CPU version. This results in
        a 0.1% higher standard deviation of the zero dm time series.
        This seems negligible, this version provides 2x speed up on a GTX 1030 over
        24 threads of X5675.
    """

    data_type = dynamic_spectra.dtype

    dynamic_spectra_fftd = cp.fft.rfft(dynamic_spectra, axis=0)
    dynamic_spectra_fftd_abs = cp.abs(dynamic_spectra_fftd)
    mask = cp.zeros_like(dynamic_spectra_fftd_abs, dtype=bool)

    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )
    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        fit = fitter(
            cp.median(dynamic_spectra_fftd_abs[subband], axis=0),
            chans_per_fit=chans_per_fit,
        )  # .astype(data_type)
        diff = dynamic_spectra_fftd_abs[subband] - cp.array(fit)
        cut = sigma * median_abs_deviation(diff, axis=1, scale="Normal")
        medians = cp.median(diff, axis=1)
        # adds some resistance to jumps in medians
        # medians = medfilt(medians, 7)

        # mask[subband] = cp.abs(diff - medians[:, None]) > cut[:, None]
        mask[subband] = diff - medians[:, None] > cut[:, None]

    # remove infomation for the bad channels, but leave power
    # this has no effect on the following filter
    # which works on gulp_fftd_abd
    if bad_chans is not None:
        logging.debug("Applying channel mask %s", bad_chans)
        mask[1:, bad_chans] = True

    mask[0, :] = False  # set the row to false to preserve the powser levels
    dynamic_spectra_fftd[mask] = 0

    # We're flagging complex data, so multiply by 2
    logging.info("Masked Percentage: %.2f %%", mask.mean() * 100 * 2)

    dynamic_spectra_cleaned = cp.fft.irfft(dynamic_spectra_fftd, axis=0)

    if return_same_dtype:
        dynamic_spectra_cleaned = to_dtype(dynamic_spectra_cleaned, dtype=data_type)

    if return_mask:
        return dynamic_spectra_cleaned, mask
    return dynamic_spectra_cleaned


def mad_spectra(
    dynamic_spectra: cp.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    return_same_dtype: bool = True,
    return_mask: bool = False,
) -> cp.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       dynamic_spectra: spectra with time on the vertical axis,
                        and freq on the horizontal

       frame: number of frequency samples to calculate the MAD

       sigma: cutoff sigma

       poly_order: polynomial order to fit for the bandpass

       return_same_dtype: return the same data type as given

    Returns:
       Dynamic Spectrum with values clipped

    Notes:
        This version differs from the cpu version.
        This uses nans, while cpu version uses np.ma mask,
        excision performance is about the same. See the cpu docstring for
        references.

        mask = True, for good values

        You should use spectral_mad_flat (which has better RFI removal) unless
        you really need to preserve the bandpass.
    """
    data_type = dynamic_spectra.dtype
    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )
    mask = cp.zeros_like(dynamic_spectra, dtype=bool)

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        fit = poly_fitter(
            cp.median(dynamic_spectra[subband], axis=0),
            chans_per_fit=chans_per_fit,
        )
        # .astype(data_type)

        diff = dynamic_spectra[subband] - fit
        cut = sigma * median_abs_deviation(diff, axis=1, scale="Normal")
        medians = cp.median(diff, axis=1)
        mask[subband] = cp.abs(diff - medians[:, None]) < cut[:, None]

        try:  # sometimes this fails to converge, if happens use original fit
            fit_clean = poly_fitter(
                cp.nanmedian(
                    cp.where(
                        mask[subband],
                        dynamic_spectra[subband],
                        np.nan,
                    ),
                    axis=0,
                ),
                chans_per_fit=chans_per_fit,
            )
        except Exception as excpt:
            logging.warning(
                "Failed to fit with Exception: %s, using original fit", excpt
            )
            fit_clean = fit
        dynamic_spectra[subband] = cp.where(
            mask[subband],
            dynamic_spectra[subband],
            fit_clean,
        )

    logging.info("Masking %.2f %%", (1 - mask.mean()) * 100)

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, dtype=data_type)

    if return_mask:
        return dynamic_spectra, mask
    return dynamic_spectra


def mad_spectra_flat(
    dynamic_spectra: cp.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    flatten_to: int = 64,
    median_time_kernel: int = 0,
    return_mask: bool = False,
    return_same_dtype: bool = True,
) -> cp.ndarray:
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

       return_same_dtype: return the same data type as given

        return_mask: return the mask where True=masked_values

    Returns:
       Dynamic Spectrum with values clipped

    See:
        https://github.com/rohinijoshi06/mad-filter-gpu

        Kendrick Smith's talks about CHIME FRB

    Note:
        This has better performance than spectral_mad, you should probably use this one.
    """

    data_type = dynamic_spectra.dtype
    iinfo = np.iinfo(data_type)

    if not iinfo.min < flatten_to < iinfo.max:
        raise ValueError(
            f"""Can't flatten {data_type}, which has a range
            [{iinfo.min}, {iinfo.max}, to {flatten_to}"""
        )

    # I medfilt to try and stabalized the subtraction process against large RFI spikes
    # I choose 7 empirically
    flattened = flattner_median(
        cp.asarray(dynamic_spectra), flatten_to=flatten_to, kernel_size=7
    )
    mask = cp.zeros_like(flattened, dtype=bool)
    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        cut = sigma * median_abs_deviation(flattened[subband], axis=1, scale="Normal")
        medians = cp.median(flattened[subband], axis=1)

        if median_time_kernel > 2:
            cut = medfilt(cut, kernel_size=median_time_kernel)
            medians = medfilt(medians, kernel_size=median_time_kernel)

        mask[subband] = cp.abs(flattened[subband] - medians[:, None]) > cut[:, None]

        flattened[subband][mask[subband]] = cp.nan

    # want kernel size to be 1, so every channel get set,
    # now that we've removed the worst RFI
    flattened = flattner_median(flattened, flatten_to=flatten_to, kernel_size=1)
    # set the masked values to what we want to flatten to
    # not obvus why this has to be done, because nans should be ok
    # but it works better this way
    flattened[mask] = flatten_to

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        # Second iteration
        # flattened[:, j : j + frame] = flattner(
        #    flattened[:, j : j + frame], flatten_to=flatten_to, kernel_size=7
        # )
        cut = sigma * median_abs_deviation(flattened[subband], axis=1, scale="Normal")
        medians = cp.median(flattened[subband], axis=1)

        if median_time_kernel > 2:
            cut = medfilt(cut, kernel_size=median_time_kernel)
            medians = medfilt(medians, kernel_size=median_time_kernel)

        mask_new = cp.abs(flattened[subband] - medians[:, None]) > cut[:, None]
        mask[subband] = mask[subband] + mask_new
        flattened[subband][mask[subband]] = cp.nan

    # mean frequency subtraction makes sure there is smooth
    # transition between the blocks
    flattened = flattner_mix(flattened, flatten_to=flatten_to, kernel_size=1)
    flattened[mask] = flatten_to

    logging.info("Masking %.2f %%", mask.mean() * 100)

    if return_same_dtype:
        flattened = to_dtype(flattened, dtype=data_type)

    if return_mask:
        return flattened, mask
    return flattened


def zero_dm_fft(
    dynamic_spectra: cp.ndarray,
    bandpass: cp.ndarray = None,
    modes_to_zero: int = 2,
    return_same_dtype: bool = True,
) -> cp.ndarray:
    """
    This removes low frequency components from each spectra. This extends 0-DM
    subtraction. 0-DM subtraction as described in Eatough 2009, involves subtraction
    of the mean of each spectra from each spectra, makeing the zero-DM time series
    contant. This is effective in removing broadband RFI that has no structure.

    This is very effective for moderate bandwidths and low dynamic ranges.
    As bandwidths increase, we can see zero-DM RFI that only extends through
    part of the band. Increases in dynamic range allow for zero-DM RFI to have
    spectral structure, either intrinsically or the result of the receiving chain.

    This attempts to corret for these problems with the subtraction method.
    This removes the zero Fourier term (the total power), equivalent to the
    subtraction method. It also can remove higher order terms, removing slowing
    signals across the band.

    You need to be careful about how many terms you remove. We will start to
    to remove more components of the pulse. When this happens is determined
    by the fraction of the band that contains the pulse. The larger the pulse,
    the lower the Fourier components.

    args:
        dynamic_spectra - The dynamic spectra you want to clean. Time axis
                         must be vertical

        bandpass - Bandpass to add. We subtract off the DC component, we
                   must add it back to safely write the data as unsigned ints
                   if no bandpass is given, this will use the bandpass from the
                   dynamic spectra given, this can cause jumps if you are processing
                   multiple chunks.

        modes_to_zero - The number of modes to filter.

        return_same_dtype: return the same data type as given

    returns:
        dynamic spectra with low frequency modes filtered, same data type as given

    notes:
        See jess.filters.zero_dm Docstring for other implementations
        of subtraction 0-dm filters.
    """
    assert isinstance(
        modes_to_zero, int
    ), f"You must give an integer number of nodes, you gave {modes_to_zero}"
    if modes_to_zero == 0:
        raise ValueError("You said to zero no modes, this will have no effect")
    if modes_to_zero == 1:
        logging.warning("Only removing first mode, consider using standard zero-dm")

    data_type = dynamic_spectra.dtype
    # dynamic_spectra = cp.asarray(dynamic_spectra)

    if bandpass is None:
        bandpass = dynamic_spectra.mean(axis=0)
    # else:
    #    bandpass = cp.asarray(bandpass)

    dynamic_spectra_fftd = cp.fft.rfft(dynamic_spectra, axis=1)

    # They FFT'd dynamic spectra will be 1/2 or 1/2+1 the size of
    # the dynamic spectra since FFT is complex
    mask = cp.zeros(dynamic_spectra_fftd.shape[1], dtype=bool)
    mask[
        :modes_to_zero,
    ] = True

    # masking complex number, multiply by two
    logging.info("Masked Percentage: %.2f %%", 2 * 100 * mask.mean())

    # zero out the modes we don't want
    dynamic_spectra_fftd[cp.broadcast_to(mask, dynamic_spectra_fftd.shape)] = 0

    # Add the bandpass back so out values are in the correct range
    dynamic_spectra_cleaned = cp.fft.irfft(dynamic_spectra_fftd, axis=1) + bandpass

    if return_same_dtype:
        dynamic_spectra_cleaned = to_dtype(dynamic_spectra_cleaned, dtype=data_type)

    return dynamic_spectra_cleaned
