#!/usr/bin/env python3
"""
The repository for all my filters
"""
import logging
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import numpy as np
from rich.progress import track
from scipy import ndimage, signal, stats
from your import Your

import jess._sumthreshold_utils as sm
from jess.calculators import (
    autocorrelate,
    balance_chans_per_subband,
    decimate,
    flattner_median,
    flattner_mix,
    mean,
    median_abs_deviation_med,
    shannon_entropy,
    to_dtype,
)
from jess.fitters import arpls_fitter, median_fitter, poly_fitter


class FilterMaskResult(NamedTuple):
    """
    dynamic_spectra - Dynamic Spectra with RFI filtered
    mask - Boolean mask
    percent_masked - The percent masked
    """

    dynamic_spectra: np.ndarray
    mask: np.ndarray
    percent_masked: np.float64


class FilterResult(NamedTuple):
    """
    dynamic_spectra - Dynamic Spectra with RFI filtered
    percent_masked - The percent masked
    """

    dynamic_spectra: np.ndarray
    percent_masked: np.float64


def run_filter(
    file: str, filter_name: str, window: int = 64, time_median_kernel: int = 0
):
    """
    Runs filter on a file
    """
    yr_file = Your(file)
    filter_name = filter_name.casefold()

    if filter == "anderson":
        test_values = anderson_calculate_values(
            yr_file, window=window, time_median_kernel=time_median_kernel
        )
    else:
        raise NotImplementedError(f"You asked for {filter}, which is not available!")

    mask = central_limit_masker(test_values, window=window)
    _ = mask


def central_limit_masker(
    test_values: np.ndarray,
    window: int,
    sigma: float = 5,
    chans_per_subband: int = 512,
    remove_lower=True,
) -> np.ndarray:
    """
    Uses the central limit theorem to look for outliers in each subband.
    When looking at a large amount of values, the central limit theorem
    says the values should start to be Guassian distributed. We can use
    this to flag outliers.

    Using subbands we can take into account changes in sensitively across
    the band/cavity filters.

    Args:
        test_values: the values from a statistical test

        window: the window size of the test

        sigma: sigma at which to flag values

        num_subbands: number of subbands

    return:
        bool array with outliers get set as true, this will be the same
        size as the data.

    notes:
        see "Spectral Kurtosis-Based RFI Mitigation for CHIME"
        https://arxiv.org/abs/1808.10365

        and

        "High cadence kurtosis based RFI excision for CHIME"
        https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0394838?o=5
    """
    mask = np.zeros((test_values.shape[0] * window, test_values.shape[1]), dtype=bool)
    num_subbands, limits = balance_chans_per_subband(
        test_values.shape[1], chans_per_subband
    )
    for jsub in range(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        median = np.median(test_values[subband])
        std = stats.median_abs_deviation(
            test_values[subband], scale="normal", axis=None
        )
        if remove_lower:
            mask[subband] = np.repeat(
                np.abs(test_values[subband] - median) > sigma * std, window, axis=0
            )
        else:
            mask[subband] = np.repeat(
                test_values[subband] - median > sigma * std, window, axis=0
            )

    return mask


def anderson_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Run a Anderson Darling test on a Fits/Filterbank

    Args:
        yr_file: Your object

        window: window size for the test

        time_median_kernel: remove baseline by subtracting a running median
                            of time_median_kernel length long. Default is
                            no subtraction

        returns:
            array of anderson darling values for each window.
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    anderson = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        for kchan in range(nchan):
            anderson[j, kchan] = stats.anderson(chunk[:, kchan]).statistic

    return anderson


def autocorrelation_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate Autocorrelation for blocks of length window.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Autocorrelation values for each block in the file

    Notes:
        Not sure what the best test for autocorrelation.

        https://arxiv.org/pdf/2108.12434.pdf uses the absolute magnitude of
        of a one sample delay.

        The Durbin-Watson statistic also uses the first lag
        https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic

        I've tried some other tests,
            - the sum of the absolute values of all the correlations
            - (std - iqr) to look for outliers
            - the absolute value of the first lag
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    autocorr = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        autocorr_abs = np.abs(autocorrelate(chunk, axis=0))
        autocorr[j, :] = autocorr_abs.sum(axis=0)
        # ac = autocorrelate(chunk, axis=0) #[1:]
        # print(ac[1].shape)
        # autocorr[j, :] = np.abs(ac[1])
        # autocorr[j, :] = np.std(ac, axis=0) - stats.iqr(ac, axis=0, scale="normal")

    return autocorr


def dagostino_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Run a D'Agostino test on a Fits/Filterbank

    Args:
        yr_file: Your object

        window: window size for the test

        time_median_kernel: remove baseline by subtracting a running median
                            of time_median_kernel length long. Default is
                            no subtraction

        returns:
            array of D'Agostino values for each window.
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    dagostino = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        dagostino[j, :] = stats.normaltest(chunk, axis=0).statistic

    return dagostino


def entropy_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate Shannon Entropy for blocks of length window.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Shannon values for each block in the file
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    entropy = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        entropy[j, :] = shannon_entropy(chunk, axis=0)

    return entropy


def fft_mad(
    dynamic_spectra: np.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    time_median_size: int = 7,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
    bad_chans: np.ndarray = None,
    return_same_dtype: bool = True,
) -> np.ndarray:
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

        time_median_size: the length of the median filter to run in time

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

    """

    data_type = dynamic_spectra.dtype

    dynamic_spectra_fftd = np.fft.rfft(dynamic_spectra, axis=0)
    dynamic_spectra_fftd_abs = np.abs(dynamic_spectra_fftd)
    mask = np.zeros_like(dynamic_spectra_fftd_abs, dtype=bool)

    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        fit = fitter(
            np.median(dynamic_spectra_fftd_abs[subband], axis=0),
            chans_per_fit=chans_per_fit,
        )  # .astype(data_type)

        diff = dynamic_spectra_fftd_abs[subband] - fit
        mads, medians = median_abs_deviation_med(diff, axis=1, scale="Normal")
        cut = sigma * mads

        # adds some resistance to jumps in medians
        if time_median_size > 1:
            logging.debug("Applying Median filter length %i in time", time_median_size)
            ndimage.median_filter(
                medians, size=time_median_size, mode="mirror", output=medians
            )
            ndimage.median_filter(cut, size=time_median_size, mode="mirror", output=cut)

        mask[subband] = np.abs(diff - medians[:, None]) > cut[:, None]

    # maybe some lekage into the nearby channels
    # but this doesn't seem to help much
    # mask = ndimage.binary_dilation(mask)

    # remove infomation for the bad channels, but leave power
    # this has no effect on the following filter
    # which works on gulp_fftd_abd
    if bad_chans is not None:
        logging.debug("Applying channel mask %s", bad_chans)
        mask[1:, bad_chans] = True

    mask[0, :] = False  # set the row to false to preserve the powser levels

    # zero masked values
    dynamic_spectra_fftd[mask] = 0

    # We're flagging complex data, so multiply by 2
    percent_flagged = mask.mean() * 100 * 2
    logging.debug("Masked Percentage: %.2f %%", percent_flagged)

    dynamic_spectra_cleaned = np.fft.irfft(dynamic_spectra_fftd, axis=0)

    if return_same_dtype:
        dynamic_spectra_cleaned = to_dtype(dynamic_spectra_cleaned, dtype=data_type)

    return FilterMaskResult(dynamic_spectra_cleaned, mask, percent_flagged)


def jarque_bera_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate the Jaque Bera statistic.
    Use robust ways to normalize the blocks.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Jarque Bera statistic
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    jarque_bera = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        for kchan in range(nchan):
            jarque_bera[j, kchan] = stats.jarque_bera(chunk[:, kchan]).statistic

    return jarque_bera


def lilliefors_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate the Lilliefors statistic.
    Use robust ways to normalize the blocks.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Lilliefors statistic
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    lilliefors = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        chunk -= np.median(chunk, axis=0)
        chunk = chunk / stats.median_abs_deviation(chunk, axis=0)
        for kchan in range(nchan):
            lilliefors[j, kchan] = stats.kstest(
                chunk[:, kchan], stats.norm.cdf
            ).statistic

    return lilliefors


def std_iqr_calculate_values(
    yr_file: object,
    window: int = 64,
    time_median_kernel: int = 0,
    bandpass_kernel: int = 7,
) -> np.ndarray:
    """
    Take the difference between Standard Deviation and Inter Quatile Range
    (scaled to give STD) This is the difference between a robust measure
    and non-robust measure of scale.
    For Gaussian data then should agree, with outliers, this will differ.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Distances for each block

    Notes:
        Kevin's idea
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    distance = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        distance[j, :] = np.std(chunk, axis=0) - signal.medfilt(
            stats.iqr(chunk, axis=0, scale="normal"), kernel_size=bandpass_kernel
        )

    return distance


def kurtosis_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate Kurtosis for blocks of length window.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Kurtosis
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    kurtosis = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        kurtosis[j, :] = stats.kurtosis(chunk, axis=0)

    return kurtosis


def mad_spectra(
    dynamic_spectra: np.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
    return_same_dtype: bool = True,
) -> np.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       dynamic_spectra: dynamic spectra with time on the vertical axis,
       and freq on the horizontal

       chans_per_subband: number of channels to calculate the MAD

       sigma: cutoff sigma

       chans_per_fit: polynomial/spline knots per channel to fit the bandpass

       fitter: which fitter to use, see jess.fitters

       return_same_dtype: return the same data type as given

       return_mask: return the mask

    Returns:
       Dynamic Spectrum with values clipped

    See:
        https://github.com/rohinijoshi06/mad-filter-gpu

    Notes:
        mad_spectra_flat has better excision performance, you should consider that
        unless you need to keep the bandpass shape
    """
    data_type = dynamic_spectra.dtype
    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )
    mask = np.zeros_like(dynamic_spectra, dtype=bool)

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        fit = fitter(
            np.median(dynamic_spectra[subband], axis=0), chans_per_fit=chans_per_fit
        )  # .astype(data_type)
        diff = dynamic_spectra[subband] - fit
        cut = sigma * stats.median_abs_deviation(diff, axis=1, scale="Normal")
        medians = np.median(diff, axis=1)

        # thresh_top = np.tile(cut+medians, ( frame, 1)).T
        # thresh_bottom = np.tile(medians-cut, ( frame, 1)).T
        # mask = (thresh_bottom < diff) & (diff < thresh_top)
        # mask is where data is good

        # thresh = np.tile(cut, (frame, 1)).T
        # medians = np.tile(medians, (frame, 1)).T
        # mask = np.abs(diff - medians) > thresh

        mask[subband] = np.abs(diff - medians[:, None]) > cut[:, None]

        try:  # sometimes this fails to converge, if happens use original fit
            masked_arr = np.ma.masked_array(
                dynamic_spectra[subband], mask=mask[subband]
            )
            fit_clean = fitter(
                np.ma.median(masked_arr, axis=0),
                chans_per_fit=chans_per_fit,
            )
        except Exception as excpt:
            logging.warning(
                "Failed to fit with Exception: %f, using original fit", excpt
            )
            fit_clean = fit
        dynamic_spectra[subband] = np.where(
            mask[subband], fit_clean, dynamic_spectra[subband]
        )
        # dynamic_spectra[subband][mask[subband]] = fit_clean[mask[subband]]

    masked_percentage = mask.mean() * 100
    logging.debug("Masked Percentage: %.2f %%", masked_percentage)

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, dtype=data_type)

    return FilterMaskResult(dynamic_spectra, mask, masked_percentage)


def mad_spectra_flat(
    dynamic_spectra: np.ndarray,
    chans_per_subband: int = 256,
    sigma: float = 3,
    flatten_to: int = 64,
    time_median_size: int = 0,
    return_same_dtype: bool = True,
    no_time_detrend: bool = False,
) -> np.ndarray:
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
    if np.issubdtype(data_type, np.integer):
        info = np.iinfo(data_type)
    else:
        info = np.finfo(data_type)

    if not info.min < flatten_to < info.max:
        raise ValueError(
            f"""Can't flatten {data_type}, which has a range
            [{info.min}, {info.max}, to {flatten_to}"""
        )

    # I medfilt to try and stabalized the subtraction process against large RFI spikes
    # I choose 7 empirically
    flattened, ts0 = flattner_median(
        dynamic_spectra, flatten_to=flatten_to, kernel_size=7, return_time_series=True
    )
    # print(f"flattned.dytpe: {flattened.dtype}")
    mask = np.zeros_like(flattened, dtype=bool)
    num_subbands, limits = balance_chans_per_subband(
        dynamic_spectra.shape[1], chans_per_subband
    )

    for jsub in np.arange(0, num_subbands):
        subband = np.index_exp[:, limits[jsub] : limits[jsub + 1]]
        mads, medians = median_abs_deviation_med(
            flattened[subband], axis=1, scale="Normal"
        )
        # medians = np.median(flattened[subband], axis=1)
        cut = sigma * mads

        if time_median_size > 1:
            logging.debug("Applying Median filter length %i in time", time_median_size)
            ndimage.median_filter(cut, size=time_median_size, mode="mirror", output=cut)
            ndimage.median_filter(
                medians, size=time_median_size, mode="mirror", output=medians
            )

        mask[subband] = np.abs(flattened[subband] - medians[:, None]) > cut[:, None]

        flattened[subband][mask[subband]] = np.nan

    # want kernel size to be 1, so every channel get set,
    # now that we've removed the worst RFI
    flattened, ts1 = flattner_median(
        flattened, flatten_to=flatten_to, kernel_size=1, return_time_series=True
    )
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
        # medians = np.median(flattened[subband], axis=1)
        cut = sigma * mads

        if time_median_size > 1:
            ndimage.median_filter(cut, size=time_median_size, mode="mirror", output=cut)
            ndimage.median_filter(
                medians, size=time_median_size, mode="mirror", output=medians
            )

        mask_new = np.abs(flattened[subband] - medians[:, None]) > cut[:, None]
        mask[subband] = mask[subband] + mask_new
        flattened[subband][mask[subband]] = np.nan

    # mean frequency subtraction makes sure there is smooth
    # transition between the blocks
    flattened, ts2 = flattner_mix(
        flattened, flatten_to=flatten_to, kernel_size=1, return_time_series=True
    )
    flattened[mask] = flatten_to

    if no_time_detrend:
        time_series = ts0 + ts1 + ts2
        time_series = time_series - np.median(time_series)
        flattened = flattened + time_series[:, None]

    masked_percentage = mask.mean() * 100
    logging.debug("Masking %.2f %%", masked_percentage)

    if return_same_dtype:
        flattened = to_dtype(flattened, dtype=data_type)

    return FilterMaskResult(flattened, mask, masked_percentage)


def filter_weights(
    dynamic_spectra: np.ndarray,
    metric: Callable = np.median,
    bandpass_smooth_length: int = 50,
    cut_sigma: float = 2 / 3,
    smooth_sigma: int = 30,
) -> np.ndarray:
    """
    Makes weights based on low values of some meteric.
    This is designed to ignore bandpass filters or tapers
    at the end of the bandpass.

    Args:
        dynamic_spectra - 2D dynamic spectra with time on the
                          vertical axis

        metric - The statistic to sample.

        bandpass_smooth_length - length of the median filter to
                                 smooth the bandpass

        sigma_cut - Cut values below (standard deviation)*(sigma cut)

        smooth_sigma - Gaussian filter smoothing sigma. If =0, return
                       the mask where True=good channels

    Returns:
        Bandpass weights for sections of spectra with low values.
        0 where the value is below the threshold and 1 elsewhere,
        with a Gaussian taper.
    """
    bandpass = metric(dynamic_spectra, axis=0)
    bandpass_std = stats.median_abs_deviation(bandpass, scale="normal")
    threshold = bandpass_std * cut_sigma
    if bandpass_smooth_length > 1:
        bandpass = median_fitter(bandpass, chans_per_fit=bandpass_smooth_length)
    mask = bandpass > threshold

    if smooth_sigma > 0:
        return ndimage.gaussian_filter1d((mask).astype(float), sigma=smooth_sigma)
    return mask


def iterative_mad(
    dynamic_spectra: np.ndarray,
    factor: int,
    sigma: float = 4,
    time_median_size: int = 64,
    chans_per_subband: int = 256,
    flatten_to: int = 64,
    **filter_weight_args,
) -> np.ndarray:
    """
    Interatively clean a chunk of dynamic spectra.
    If filter_weight_args is not `None`, masks channels that
    are unlickely to have signal. MAD and MAD FFT are run.
    Then the channels are decimated by `factor` and MAD is run
    again, decreasing `chans_per_subband` by `factor`. This
    is repeated until `factor` channels, when the mean is taken

    Args:
        dynamic_spectra: a dynamic spectra with time on the vertical axis,
                        and freq on the horizontal

        factor - Factor to reduce each iteration

        chans_per_subband: Number of channels to calculate the MAD

        sigma: sigma which to reject outliers

        flatten_to: the median of the output data

        time_median_size: the length of the median filter to run in time

        filter_weight_args: Passed to `filter_weights`, if `None` no
                            filter of channels.

    Returns:
        Cleaned Time Series

    """
    if filter_weight_args is not None:
        chan_mask = filter_weights(
            dynamic_spectra, smooth_sigma=0, **filter_weight_args
        )
        dynamic_spectra = dynamic_spectra[:, chan_mask]

    dynamic_spectra, _, mask_percentage = mad_spectra_flat(
        dynamic_spectra,
        no_time_detrend=True,
        sigma=sigma,
        time_median_size=time_median_size,
        return_same_dtype=False,
    )
    dynamic_spectra, _, fft_mask_percentage, = fft_mad(
        dynamic_spectra,
        sigma=sigma,
        time_median_size=time_median_size // 2,
        return_same_dtype=False,
    )

    loop_mask_percentage = 0.0
    while dynamic_spectra.shape[1] > factor * factor:
        logging.debug("Number channels %i", dynamic_spectra.shape[1])
        dynamic_spectra = decimate(
            dynamic_spectra,
            freq_factor=factor,
            backend=partial(mean, pad="reflect"),
        )

        chans_per_subband = np.around(chans_per_subband / factor).astype(int)
        dynamic_spectra, _, looped_mask = mad_spectra_flat(
            dynamic_spectra,
            no_time_detrend=True,
            sigma=sigma,
            time_median_size=time_median_size,
            return_same_dtype=False,
            chans_per_subband=chans_per_subband,
            flatten_to=flatten_to,
        )
        loop_mask_percentage += looped_mask
    total_mask = mask_percentage + fft_mask_percentage + loop_mask_percentage
    logging.debug(
        "Inital Mad: %.2f, FFT %.2f, Loop Mask %.2f, Total Masked %.2f",
        mask_percentage,
        fft_mask_percentage,
        loop_mask_percentage,
        total_mask,
    )
    decimated = dynamic_spectra.mean(axis=1)
    return FilterResult(decimated, total_mask)


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
        savgol_sub_array = signal.savgol_filter(
            gulp[j : j + frame, :], window, 2, axis=0
        )
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
    savgol_array = signal.savgol_filter(gulp, window, 2, axis=1)
    for j in np.arange(0, len(gulp[1]) - frame + 1, frame):
        savgol_sub_array = signal.savgol_filter(
            savgol_array[:, j : j + frame], window, 2, axis=1
        )
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


def sum_threshold(
    dynamic_spectra: np.ndarray,
    mask: np.ndarray = None,
    eta_i: Union[List[float], Tuple[float]] = (0.5, 0.55, 0.62, 0.75, 1),
    chi_1: float = 35000,
    normalize_standing_waves: bool = True,
    suppress_dilation: bool = False,
    sm_kwargs: Dict = None,
    di_kwargs: Dict = None,
    max_pixels: int = 8,
) -> np.ndarray:
    """
    Computes a mask to cover the RFI in a data set.

    Args:
        dynamic_spectra - Array containing the signal and RFI

        chi_1 - First threshold

        eta_i - List of sensitivities

        max_pixels - Controls the max iteration and chi_1

    KWArgs:
        normalize_standing_waves - Whether to normalize standing waves

        suppress_dilation -  If true, mask dilation is suppressed

        plotting - True if statistics plot should be displayed

        sm_kwargs - Smoothing key words

        di_kwargs - dilation key words

    Returns:
        mask - the mask covering the identified RFI

    Note:
        From
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/mitigation/sum_threshold.py
        Cite https://arxiv.org/abs/1607.07443
    """

    if mask is None:
        mask = np.zeros_like(dynamic_spectra, dtype=bool)

    if sm_kwargs is None:
        sm_kwargs = sm.get_sm_kwargs()

    # if plotting: sum_threshold_utils.plot_moments(data)

    if normalize_standing_waves:
        dynamic_spectra = sm.normalize(dynamic_spectra, mask)

        # if plotting: sum_threshold_utils.plot_moments(data)

    p = 1.5  # pylint: disable=invalid-name
    pixel_range = np.arange(1, max_pixels)
    pixel_raised = 2 ** (pixel_range - 1)
    chi_i = chi_1 / p ** np.log2(pixel_range)

    st_mask = mask
    for eta in eta_i:
        st_mask = sm.run_sumthreshold(
            dynamic_spectra,
            st_mask,
            eta,
            pixel_raised,
            chi_i,
            sm_kwargs=sm_kwargs,
        )

    dilated_mask = st_mask
    if not suppress_dilation:
        if di_kwargs is None:
            di_kwargs = sm.get_di_kwargs()

        dilated_mask = sm.binary_mask_dilation(dilated_mask ^ mask, **di_kwargs)

        # if plotting:
        #     sum_threshold_utils.plot_dilation(st_mask, mask, dilated_mask)

    return dilated_mask + mask


def sum_threasthold_aprls(
    dynamic_spectra: np.ndarray,
) -> np.ndarray:
    """
    Computes a mask to cover the RFI in a data set based on ArPLS-ST.

    Args:
        dynamic_spectra - Array containing the signal and RFI

        eta_i - List of sensitivities

    Returns:
        2D mask where True = RFI

    Note:
        From http://zmtt.bao.ac.cn/GPPS/RFI/
        Cite: https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2969Z
    """
    # eta_i: Union[List[float], Tuple[float]] = (0.5, 0.55, 0.62, 0.75, 1),
    # Find bandpass and then use ArPLS to estimate
    # what the RFI bandpass lookslike
    freq_mean = dynamic_spectra.mean(axis=0)
    baseline = arpls_fitter(freq_mean, lam=100000)
    # compute the difference between SED curve and its baseline
    diff = freq_mean - baseline
    # compute the first threshold value for band RFI mitigation
    popt = sm.ksigma(diff)
    # band RFI mitigation
    line_mask = sm.run_sumthreshold_arpls(diff, 2 * popt)

    line_index = np.where(line_mask)[0]
    freq_mean[line_index] = baseline[line_index]
    valid_index = np.where(~line_mask)[0]

    valid_data = dynamic_spectra - freq_mean
    valid_data = valid_data[valid_index]

    # compute the first threshold value for blob RFI mitgation
    popt_point = sm.ksigma(valid_data)
    # blob RFI mitigation
    mask = sm.blob_mitigation(dynamic_spectra, freq_mean, line_mask, 5 * popt_point)
    mask[:, line_index] = True

    return mask


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


def skew_calculate_values(yr_file, window=64, time_median_kernel=0):
    """
    Calculate Skew for blocks of length window.

    Args:
        yr_file: The Your object for a FITS/fil file to process

        window: window length in number of time samples

        time_median_kernel: Detrend in time by subtracting a smoothed running
                            median, smoothed bt time_median_kernel

        bandpass_kernel: bandpass kernel to smooth bandpass

    Returns:
        Skew values for each block in the file
    """
    nspectra = yr_file.your_header.nspectra
    nchan = yr_file.your_header.nchans
    num_stat_samples = np.ceil(nspectra / window).astype(int)
    skew = np.zeros((num_stat_samples, nchan), dtype=np.float64)
    for j in track(range(num_stat_samples)):
        if j * window + window > nspectra:
            gulp = nspectra - j * window
        else:
            gulp = window
        chunk = yr_file.get_data(j * window, gulp)

        if time_median_kernel > 0:
            time_series = np.nanmean(chunk, axis=1)
            time_series = signal.medfilt(time_series, kernel_size=time_median_kernel)
            chunk = chunk - time_series[:, None]

        skew[j, :] = stats.skew(chunk, axis=0)

    return skew


def zero_dm(
    dynamic_spectra: np.ndarray,
    bandpass: np.ndarray = None,
    return_same_dtype: bool = True,
    intermediate_dtype: type = np.float32,
) -> FilterResult:
    """
    Mask-safe zero-dm subtraction

    args:
        dynamic_spectra: The data you want to zero-dm, expects times samples
                         on the vertical axis. Accepts numpy.ma.arrays.

        bandpass - Use if a large file is broken up into pieces.
                   Be careful about how you use this with masks.

        intermediate_dtype - The data type to do the calculations

        return_same_dtype: return the same data type as given

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
        https://github.com/SixByNine/sigproc/blob/28ba4f4539d41a8722c6ed194fa66e87bf4610fc/src/zerodm.c#L195

        https://sourceforge.net/p/heimdall-astro/code/ci/master/tree/Pipeline/clean_filterbank_rfi.cu

        https://github.com/scottransom/presto/blob/de2cf58262190d35fb37dbebf8308a6e29d72adf/src/zerodm.c

        https://github.com/thepetabyteproject/your/blob/1f4b39326835e6bb87e0003318b433dc1455a137/your/writer.py#L232

        https://sigpyproc3.readthedocs.io/en/latest/_modules/sigpyproc/Filterbank.html#Filterbank.removeZeroDM
    """
    data_type = dynamic_spectra.dtype

    time_series = np.ma.mean(dynamic_spectra, axis=1).astype(intermediate_dtype)
    if bandpass is None:
        bandpass = np.ma.mean(dynamic_spectra, axis=0)  # .astype(data_type)

    dynamic_spectra = dynamic_spectra - time_series[:, None] + bandpass
    percent_masked = 1 / len(bandpass) * 100

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, dtype=data_type)

    return FilterResult(dynamic_spectra, percent_masked)


def zero_dm_fft(
    dynamic_spectra: np.ndarray,
    bandpass: np.ndarray = None,
    modes_to_zero: int = 2,
    return_same_dtype: bool = True,
) -> FilterResult:
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
                         must be verticals

        bandpass - Bandpass to add. We subtract off the DC component, we
                   must add it back to safely write the data as unsigned ints
                   if no bandpass is given, this will use the bandpass from the
                   dynamic spectra given, this can cause jumps if you are processing
                   multiple chunks.

        modes_to_zero - The number of modes to filter, starting at the lowest mode

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

    if bandpass is None:
        bandpass = dynamic_spectra.mean(axis=0)

    data_type = dynamic_spectra.dtype

    dynamic_spectra_fftd = np.fft.rfft(dynamic_spectra, axis=1)

    # They FFT'd dynamic spectra will be 1/2 or 1/2+1 the size of
    # the dynamic spectra since FFT is complex
    mask = np.zeros(dynamic_spectra_fftd.shape[1], dtype=bool)
    mask[
        :modes_to_zero,
    ] = True

    # complex data, we are projecting two numbers
    # except for the first (DC) component
    percent_masked = (mask.mean() * 2) * 100
    logging.debug("Masked Percentage: %.2f %%", mask.mean() * 2 * 100)

    # zero out the modes we don't want
    dynamic_spectra_fftd[np.broadcast_to(mask, dynamic_spectra_fftd.shape)] = 0

    # Add the bandpass back so out values are in the correct range
    dynamic_spectra_cleaned = np.fft.irfft(dynamic_spectra_fftd, axis=1) + bandpass

    if return_same_dtype:
        dynamic_spectra_cleaned = to_dtype(dynamic_spectra_cleaned, dtype=data_type)

    return FilterResult(dynamic_spectra_cleaned, percent_masked)
