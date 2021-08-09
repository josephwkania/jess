#!/usr/bin/env python3
"""
This contains cupy versions of some of JESS_filters
 """
import logging

import cupy as cp
import numpy as np
from cupyx.scipy.signal import medfilt

# from jess.fitters import poly_fitter
from jess.fitters_cupy import poly_fitter
from jess.scipy_cupy.stats import median_abs_deviation_gpu


def fft_mad(
    gulp: cp.ndarray,
    frame: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
    bad_chans: np.ndarray = None,
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
       gulp: a dynamic with time on the vertical axis,
       and freq on the horizontal

       frame (int): number of frequency samples to calculate MAD, i.e. the
                    channels per subband

       sigma (float): cutoff sigma

       chans_per_fit (int): polynomial/spline knots per channel to fit the bandpass

       fitter: which fitter to use, see jess.fitters for options

        bad_chans: list of bad channels - these have all information
                  removed except for the power

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
    frame = int(frame)
    data_type = gulp.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    # gulp = cp.asarray(gulp)
    gulp_fftd = cp.fft.rfft(gulp, axis=0)
    gulp_fftd_abs = cp.abs(gulp_fftd)
    mask = cp.zeros_like(gulp_fftd_abs, dtype=bool)

    for j in np.arange(0, len(gulp_fftd_abs[1]) - frame + 1, frame):
        fit = fitter(
            cp.median(gulp_fftd_abs[:, j : j + frame], axis=0),
            chans_per_fit=chans_per_fit,
        )  # .astype(data_type)
        diff = gulp_fftd_abs[:, j : j + frame] - cp.array(fit)
        cut = sigma * median_abs_deviation_gpu(diff, axis=None, scale="Normal")
        medians = cp.median(diff, axis=1)
        # adds some resistance to jumps in medians
        medians = medfilt(medians, 7)

        mask[:, j : j + frame] = cp.abs(diff - medians[:, None]) > cut

    # remove infomation for the bad channels, but leave power
    # this has no effect on the following filter
    # which works on gulp_fftd_abd
    if bad_chans is not None:
        logging.debug("Applying channel mask %s", bad_chans)
        mask[1:, bad_chans] = True

    mask[0, :] = False  # set the row to false to preserve the powser levels
    gulp_fftd[mask] = 0

    # We're flagging complex data, so multiply by 2
    logging.info("Masked Percentage: %.2f %%", mask.mean() * 100 * 2)

    gulp_cleaned = cp.fft.irfft(gulp_fftd, axis=0)

    cp.clip(gulp_cleaned, min_value, max_value, out=gulp_cleaned)

    gulp_cleaned = gulp_cleaned.astype(data_type)

    if return_mask:
        return gulp_cleaned, mask

    return gulp_cleaned


def mad_spectra(
    dynamic_spectra: cp.ndarray,
    frame: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
) -> cp.ndarray:
    """
    Calculates Median Absolute Deviations along the spectral axis
    (i.e. for each time sample across all channels)

    Args:
       gulp: a dynamic with time on the vertical axis, and freq on the horizontal

       frame (int): number of frequency samples to calculate the MAD

       sigma (float): cutoff sigma

       poly_order (int): polynomial order to fit for the bandpass

    Returns:

       Dynamic Spectrum with values clipped

    Notes:
        This version differs from the cpu version.
        This uses nans, while cpu version uses np.ma mask,
        excision performance is about the same. See the cpu docstring for
        references.

        You should use spectral_mad_flat (which has better RFI removal) unless
        you really need to preserve the bandpass.
    """
    frame = int(frame)
    data_type = dynamic_spectra.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    mask = cp.zeros_like(dynamic_spectra, dtype=bool)

    for j in np.arange(0, len(dynamic_spectra[1]) - frame + 1, frame):
        fit = poly_fitter(
            cp.median(dynamic_spectra[:, j : j + frame], axis=0),
            chans_per_fit=chans_per_fit,
        )
        # .astype(data_type)

        diff = dynamic_spectra[:, j : j + frame] - fit
        cut = sigma * median_abs_deviation_gpu(diff, axis=1, scale="Normal")
        medians = cp.median(diff, axis=1)
        mask[:, j : j + frame] = cp.abs(diff - medians[:, None]) < cut[:, None]

        try:  # sometimes this fails to converge, if happens use original fit
            fit_clean = poly_fitter(
                cp.nanmedian(
                    cp.where(
                        mask[:, j : j + frame],
                        dynamic_spectra[:, j : j + frame],
                        np.nan,
                    ),
                    axis=0,
                ),
                chans_per_fit=chans_per_fit,
            )
        except Exception as e:
            logging.warning("Failed to fit with Exception: %s, using original fit", e)
            fit_clean = fit
        cp.clip(
            fit_clean, min_value, max_value, out=fit_clean
        )  # clip the values so they don't wrap when converted to ints
        dynamic_spectra[:, j : j + frame] = cp.where(
            mask[:, j : j + frame],
            dynamic_spectra[:, j : j + frame],
            fit_clean,
        )

    logging.info("Masking %.2f %%", (1 - mask.mean()) * 100)
    return dynamic_spectra.astype(data_type)


def flattner(
    dynamic_spectra: cp.ndarray, flatten_to: int = 64, kernel_size: int = 1
) -> cp.ndarray:
    """
    This flattens the dynamic spectra by subtracting the medians of the time series
    and then the medians of the of bandpass. Then add flatten_to to all the pixels
    so that the data can be keep as the same data type.

    args:
        dynamic_spectra: The dynamic spectra you want to flatten

        flatten_to: The number to set as the baseline

        kernel_size: The size of the median filter to run over the medians

    returns:
        Dynamic spectra flattened in frequency and time
    """
    if kernel_size > 1:
        ts_medians = medfilt(
            cp.nanmedian(dynamic_spectra, axis=1), kernel_size=kernel_size
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_medians = medfilt(
            cp.nanmedian(dynamic_spectra, axis=0), kernel_size=kernel_size
        )
        return dynamic_spectra - spectra_medians + flatten_to

    ts_medians = cp.nanmedian(dynamic_spectra, axis=1)
    # break up into two subtractions so the final number comes out where we want it
    dynamic_spectra = dynamic_spectra - ts_medians[:, None]
    spectra_medians = cp.nanmedian(dynamic_spectra, axis=0)
    return dynamic_spectra - spectra_medians + flatten_to


def mad_spectra_flat(
    dynamic_spectra: cp.ndarray,
    frame: int = 256,
    sigma: float = 3,
    flatten_to: int = 64,
    return_mask: bool = False,
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

       frame: number of channels to calculate the MAD

       sigma: sigma which to reject outliers

       flatten_to: the median of the output data

    Returns:
       Dynamic Spectrum with values clipped

    See:
        https://github.com/rohinijoshi06/mad-filter-gpu

        Kendrick Smith's talks about CHIME FRB

    Note:
        This has better performance than spectral_mad, you should probably use this one.
    """
    frame = int(frame)
    data_type = dynamic_spectra.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    if not min_value < flatten_to < max_value:
        raise ValueError(
            f"""Can't flatten {data_type}, which has a range
            [{min_value}, {max_value}, to {flatten_to}"""
        )

    # I medfilt to try and stabalized the subtraction process against large RFI spikes
    # I choose 7 empirically
    flattened = flattner(
        cp.asarray(dynamic_spectra), flatten_to=flatten_to, kernel_size=7
    )
    mask = cp.zeros_like(flattened, dtype=bool)

    for j in np.arange(0, len(dynamic_spectra[1]) - frame + 1, frame):

        cut = sigma * median_abs_deviation_gpu(
            flattened[:, j : j + frame], axis=1, scale="Normal"
        )
        medians = cp.median(flattened[:, j : j + frame], axis=1)
        mask[:, j : j + frame] = (
            cp.abs(flattened[:, j : j + frame] - medians[:, None]) > cut[:, None]
        )

        flattened[:, j : j + frame][mask[:, j : j + frame]] = cp.nan

    # want kernel size to be 1, so every channel get set,
    # now that we've removed the worst RFI
    flattened = flattner(flattened, flatten_to=flatten_to, kernel_size=1)
    # set the masked values to what we want to flatten to
    # not obvus why this has to be done, because nans should be ok
    # but it works better this way
    flattened[mask] = flatten_to

    for j in np.arange(0, len(dynamic_spectra[1]) - frame + 1, frame):
        # Second iteration
        # flattened[:, j : j + frame] = flattner(
        #    flattened[:, j : j + frame], flatten_to=flatten_to, kernel_size=7
        # )
        cut = sigma * median_abs_deviation_gpu(
            flattened[:, j : j + frame], axis=1, scale="Normal"
        )

        medians = cp.median(flattened[:, j : j + frame], axis=1)
        mask_new = cp.abs(flattened[:, j : j + frame] - medians[:, None]) > cut[:, None]
        mask[:, j : j + frame] = mask[:, j : j + frame] + mask_new
        flattened[:, j : j + frame][mask[:, j : j + frame]] = cp.nan
        flattened[:, j : j + frame] = flattner(
            flattened[:, j : j + frame], flatten_to=flatten_to, kernel_size=1
        )
        flattened[:, j : j + frame][mask[:, j : j + frame]] = flatten_to

    logging.info("Masking %.2f %%", mask.mean() * 100)

    cp.clip(
        flattened, min_value, max_value, out=flattened
    )  # clip the values so they don't wrap when converted to ints

    if return_mask:
        return flattened.astype(data_type), mask
    return flattened.astype(data_type)


def zero_dm_fft(
    dynamic_spectra: cp.ndarray,
    bandpass: cp.ndarray = None,
    modes_to_zero: int = 2,
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
    iinfo = np.iinfo(data_type)
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

    logging.info("Masked Percentage: %.2f %%", mask.mean())

    # zero out the modes we don't want
    dynamic_spectra_fftd[cp.broadcast_to(mask, dynamic_spectra_fftd.shape)] = 0

    # Add the bandpass back so out values are in the correct range
    dynamic_spectra_cleaned = cp.fft.irfft(dynamic_spectra_fftd, axis=1) + bandpass

    # clip so astype doesn't wrap
    cp.clip(dynamic_spectra_cleaned, iinfo.min, iinfo.max, out=dynamic_spectra_cleaned)

    return cp.around(dynamic_spectra_cleaned).astype(data_type)
