#!/usr/bin/env python3
"""
The repository for all my filters
"""
import logging

import numpy as np
from scipy import signal, stats
from scipy.signal import savgol_filter as sg

from jess.calculators import highpass_window
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

    `   threshhold: abs threshold to filter kurtosis values

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

        # thresh = np.tile(cut, (frame, 1)).T
        # medians = np.tile(medians, (frame, 1)).T
        # mask = np.abs(diff - medians) > thresh

        mask = np.abs(diff - medians[:, None]) > cut[:, None]

        logging.info("Masked Percentage: %.2f %%", mask.mean() * 100)

        try:  # sometimes this fails to converge, if happens use original fit
            masked_arr = np.ma.masked_array(gulp[:, j : j + frame], mask=mask)
            fit_clean = fitter(
                np.ma.median(masked_arr, axis=0),
                chans_per_fit=chans_per_fit,
            )
        except Exception as e:
            logging.warning("Failed to fit with Exception: %f, using original fit", e)
            fit_clean = fit

        np.clip(fit_clean, min_value, max_value, out=fit_clean)
        # clip the values so they don't wrap
        # when converted to ints
        fit_clean = fit_clean.astype(data_type)
        # convert to dtype of the original
        gulp[:, j : j + frame] = np.where(mask, fit_clean, gulp[:, j : j + frame])

    return gulp.astype(data_type)


def flattner(
    dynamic_spectra: np.ndarray, flatten_to: int = 64, kernel_size: int = 1
) -> np.ndarray:
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
        ts_medians = signal.medfilt(
            np.nanmedian(dynamic_spectra, axis=1), kernel_size=kernel_size
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_medians = signal.medfilt(
            np.nanmedian(dynamic_spectra, axis=0), kernel_size=kernel_size
        )
        return dynamic_spectra - spectra_medians + flatten_to

    ts_medians = np.nanmedian(dynamic_spectra, axis=1)
    # break up into two subtractions so the final number comes out where we want it
    dynamic_spectra = dynamic_spectra - ts_medians[:, None]
    spectra_medians = np.nanmedian(dynamic_spectra, axis=0)
    return dynamic_spectra - spectra_medians + flatten_to


def mad_spectra_flat(
    dynamic_spectra: np.ndarray,
    frame: int = 256,
    sigma: float = 3,
    flatten_to: int = 64,
    return_mask: bool = False,
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
    flattened = flattner(dynamic_spectra, flatten_to=flatten_to, kernel_size=7)
    mask = np.zeros_like(flattened, dtype=bool)

    for j in np.arange(0, len(dynamic_spectra[1]) - frame + 1, frame):

        cut = sigma * stats.median_abs_deviation(
            flattened[:, j : j + frame], axis=1, scale="Normal"
        )
        medians = np.median(flattened[:, j : j + frame], axis=1)
        mask[:, j : j + frame] = (
            np.abs(flattened[:, j : j + frame] - medians[:, None]) > cut[:, None]
        )

        flattened[:, j : j + frame][mask[:, j : j + frame]] = np.nan

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
        cut = sigma * stats.median_abs_deviation(
            flattened[:, j : j + frame], axis=1, scale="Normal"
        )

        medians = np.median(flattened[:, j : j + frame], axis=1)
        mask_new = np.abs(flattened[:, j : j + frame] - medians[:, None]) > cut[:, None]
        mask[:, j : j + frame] = mask[:, j : j + frame] + mask_new
        flattened[:, j : j + frame][mask[:, j : j + frame]] = np.nan
        flattened[:, j : j + frame] = flattner(
            flattened[:, j : j + frame], flatten_to=flatten_to, kernel_size=1
        )
        flattened[:, j : j + frame][mask[:, j : j + frame]] = flatten_to

    logging.info("Masking %.2f %%", mask.mean() * 100)

    np.clip(
        flattened, min_value, max_value, out=flattened
    )  # clip the values so they don't wrap when converted to ints

    if return_mask:
        return flattened.astype(data_type), mask
    return flattened.astype(data_type)


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


def mad_fft(
    gulp: np.ndarray,
    frame: int = 256,
    sigma: float = 3,
    chans_per_fit: int = 50,
    fitter: object = poly_fitter,
    bad_chans: np.ndarray = None,
    window_length: int = None,
    return_mask: bool = False,
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

    """
    frame = int(frame)
    data_type = gulp.dtype
    iinfo = np.iinfo(data_type)
    min_value = iinfo.min
    max_value = iinfo.max

    # if we run the highpass filter, we will
    # need to add the DC levels back in
    if window_length is not None:
        bandpass = gulp.mean(axis=0)

    gulp_fftd = np.fft.rfft(gulp, axis=0)
    gulp_fftd_abs = np.abs(gulp_fftd)
    mask = np.zeros_like(gulp_fftd_abs, dtype=bool)

    for j in np.arange(0, len(gulp_fftd_abs[1]) - frame + 1, frame):
        fit = fitter(
            np.median(gulp_fftd_abs[:, j : j + frame], axis=0),
            chans_per_fit=chans_per_fit,
        )  # .astype(data_type)

        diff = gulp_fftd_abs[:, j : j + frame] - fit
        cut = sigma * stats.median_abs_deviation(diff, axis=None, scale="Normal")
        # adds some resistance to jumps in medians
        medians = signal.medfilt(np.median(diff, axis=1), 7)
        mask[:, j : j + frame] = np.abs(diff - medians[:, None]) > cut

    # maybe some lekage into the nearby channels
    # but this doesn't seem to help much
    # mask = ndimage.binary_dilation(mask)

    # remove infomation for the bad channels, but leave power
    # this has no effect on the following filter
    # which works on gulp_fftd_abd
    if bad_chans is not None:
        logging.debug("Applying channel mask %s", bad_chans)
        mask[1:, bad_chans] = True

    if window_length is not None:
        window = highpass_window(window_length)
        gulp_fftd[:window_length] *= window[:, None]
    else:
        # Consider changing this
        mask[0, :] = False  # set the row to false to preserve the powser levels

    # zero masked values
    gulp_fftd[mask] = 0

    # We're flagging complex data, so multiply by 2
    logging.info("Masked Percentage: %.2f %%", mask.mean() * 100 * 2)

    gulp_cleaned = np.fft.irfft(gulp_fftd, axis=0)

    if window_length is not None:
        gulp_cleaned += bandpass

    np.clip(gulp_cleaned, min_value, max_value, out=gulp_cleaned)

    gulp_cleaned = gulp_cleaned.astype(data_type)

    if return_mask:
        return gulp_cleaned, mask

    return gulp_cleaned


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
        https://github.com/SixByNine/sigproc/blob/28ba4f4539d41a8722c6ed194fa66e87bf4610fc/src/zerodm.c#L195

        https://sourceforge.net/p/heimdall-astro/code/ci/master/tree/Pipeline/clean_filterbank_rfi.cu

        https://github.com/scottransom/presto/blob/de2cf58262190d35fb37dbebf8308a6e29d72adf/src/zerodm.c

        https://github.com/thepetabyteproject/your/blob/1f4b39326835e6bb87e0003318b433dc1455a137/your/writer.py#L232

        https://sigpyproc3.readthedocs.io/en/latest/_modules/sigpyproc/Filterbank.html#Filterbank.removeZeroDM
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


def zero_dm_fft(
    dynamic_spectra: np.ndarray,
    bandpass: np.ndarray = None,
    modes_to_zero: int = 2,
) -> np.ndarray:
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

    if bandpass is None:
        bandpass = dynamic_spectra.mean(axis=0)

    data_type = dynamic_spectra.dtype
    iinfo = np.iinfo(data_type)

    dynamic_spectra_fftd = np.fft.rfft(dynamic_spectra, axis=1)

    # They FFT'd dynamic spectra will be 1/2 or 1/2+1 the size of
    # the dynamic spectra since FFT is complex
    mask = np.zeros(dynamic_spectra_fftd.shape[1], dtype=bool)
    mask[
        :modes_to_zero,
    ] = True

    # complex data, we are projecting two numbers
    logging.info("Masked Percentage: %.2f %%", mask.mean() * 2 * 100)

    # zero out the modes we don't want
    dynamic_spectra_fftd[np.broadcast_to(mask, dynamic_spectra_fftd.shape)] = 0

    # Add the bandpass back so out values are in the correct range
    dynamic_spectra_cleaned = np.fft.irfft(dynamic_spectra_fftd, axis=1) + bandpass

    # clip so astype doesn't wrap
    np.clip(dynamic_spectra_cleaned, iinfo.min, iinfo.max, out=dynamic_spectra_cleaned)

    return np.round(dynamic_spectra_cleaned).astype(data_type)
