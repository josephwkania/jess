#!/usr/bin/env python3
"""
The repository for all calculators
"""

import logging
import random
import warnings
from typing import Callable, Tuple, Union

import numpy as np
from scipy import ndimage, signal, stats
from scipy.stats import entropy, median_abs_deviation
from your import Your

random.seed(2021)


def accumulate(data_array: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """
    Reduce the data along an axis by taking the mean of a 'factor' of rows along
    the axis

    args:
        data_array: array of data to be reduces

        factor: the factor to reduce the dimension by

        axis: axis to operate on

    returns:
        array with axis reduced by factor
    """
    if axis == 0:
        reshaped = data_array.reshape(
            data_array.shape[0] // factor, factor, data_array.shape[1]
        )
        return reshaped.sum(axis=1)
    if axis == 1:
        reshaped = data_array.reshape(
            data_array.shape[0], data_array.shape[1] // factor, factor
        )
        return reshaped.sum(axis=2)
    raise NotImplementedError(f"Asked for axis {axis} which is not available")


def autocorrelate(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Auto correlation along an axis

    Args:
        data: data to find the autocorrelation

        axis: axis to find the autocorrelation, -1, 0, 1 available

    Returns:
        Auto correlation along an axis

    Notes:
        Uses mean to flatten, if complex structure, should do a better
        detrend

    """
    if axis > 1:
        raise NotImplementedError(f"Not Available for axis {axis}")
    if axis == 1:
        data = data.T
        axis = 0
        transpose = True
    else:
        transpose = False

    data = data - data.mean(axis=axis)
    correlation = signal.fftconvolve(
        data, np.flip(data, axis=axis), mode="same", axes=axis
    )[len(data) // 2 :]
    correlation /= np.max(correlation, axis=axis)

    if transpose:
        return correlation.T
    return correlation


def closest_larger_factor(num: int, factor: int) -> int:
    """
    Find the closest factor that is larger than a number.

    args:
        num: The number of to find the largest factor

        factor: Factor to divide by

    returns:
        Closest factor of `factor` larger than `num`
    """
    return int(np.ceil(num / factor) * factor)


def mean(
    data_array: np.ndarray, factor: int, axis: int, pad: str = "median"
) -> np.ndarray:
    """
    Reduce the data along an axis by taking the mean of a 'factor' of rows along
    the axis

    args:
        data_array: array of data to be reduces

        factor: the factor to reduce the dimension by

        axis: axis to operate on

        pad: method to pad if axis is not divisible. If None
             will not pad

    returns:
        array with axis reduced by factor
    """
    if axis > 1:
        raise NotImplementedError(f"Asked for axis {axis} which is not available")

    axis_length = data_array.shape[axis]
    if axis_length % factor != 0 and pad is not None:
        new_length = closest_larger_factor(axis_length, factor)
        data_array = pad_along_axis(
            data_array, new_length=new_length, axis=axis, mode=pad
        )
        axis_length = new_length

    if axis == 0:
        reshaped = data_array.reshape(
            axis_length // factor, factor, data_array.shape[1]
        )
        reduced = reshaped.mean(axis=1)
    else:
        # axis == 1
        reshaped = data_array.reshape(
            data_array.shape[0], axis_length // factor, factor
        )
        reduced = reshaped.mean(axis=2)

    return reduced


def _contains_nan(a, nan_policy="propagate"):
    policies = ["propagate", "raise", "omit"]
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join("'%s'" % s for s in policies)
        )
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = "omit"
            warnings.warn(
                "The input array could not be properly "
                "checked for nan values. nan values "
                "will be ignored.",
                RuntimeWarning,
            )

    if contains_nan and nan_policy == "raise":
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy


def median_abs_deviation_med(
    x: np.ndarray,
    axis: int = 0,
    center: Callable = np.median,
    scale: float = 1.0,
    nan_policy: str = "propagate",
):
    r"""
    Compute the median absolute deviation of the data along the given axis.
    The median absolute deviation (MAD, [1]_) computes the median over the
    absolute deviations from the median. It is a measure of dispersion
    similar to the standard deviation but more robust to outliers [2]_.
    The MAD of an empty array is ``np.nan``.
    .. versionadded:: 1.5.0
    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the
        function signature ``func(arr, axis)``.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The default is 1.0. The string "normal" is also accepted,
        and results in `scale` being the inverse of the standard normal
        quantile function at 0.75, which is approximately 0.67449.
        Array-like scale is also allowed, as long as it broadcasts correctly
        to the output such that ``out / scale`` is a valid operation. The
        output dimensions depend on the input array, `x`, and the `axis`
        argument.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    mad : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.

    center : The centers of each array
    See Also
    --------
    numpy.std, numpy.var, numpy.median, scipy.stats.iqr, scipy.stats.tmean,
    scipy.stats.tstd, scipy.stats.tvar
    Notes
    -----
    Modified from scipy.stats.median_abs_devation

    The `center` argument only affects the calculation of the central value
    around which the MAD is calculated. That is, passing in ``center=np.mean``
    will calculate the MAD around the mean - it will not calculate the *mean*
    absolute deviation.
    The input array may contain `inf`, but if `center` returns `inf`, the
    corresponding MAD for that data will be `nan`.
    References
    ----------
    .. [1] "Median absolute deviation",
           https://en.wikipedia.org/wiki/Median_absolute_deviation
    .. [2] "Robust measures of scale",
           https://en.wikipedia.org/wiki/Robust_measures_of_scale
    Examples
    --------
    When comparing the behavior of `median_abs_deviation` with ``np.std``,
    the latter is affected when we change a single value of an array to have an
    outlier value while the MAD hardly changes:
    >>> from scipy import stats
    >>> x = stats.norm.rvs(size=100, scale=1, random_state=123456)
    >>> x.std()
    0.9973906394005013
    >>> stats.median_abs_deviation(x)
    0.82832610097857
    >>> x[0] = 345.6
    >>> x.std()
    34.42304872314415
    >>> stats.median_abs_deviation(x)
    0.8323442311590675
    Axis handling example:
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> stats.median_abs_deviation(x)
    array([3.5, 2.5, 1.5])
    >>> stats.median_abs_deviation(x, axis=None)
    2.0
    Scale normal example:
    >>> x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    >>> stats.median_abs_deviation(x)
    1.3487398527041636
    >>> stats.median_abs_deviation(x, scale='normal')
    1.9996446978061115
    """
    if not callable(center):
        raise TypeError(
            "The argument 'center' must be callable. The given "
            f"value {repr(center)} is not callable."
        )

    # An error may be raised here, so fail-fast, before doing lengthy
    # computations, even though `scale` is not used until later
    if isinstance(scale, str):
        if scale.lower() == "normal":
            scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = np.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return np.nan, np.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return np.nan, np.nan
        return np.full(nan_shape, np.nan), np.nan

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan:
        if axis is None:
            mad = stats._stats_py._mad_1d(x.ravel(), center, nan_policy)
            centers = center(x.ravel())
        else:
            mad = np.apply_along_axis(
                stats._stats_py._mad_1d, axis, x, center, nan_policy
            )
            centers = center(x, axis=axis)
    else:
        if axis is None:
            centers = center(x, axis=None)
            mad = np.median(np.abs(x - centers))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            centers = center(x, axis=axis)
            med = np.expand_dims(centers, axis)
            mad = np.median(np.abs(x - med), axis=axis)

    return mad / scale, centers


def decimate(
    dynamic_spectra: np.ndarray,
    time_factor: int = None,
    freq_factor: int = None,
    backend: Callable = signal.decimate,
) -> np.ndarray:
    """
    Makes decimates along either/both time and frequency axes.
    Flattens data along frequency before frequency decimation.
    Fattens again in frequency before returning.

    args:
        dynamic_spectra: dynamic spectra with time on the ventricle axis

        time_factor: factor to reduce time sampling by

        freq_factor: factor to reduce frequency channels

        backend: backend to use to reduce the dimension, default is
                 signal.decimate, consider using jess.calculator.mean

    returns:
        Flattened in frequency dynamic spectra, reduced in time and/or frequency
    """
    if time_factor is not None:
        if not isinstance(time_factor, int):
            time_factor = int(time_factor)
        logging.debug("time_factor was not an int: now is %i", time_factor)
        dynamic_spectra = backend(dynamic_spectra, time_factor, axis=0)
    if freq_factor is not None:
        if not isinstance(freq_factor, int):
            freq_factor = int(freq_factor)
        logging.debug("freq_factor was not an int: now is %i", freq_factor)
        dynamic_spectra = backend(
            dynamic_spectra - np.median(dynamic_spectra, axis=0), freq_factor, axis=1
        )
    return dynamic_spectra - np.median(dynamic_spectra, axis=0)


def flattner_median(
    dynamic_spectra: np.ndarray,
    flatten_to: int = 64,
    kernel_size: int = 0,
    return_same_dtype: bool = False,
    return_time_series: bool = False,
    intermediate_dtype: type = np.float32,
) -> Union[np.ndarray, Tuple]:
    """
    This flattens the dynamic spectra by subtracting the medians of the time series
    and then the medians of the of bandpass. Then add flatten_to to all the pixels
    so that the data can be keep as the same data type.

    args:
        dynamic_spectra: The dynamic spectra you want to flatten

        flatten_to: The number to set as the baseline

        kernel_size: The size of the median filter to run over the medians
                     0,1 => nothing is applied

        intermediate_dtype: the data type of the intermediate calculation

        return_same_dtype: return the same data type as dynamic_spectra

        return_time_series: return the time series difference from flatten_to
                            median_time_series - flatten_to

    returns:
        Dynamic spectra flattened in frequency and time
        (optional) time series median
    """
    original_dtype = dynamic_spectra.dtype
    ts_medians = np.nanmedian(dynamic_spectra, axis=1).astype(intermediate_dtype)
    ts_medians -= flatten_to

    if kernel_size > 1:
        ndimage.median_filter(
            ts_medians, size=kernel_size, mode="mirror", output=ts_medians
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_medians = ndimage.median_filter(
            np.nanmedian(dynamic_spectra, axis=0).astype(intermediate_dtype),
            size=kernel_size,
        )
    else:
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_medians = np.nanmedian(dynamic_spectra, axis=0).astype(
            intermediate_dtype
        )

    spectra_medians -= flatten_to
    dynamic_spectra -= spectra_medians

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, original_dtype)

    if return_time_series:
        return dynamic_spectra, ts_medians
    return dynamic_spectra


def flattner_mix(
    dynamic_spectra: np.ndarray,
    flatten_to: int = 64,
    kernel_size: int = 0,
    return_same_dtype: bool = False,
    return_time_series: bool = False,
    intermediate_dtype: type = np.float32,
) -> Union[np.ndarray, Tuple]:
    """
    This flattens the dynamic spectra by subtracting the medians of the time series
    and then the medians of the of bandpass. Then add flatten_to to all the pixels
    so that the data can be keep as the same data type.

    This uses medians subtraction on the time series. This is less aggressive and
    leaved the mean subtraction for the zero-dm.

    Mean subtraction across the spectrum allows for smoother transition between blocks.

    args:
        dynamic_spectra: The dynamic spectra you want to flatten

        flatten_to: The number to set as the baseline
                    0,1 => nothing is applied

        kernel_size: The size of the median filter to run over the medians

        return_same_dtype: return the same data type as given, else it will
                           at least np.int64

        return_time_series: return the time series median differences from flatten_to
                            median_time_series - flatten_to

    returns:
        Dynamic spectra flattened in frequency and time
        (optional) time series medians
    """
    original_dtype = dynamic_spectra.dtype
    ts_medians = np.nanmedian(dynamic_spectra, axis=1).astype(intermediate_dtype)
    ts_medians -= flatten_to

    if kernel_size > 1:
        ndimage.median_filter(
            ts_medians, size=kernel_size, mode="mirror", output=ts_medians
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]

        spectra_means = ndimage.median_filter(
            np.nanmedian(dynamic_spectra, axis=0).astype(intermediate_dtype),
            size=kernel_size,
            mode="mirror",
        )
    else:
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_means = np.nanmean(dynamic_spectra, axis=0)

    spectra_means -= flatten_to
    dynamic_spectra -= spectra_means

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, dtype=original_dtype)

    if return_time_series:
        return dynamic_spectra, ts_medians
    return dynamic_spectra


def highpass_window(window_length: int) -> np.ndarray:
    """
    Calculates the coefficients to multiply the Fourier components
    to make a highpass filter.

    Args:
        window_length: the length of the half window

    Returns:
        Half of an inverted blackman window, will bw window_length long
    """
    return 1 - np.blackman(2 * window_length)[window_length:]


def guassian_noise_adder(standard_deviations: np.ndarray) -> float:
    """
    Add Gaussian noise from multiple channels to estimate the time series

    Args:
        standard_deviation: Standard Deviations along the bandpass

    Returns:
        time series standard deviation

    Notes:
        https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables#Independent_random_variables

        Variances add at the sum of squares, we are interested in Standard deviation,
        so talk the square root.

        We are taking the mean, do divide by the number of channels
        (=number of standard deviations)
    """
    squares = standard_deviations**2
    summed = np.sum(squares)
    return np.sqrt(summed) / len(standard_deviations)


def noise_calculator(
    yr_obj: Your,
    num_samples: int = 1000,
    len_block: int = 128,
    detrend: bool = True,
    kernel_size: int = 17,
) -> Tuple[np.ndarray, np.float64]:
    """
    Calculate the ideal noise of a file

    1) Get num_samples random start samples of len_block

    2) Optionally detrend the data in time to remove power level trends

    3) Use a robust measure to estimate the noise in each channel
       Here we use IQR that is then scared

    4) Optionally use a Median filter to filter channels that are
       contaminated at at the 25th to 75th level

    5) Add up these standard deviations to find the time series standard
       deviations

    Args:
        yr_object: your object for the file to process

        num_samples: number of samples to take

        detrend: detrend along the time axis before finding the dispersion

        kernel_size: the kernel size for the median filter. If 0 or 1
                     no filter is applied

    Returns:
        Ideal standard deviation of time series,
        Real standard deviation of the time series

    Notes:
        detrend would be useful if there are power level changes in the data. This will
        cause the Standard deviation to be to high. This will remove random trends
        as well, so this could lead to underestimating the noise level

        IQR and MAD do a reasonable job at estimating the dispersion of a channel
        as long as the channel is not filled with RFI. If there is RFI power
        at all the percentiles this will cause the IQR to be too high,
        so we use a median filter to remove these channels.
        By using smallish blocks, we hope to avoid trends that can't
        be detrened linearly

        By calculating all the channels independently, then adding the
        standard deviations we should avoid correlated (zero dm)
        noise that shows up across off the channels

        Calculate the standard deviation of the zero dm time series by collapsing
        all the channels for each random block. Then take the median of all these
        standard deviations.

    """
    start_sample_range = yr_obj.your_header.nspectra - len_block
    starts = [random.randrange(0, start_sample_range) for _ in range(num_samples)]
    iqrs = np.zeros((num_samples, yr_obj.your_header.nchans), dtype=np.float64)
    zero_dm_stds = np.zeros(num_samples, dtype=np.float64)

    for j, jstart in enumerate(starts):
        data = yr_obj.get_data(jstart, len_block)

        if detrend:
            # detrend in time
            signal.detrend(data, axis=0, overwrite_data=True)

        iqr = stats.iqr(data, axis=0, scale="normal")

        if kernel_size > 1:
            iqr = ndimage.median_filter(iqr, size=kernel_size, mode="nearest")

        iqrs[j] = iqr
        zero_dm_stds[j] = np.std(data.mean(axis=1))

    ideal_noise = np.median(iqrs, axis=0)
    zero_dm_noise = np.median(zero_dm_stds)

    return ideal_noise, zero_dm_noise


def pad_along_axis(
    array: np.ndarray,
    new_length: int,
    axis: int = 0,
    mode: str = "median",
    location: str = "middle",
):
    """
    Pad along an axis.

    args:
        array: Array to pad

        new_length: New length of the axis

        axis: Axis to be padded

        mode: mode to pad, see numpy.pad

        location: Location of the pad. Options are
                  [end, start, middle]

        return:
            Array padded along `axis`

    Based on
    https://stackoverflow.com/a/49766444
    """

    pad_size = new_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    location = location.casefold()
    if location == "end":
        npad[axis] = (0, pad_size)
    elif location == "start":
        npad[axis] = (pad_size, 0)
    elif location == "middle":
        start = np.ceil(pad_size / 2).astype(int)
        end = np.floor(pad_size / 2).astype(int)
        npad[axis] = (start, end)

    return np.pad(array, pad_width=npad, mode=mode)


def preprocess(
    data: np.ndarray, central_value_calc: str = "mean", disperion_calc: str = "std"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the array for later statistical tests

    Args:
        data: 2D dynamic spectra to process

        central_value_calc: The method to calculate the central value,

        dispersion_calc: The method to calculate the dispersion

    Returns:
        data preprocessed along axis=0, axis=1
    """
    central_value_calc = central_value_calc.lower()
    disperion_calc = disperion_calc.lower()
    if central_value_calc == "mean":
        central_0 = np.mean(data, axis=0)
        central_1 = np.mean(data, axis=1)
    elif central_value_calc == "median":
        central_0 = np.median(data, axis=0)
        central_1 = np.median(data, axis=1)
    else:
        raise NotImplementedError(
            f"Given {central_value_calc} for the cental value calculator"
        )

    if disperion_calc == "std":
        dispersion_0 = np.std(data, axis=0, ddof=1)
        dispersion_1 = np.std(data, axis=1, ddof=1)
    elif disperion_calc == "mad":
        dispersion_0 = median_abs_deviation(data, axis=0, scale="normal")
        dispersion_1 = median_abs_deviation(data, axis=1, scale="normal")
    else:
        raise NotImplementedError(f"Given {disperion_calc} for dispersion calculator")

    return (
        (data - central_0) / dispersion_0,
        (data - central_1[:, None]) / dispersion_1[:, None],
    )


def shannon_entropy(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculates the Shannon Entropy along a given axis.

    Return entropy in natural units.

    Args:
        data: 2D Array to calculate entropy

        axis: axis to calculate entropy

    Returns:
        Shannon Entropy along an axis
    """
    if axis == 0:
        data = data.T
    elif axis > 1:
        raise ValueError(f"Axis out of bounds, given axis={axis}")
    length, _ = data.shape

    entropies = np.zeros(length, dtype=float)
    # Need to loop because np.unique doesn't
    # return counts for all
    for j in range(0, length):
        _, counts = np.unique(
            data[
                j,
            ],
            return_counts=True,
        )
        entropies[j] = entropy(counts)
    return entropies


def balance_chans_per_subband(
    num_chans: int, chans_per_subband: int
) -> Tuple[int, np.ndarray]:
    """
    Balance chan_per_subband when they are not evenly dividable

    Args:
        num_chans: total number of channels

        num_subbands: number of subbands

    Return:
        [number of sections, array with start and stops of each of the subband]
    """
    num_sections = num_chans // chans_per_subband
    if num_sections == 0:
        logging.debug("chans_per_subband > num_chans")
        return 1, np.array((0, num_chans))
    return num_sections, divide_range(num_chans, num_sections)


def divide_range(length: int, num_sections: int) -> np.ndarray:
    """
    Divide range as evenly as possible.

    Args:
        length: length of array

        num_sections: number of sections to divide array

    Return:
        array with start and stops of each of the subsections

    Note:
        Adapted from numpy.lib.shape_base.array_split
        and subject to the numpy license
    """
    neach_section, extras = divmod(length, num_sections)
    section_sizes = (
        [0] + extras * [neach_section + 1] + (num_sections - extras) * [neach_section]
    )
    return np.array(section_sizes, dtype=int).cumsum()


def to_dtype(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Takes a chunk of data and changes it to a given data type.

    Args:
        data: Array that you want to convert

        dtype: The output data type

    Returns:
        data converted to dtype
    """
    iinfo = np.iinfo(dtype)

    # Round the data
    np.around(data, out=data)

    # Clip to stop wrapping
    np.clip(data, iinfo.min, iinfo.max, out=data)

    return data.astype(dtype)


def get_flatten_to(nbits: int) -> int:
    """
    Get the flatten to number of a given number of bits.
    Sets the data 1/4 of the way to zero.

    Args:
        nbits - Number of bits.

    Returns:
        Number to flatten to.
    """
    if nbits < 4:
        raise NotImplementedError("{nbints} not implmented!")
    return 2 ** (nbits - 2)
