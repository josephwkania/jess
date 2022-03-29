#!/usr/bin/env python3
"""
The repository for all calculators
"""

import logging
from typing import Callable, Tuple, Union

import cupy as cp
from cupyx.scipy import ndimage

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201


def closest_larger_factor(num: int, factor: int) -> int:
    """
    Find the closest factor that is larger than a number.

    args:
        num: The number of to find the largest factor

        factor: Factor to divide by

    returns:
        Closest factor of `factor` larger than `num`
    """
    return int(cp.ceil(num / factor) * factor)


def mean(
    data_array: cp.ndarray, factor: int, axis: int, pad: str = "median"
) -> cp.ndarray:
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


def decimate(
    dynamic_spectra: cp.ndarray,
    time_factor: int = None,
    freq_factor: int = None,
    backend: Callable = mean,
) -> cp.ndarray:
    """
    Makes decimates along either/both time and frequency axes.
    Flattens data along frequency before freqency decimation.
    Fattens again in frequency before returning.

    args:
        dynamic_spectra: dynamic spectra with time on the ventricle axis

        time_factor: factor to reduce time sampling by

        freq_factor: factor to reduce freqency channels

    returns:
        Flattened in frequency dynamic spectra, reduced in time and/or freqency

    notes:
        Always uses jess.calculator_cupy.mean to add data
    """
    if time_factor is not None:
        if not isinstance(time_factor, int):
            time_factor = int(time_factor)
        logging.warning("time_factor was not an int: now is %i", time_factor)
        dynamic_spectra = mean(dynamic_spectra, time_factor, axis=0)
    if freq_factor is not None:
        if not isinstance(freq_factor, int):
            freq_factor = int(freq_factor)
        logging.warning("freq_factor was not an int: now is %i", freq_factor)
        dynamic_spectra = backend(
            dynamic_spectra - cp.median(dynamic_spectra, axis=0), freq_factor, axis=1
        )
    return dynamic_spectra - cp.median(dynamic_spectra, axis=0)


def flattner_median(
    dynamic_spectra: cp.ndarray,
    flatten_to: int = 64,
    kernel_size: int = 0,
    return_same_dtype: bool = False,
    return_time_series: bool = False,
    intermediate_dtype: type = cp.float32,
) -> Union[cp.ndarray, Tuple]:
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
        (optional) time series medians
    """
    original_dtype = dynamic_spectra.dtype
    spectra_medians = cp.nanmedian(dynamic_spectra, axis=0).astype(intermediate_dtype)
    spectra_medians -= flatten_to

    if kernel_size > 1:
        ndimage.median_filter(
            spectra_medians, size=kernel_size, mode="mirror", output=spectra_medians
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - spectra_medians
        ts_medians = ndimage.median_filter(
            cp.nanmedian(dynamic_spectra, axis=1).astype(intermediate_dtype),
            size=kernel_size,
        )
    else:
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - spectra_medians
        ts_medians = cp.nanmedian(dynamic_spectra, axis=1).astype(intermediate_dtype)

    ts_medians -= flatten_to
    result = dynamic_spectra - ts_medians[:, None]

    if return_same_dtype:
        result = to_dtype(result, original_dtype)

    if return_time_series:
        return result, ts_medians
    return result


def flattner_mix(
    dynamic_spectra: cp.ndarray,
    flatten_to: int = 64,
    kernel_size: int = 0,
    return_same_dtype: bool = False,
    return_time_series: bool = False,
    intermediate_dtype: type = cp.float32,
) -> Union[cp.ndarray, Tuple]:
    """
    This flattens the dynamic spectra by subtracting the medians of the time series
    and then the medians of the of bandpass. Then add flatten_to to all the pixels
    so that the data can be keep as the same data type.

    This uses medians subtraction on the time series. This is less agressive and
    leaved the mean subtraction for the zero-dm.

    Mean subtraction across the spectrum allows for smoother transition between blocks.

    args:
        dynamic_spectra: The dynamic spectra you want to flatten

        flatten_to: The number to set as the baseline

        kernel_size: The size of the median filter to run over the medians
                     0,1 => nothing is applied

        return_same_dtype: return the same data type as given, else it will
                           at least np.int64

        return_time_series: return the time series median differences from flatten_to
                            median_time_series - flatten_to

    returns:
        Dynamic spectra flattened in frequency and time
        (optional) time series medians
    """
    original_dtype = dynamic_spectra.dtype
    ts_medians = cp.nanmedian(dynamic_spectra, axis=1).astype(intermediate_dtype)
    ts_medians -= flatten_to

    if kernel_size > 1:
        ndimage.median_filter(
            ts_medians, size=kernel_size, mode="mirror", output=ts_medians
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]

        spectra_means = ndimage.median_filter(
            cp.nanmedian(dynamic_spectra, axis=0).astype(intermediate_dtype),
            size=kernel_size,
            mode="mirror",
        )
    else:
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_means = cp.nanmean(dynamic_spectra, axis=0)

    spectra_means -= flatten_to
    dynamic_spectra -= spectra_means

    if return_same_dtype:
        dynamic_spectra = to_dtype(dynamic_spectra, dtype=original_dtype)

    if return_time_series:
        return dynamic_spectra, ts_medians
    return dynamic_spectra


def pad_along_axis(
    array: cp.ndarray,
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
        start = cp.ceil(pad_size / 2).astype(int)
        end = cp.floor(pad_size / 2).astype(int)
        npad[axis] = (start, end)

    return cp.pad(array, pad_width=npad, mode=mode)


def to_dtype(data: cp.ndarray, dtype: cp.dtype) -> cp.ndarray:
    """
    Takes a chunk of data and changes it to a given data type.

    Args:
        data: Array that you want to convert

        dtype: The output data type

    Returns:
        data converted to dtype
    """
    iinfo = cp.iinfo(dtype)

    # Round the data
    cp.around(data, out=data)

    # Clip to stop wrapping
    cp.clip(data, iinfo.min, iinfo.max, out=data)

    return data.astype(dtype)
