#!/usr/bin/env python3
"""
The repository for all calculators
"""

import logging

import cupy as cp
from cupyx.scipy.signal import medfilt


def mean(data_array: cp.ndarray, factor: int, axis: int) -> cp.ndarray:
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
        return reshaped.mean(axis=1)
    if axis == 1:
        reshaped = data_array.reshape(
            data_array.shape[0], data_array.shape[1] // factor, factor
        )
        return reshaped.mean(axis=2)
    raise NotImplementedError(f"Asked for axis {axis} which is not available")


def decimate(
    dynamic_spectra: cp.ndarray,
    time_factor: int = None,
    freq_factor: int = None,
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
        dynamic_spectra = mean(
            dynamic_spectra - cp.median(dynamic_spectra, axis=0), freq_factor, axis=1
        )
    return dynamic_spectra - cp.median(dynamic_spectra, axis=0)


def flattner_median(
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


def flattner_mix(
    dynamic_spectra: cp.ndarray, flatten_to: int = 64, kernel_size: int = 1
) -> cp.ndarray:
    """
    This flattens the dynamic spectra by subtracting the medians of the time series
    and then the means of the of bandpass. Then add flatten_to to all the pixels
    so that the data can be keep as the same data type.

    This uses medians subtraction on the time series. This is less agressive and
    leaved the mean subtraction for the zero-dm.

    Mean subtraction across the spectrum allows for smoother transition between blocks.

    args:
        dynamic_spectra: The dynamic spectra you want to flatten

        flatten_to: The number to set as the baseline

        kernel_size: The size of the median filter to run over the medians

    returns:
        Dynamic spectra flattened with mean in frequency and median in time
    """
    if kernel_size > 1:
        ts_medians = medfilt(
            cp.nanmedian(dynamic_spectra, axis=1), kernel_size=kernel_size
        )
        # break up into two subtractions so the final number comes out where we want it
        dynamic_spectra = dynamic_spectra - ts_medians[:, None]
        spectra_medians = medfilt(
            cp.nanmean(dynamic_spectra, axis=0), kernel_size=kernel_size
        )
        return dynamic_spectra - spectra_medians + flatten_to

    ts_medians = cp.nanmedian(dynamic_spectra, axis=1)
    # break up into two subtractions so the final number comes out where we want it
    dynamic_spectra = dynamic_spectra - ts_medians[:, None]
    spectra_medians = cp.nanmean(dynamic_spectra, axis=0)
    return dynamic_spectra - spectra_medians + flatten_to


def to_dtype(data: cp.ndarray, dtype: object) -> cp.ndarray:
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
