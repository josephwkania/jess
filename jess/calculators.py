#!/usr/bin/env python3
"""
The repository for all calculators
"""

import logging
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.stats import entropy, median_abs_deviation


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


def mean(data_array: np.ndarray, factor: int, axis: int) -> np.ndarray:
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
    dynamic_spectra: np.ndarray,
    time_factor: int = None,
    freq_factor: int = None,
    backend: object = signal.decimate,
) -> np.ndarray:
    """
    Makes decimates along either/both time and frequency axes.
    Flattens data along frequency before freqency decimation.
    Fattens again in frequency before returning.

    args:
        dynamic_spectra: dynamic spectra with time on the ventricle axis

        time_factor: factor to reduce time sampling by

        freq_factor: factor to reduce freqency channels

        backend: backend to use to reduce the dimension, default is
                 signal.decimate, consider using jess.calculator.mean

    returns:
        Flattened in frequency dynamic spectra, reduced in time and/or freqency
    """
    if time_factor is not None:
        if not isinstance(time_factor, int):
            time_factor = int(time_factor)
        logging.warning("time_factor was not an int: now is %i", time_factor)
        dynamic_spectra = backend(dynamic_spectra, time_factor, axis=0)
    if freq_factor is not None:
        if not isinstance(freq_factor, int):
            freq_factor = int(freq_factor)
        logging.warning("freq_factor was not an int: now is %i", freq_factor)
        dynamic_spectra = backend(
            dynamic_spectra - np.median(dynamic_spectra, axis=0), freq_factor, axis=1
        )
    return dynamic_spectra - np.median(dynamic_spectra, axis=0)


def flattner_median(
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


def flattner_mix(
    dynamic_spectra: np.ndarray, flatten_to: int = 64, kernel_size: int = 1
) -> np.ndarray:
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
            np.nanmean(dynamic_spectra, axis=0), kernel_size=kernel_size
        )
        return dynamic_spectra - spectra_medians + flatten_to

    ts_medians = np.nanmedian(dynamic_spectra, axis=1)
    # break up into two subtractions so the final number comes out where we want it
    dynamic_spectra = dynamic_spectra - ts_medians[:, None]
    spectra_medians = np.nanmean(dynamic_spectra, axis=0)
    return dynamic_spectra - spectra_medians + flatten_to


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

    return (data - central_0) / dispersion_0, (
        data - central_1[:, None]
    ) / dispersion_1[:, None]


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
        raise ValueError("Axis out of bounds, given axis=%i" % axis)
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


def divide_range(length, num_sections: int) -> np.ndarray:
    """
    Divide range as evenly as possiable.

    Args:
        length: lenght of array

        num_sections: number of sections to divide array

    return:
        array with start and stops of each of the subsections

    note:
        Adapted from numpy.lib.shape_base.array_split
        and subject to the numpy license
    """
    neach_section, extras = divmod(length, num_sections)
    section_sizes = (
        [0] + extras * [neach_section + 1] + (num_sections - extras) * [neach_section]
    )
    return np.array(section_sizes, dtype=int).cumsum()


def to_dtype(data: np.ndarray, dtype: object) -> np.ndarray:
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
