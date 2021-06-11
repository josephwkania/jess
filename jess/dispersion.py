#!/usr/bin/env python3
"""
Dispersion Utilities

dedisperse
"""

import numpy as np


def dedisperse(
    data: np.ndarray,
    dm: float,
    tsamp: float,
    chan_freqs: np.ndarray = [],
    delays: np.ndarray = [],
) -> np.ndarray:
    """
    Dedisperse a chunk of data..

    Note:
        Our method rolls the data around while dedispersing it.

    Args:
        data: data to dedisperse
        dm (float): The DM to dedisperse the data at.
        chan_freqs (float): frequencies
        tsamp (float): sampling time in seconds
        delays (float): dispersion delays for each channel (in seconds)

    Returns:
        dedispersed (float): Dedispersed data
    """
    nt, nf = data.shape
    if np.any(delays):
        assert len(delays) == nf
    else:
        assert nf == len(chan_freqs)
        delays = calc_dispersion_delays(dm, chan_freqs)

    delay_bins = np.round(delays / tsamp).astype("int64")

    dedispersed = np.zeros(data.shape, dtype=data.dtype)
    for ii in range(nf):
        dedispersed[:, ii] = np.concatenate(
            [
                data[-delay_bins[ii] :, ii],
                data[: -delay_bins[ii], ii],
            ]
        )
    return dedispersed


def delay_lost(dm: float, chan_freqs: np.ndarray, tsamp: float) -> np.int64:
    """
    Calculates the maximum dispersion delay in number of samples
    """
    max_delay = (
        4148808.0 * dm * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[-1]) ** 2) / 1000
    )
    return np.round(np.abs(max_delay) / tsamp).astype(np.int64)


def calc_dispersion_delays(dm: float, chan_freqs: np.ndarray) -> np.ndarray:
    """
    Calculates dispersion delays at an input DM and a frequency array.

    Args:
        dm (float): DM to calculate the delay
        chan_freqs (float): Frequencies

    Returns:
        delays (float): dispersion delays at each frequency channel
        (in seconds)
    """
    return 4148808.0 * dm * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs) ** 2) / 1000
