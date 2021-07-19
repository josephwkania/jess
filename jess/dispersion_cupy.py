#!/usr/bin/env python3
"""
Dispersion Utilities using cupy
"""

import cupy as cp


def dedisperse(
    data: cp.ndarray,
    dm: float,
    tsamp: float,
    chan_freqs: cp.ndarray = None,
    delays: cp.ndarray = None,
) -> cp.ndarray:
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
    _, num_freq = data.shape
    if delays is not None:
        assert len(delays) == num_freq
    else:
        assert num_freq == len(chan_freqs)
        delays = calc_dispersion_delays(dm, chan_freqs)

    delay_bins = cp.round(delays / tsamp).astype("int64")

    dedispersed = cp.zeros(data.shape, dtype=data.dtype)
    for ichan in range(num_freq):
        dedispersed[:, ichan] = cp.concatenate(
            [
                data[-delay_bins[ichan] :, ichan],
                data[: -delay_bins[ichan], ichan],
            ]
        )
    return dedispersed


def delay_lost(dm: float, chan_freqs: cp.ndarray, tsamp: float) -> cp.int64:
    """
    Calculates the maximum dispersion delay in number of samples
    """
    max_delay = (
        4148808.0 * dm * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[-1]) ** 2) / 1000
    )
    return cp.around(cp.abs(max_delay) / tsamp).astype(cp.int64)


def calc_dispersion_delays(dm: float, chan_freqs: cp.ndarray) -> cp.ndarray:
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
