#!/usr/bin/env python3
"""
Dispersion Utilities

dedisperse
"""

import logging

import numpy as np

from jess import _fdmt_utils


# pylint: disable=invalid-name
def dedisperse(
    data: np.ndarray,
    dm: float,
    tsamp: float,
    chan_freqs: np.ndarray = None,
    delays: np.ndarray = None,
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
    _, num_freq = data.shape
    if delays is not None:
        assert len(delays) == num_freq
    elif chan_freqs is not None:
        assert num_freq == len(chan_freqs)
        delays = calc_dispersion_delays(dm, chan_freqs)
    else:
        raise RuntimeError("Must provide chan_freqs or delays")

    delay_bins = np.round(delays / tsamp).astype("int64")

    dedispersed = np.zeros(data.shape, dtype=data.dtype)
    for ichan in range(num_freq):
        dedispersed[:, ichan] = np.concatenate(
            [
                data[-delay_bins[ichan] :, ichan],
                data[: -delay_bins[ichan], ichan],
            ]
        )
    return dedispersed


# pylint: disable=invalid-name
def delay_lost(dm: float, chan_freqs: np.ndarray, tsamp: float) -> np.int64:
    """
    Calculates the maximum dispersion delay in number of samples
    """
    max_delay = (
        4148808.0 * dm * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[-1]) ** 2) / 1000
    )
    return np.round(np.abs(max_delay) / tsamp).astype(np.int64)


# pylint: disable=invalid-name
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


def fdmt(
    dynamic_spectra: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This function implements the  FDMT algorithm.
    Note that the bandpass must be flipped, and time
    on the horizontal

    Args:
        dynamic_spectra - Dynamic Spectra to dedisperse

        f_min,f_max are the base-band begin and end frequencies.
                   The frequencies should be entered in MHz

        max_dt - the maximal delay (in time bins) of the maximal dispersion.
                   Appears in the paper as N_{Delta}
                   A typical input is maxDT = N_f

        dtype - a valid numpy dtype.
                reccomended: either int32, or int64.

    Returns:
        The dispersion measure transform of the Input matrix.
        he output dimensions are [Input.shape[1],maxDT]

    For details, see algorithm 1 in Zackay & Ofek (2014)
    See the Lincense in _fdmt_utils.py
    """
    nfreqs, ntimes = dynamic_spectra.shape

    freq_log = int(np.log2(nfreqs))
    powers_of_two = {2**i for i in range(1, 30)}
    if nfreqs not in powers_of_two:
        raise RuntimeError(f"{nfreqs=} is not a power of 2")
    if ntimes not in powers_of_two:
        raise RuntimeError(f"{ntimes=} is not a power of 2")

    # start = time.time()
    state = _fdmt_utils.fdmt_initialization(
        dynamic_spectra, f_min, f_max, max_dt, dtype
    )
    logging.debug("Initialization ended")

    for i_t in range(1, freq_log + 1):
        state = _fdmt_utils.fdmt_iteration(
            state, max_dt, nfreqs, f_min, f_max, i_t, dtype
        )

    # logging.debug("total_time: %.2f", time.time() - start)
    _, d_t, t_s = state.shape
    dmt = np.reshape(state, [d_t, t_s])
    return dmt


def fdmt_fft(
    dynamic_spectra: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This function implements the  FDMT-FFT algorithm.

    Args:
        dynamic_spectra -

        f_min,f_max - are the base-band begin and end frequencies.
                   he frequencies can be entered in both MHz and GHz, units
                   are factored out in all uses.
        maxDT - The maximal delay (in time bins) of the maximal dispersion.
                Appears in the paper as N_{Delta}
                A typical input is maxDT = N_f
        dtype - To naively use FFT, one must use floating point types.
                    Due to casting, use either complex64 or complex128.

    Returns:
        The dispersion measure transform of the Input matrix.
        The output dimensions are [Input.shape[1],maxDT]

    For details, see algorithm 2 in Zackay & Ofek (2014)
    See the Lincense in _fdmt_utils.py
    """
    nfreqs, ntimes = dynamic_spectra.shape

    freq_log = int(np.log2(nfreqs))
    powers_of_two = {2**i for i in range(1, 30)}
    if nfreqs not in powers_of_two:
        raise RuntimeError(f"{nfreqs=} is not a power of 2")
    if ntimes not in powers_of_two:
        raise RuntimeError(f"{ntimes=} is not a power of 2")

    # start = time.time()
    state = _fdmt_utils.fdmtfft_initialization(
        dynamic_spectra, f_min, f_max, max_dt, dtype
    )
    logging.debug("Initialization ended!")

    for i_t in range(1, freq_log + 1):
        # print i_t
        # xx = time.time()
        state = _fdmt_utils.fdmtfft_iteration(
            state, max_dt, nfreqs, f_min, f_max, i_t, dtype
        )
        # print time.time() - xx
    # logging.debug("total_time: %.3f", time.time() - start)
    t_s, _, d_t = state.shape
    state = np.transpose(state, axes=[1, 2, 0])
    dmt = np.reshape(np.fft.ifft(state, axis=2), [d_t, t_s])
    return dmt
