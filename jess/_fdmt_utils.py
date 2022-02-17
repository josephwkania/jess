#!/usr/bin/env python3
"""
Fast Dispersion Measure Transform utilities.
Based on Barak Zackay, see __License__
"""

import logging
from typing import Callable, List, Tuple, Union

import numpy as np

DISPERSION_CONSTANT = 4.148808 * 10**9  # Mhz * pc^-1 * cm^3


def get_dmt_function(func_name: str) -> Tuple[Callable, type]:
    """
    Get the Dedispersion function
    """
    func_name = func_name.casefold()
    if func_name == "fdmtfft":
        return fdmt_fft, np.complex128
    if func_name == "fdmt":
        return fdmt, np.int64

    raise NotImplementedError(f"{func_name} is not avaliable")


def argmax_nd(array: np.ndarray) -> Tuple:
    """
    Find the the location of the maximum in an N dimension array

    Args:
        Nd array

    Returns:
        Nd max location
    """
    return np.unravel_index(np.argmax(array), array.shape)


def max_nd(array: np.ndarray) -> float:
    """
    Find max in Nd array

    Args:
        Nd array

    Returns:
        Nd max
    """
    return array[argmax_nd(array)]


def fdmt(
    dynamic_spectra: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    dtype: Union[np.dtype, type],
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
    state = fdmt_initialization(dynamic_spectra, f_min, f_max, max_dt, dtype)
    logging.debug("Initialization ended")

    for i_t in range(1, freq_log + 1):
        state = fdmt_iteration(state, max_dt, nfreqs, f_min, f_max, i_t, dtype)

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
    state = fdmtfft_initialization(dynamic_spectra, f_min, f_max, max_dt, dtype)
    logging.debug("Initialization ended!")

    for i_t in range(1, freq_log + 1):
        # print i_t
        # xx = time.time()
        state = fdmtfft_iteration(state, max_dt, nfreqs, f_min, f_max, i_t, dtype)
        # print time.time() - xx
    # logging.debug("total_time: %.3f", time.time() - start)
    t_s, _, d_t = state.shape
    state = np.transpose(state, axes=[1, 2, 0])
    dmt = np.reshape(np.fft.ifft(state, axis=2), [d_t, t_s])
    return dmt


def fdmt_initialization(
    dynamic_spectra: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    dtype: Union[np.dtype, type],
) -> np.ndarray:
    """
    dynamic_spectra - Dynamic Spectra to dedisperse

    f_min,f_max - are the base-band begin and end frequencies.
        The frequencies can be entered in both MHz and GHz,
        units are factored out in all uses.

    maxDT - the maximal delay (in time bins) of the maximal dispersion.
        Appears in the paper as N_{Delta}
        A typical input is maxDT = N_f

    dtype - To naively use FFT, one must use floating point types.
            Due to casting, use either complex64 or complex128.

    Returns:
        3d array, with dimensions [N_f,N_d0,Nt]
        where N_d0 is the maximal number of bins the
        dispersion curve travels at one frequency bin

    For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    # Data initialization is done prior to the first FDMT iteration
    # See Equations 17 and 19 in Zackay & Ofek (2014)

    nfreqs, ntimes = dynamic_spectra.shape

    delta_f = (f_max - f_min) / float(nfreqs)
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1.0 / f_min**2 - 1.0 / (f_min + delta_f) ** 2)
            / (1.0 / f_min**2 - 1.0 / f_max**2)
        )
    )

    output = np.zeros([nfreqs, delta_t + 1, ntimes], dtype)
    output[:, 0, :] = dynamic_spectra

    for i_dt in range(1, delta_t + 1):
        output[:, i_dt, i_dt:] = output[:, i_dt - 1, i_dt:] + dynamic_spectra[:, :-i_dt]
    return output


def fdmtfft_initialization(
    dynamic_spectra: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Args:
        dynamic_spectra - power matrix I(f,t)

        f_min,f_max - are the base-band begin and end frequencies.
                    The frequencies can be entered in both MHz and GHz,
                    units are factored out in all uses.

        maxDT - the maximal delay (in time bins) of the maximal dispersion.
                Appears in the paper as N_{Delta}
                A typical input is maxDT = N_f

        dtype - To naively use FFT, one must use floating point types.
                Due to casting, use either complex64 or complex128.

    Returns:
        3d array, with dimensions [N_f,N_d0,Nt]
        where N_d0 is the maximal number of bins the dispersion curve travels
        at one frequency binthe difference from FDMT_FFT_initialization
        is that the time axis is FFT'ed

    For details, see algorithm 2 in Zackay & Ofek (2014)
    """

    freqs, times = dynamic_spectra.shape

    delta_f = (f_max - f_min) / float(freqs)
    # determining the maximal deltaT that we will encounter in the first
    # iteration.
    # if deltaT is too large, consider binning
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1.0 / f_min**2 - 1.0 / (f_min + delta_f) ** 2)
            / (1.0 / f_min**2 - 1.0 / f_max**2)
        )
    )

    output = np.zeros([freqs, delta_t + 1, times], dtype)

    # Initializing the "A_f^{f + \delta f} (t_0,\Delta t)" array
    output[:, 0, :] = dynamic_spectra
    for i_dt in range(1, delta_t + 1):
        output[:, i_dt, i_dt:] = output[:, i_dt - 1, i_dt:] + dynamic_spectra[:, :-i_dt]

    # FFT-ing the time axis and transposing the data
    return np.transpose(np.fft.fft(output, axis=2), axes=[2, 0, 1])


def fdmt_iteration(
    input_cube: np.ndarray,
    max_dt: int,
    nfreq: int,
    f_min: float,
    f_max: float,
    iteration_num: int,
    dtype: Union[np.dtype, type],
) -> np.ndarray:
    """
    Input:
        Input - 3d array, with dimensions [N_f,N_d0,Nt]
        f_min,f_max - are the base-band begin and end frequencies.
            The frequencies can be entered in both MHz and GHz, units are
            factored out in all uses.
        maxDT - the maximal delay (in time bins) of the maximal dispersion.
            Appears in the paper as N_{Delta}
            A typical input is maxDT = N_f
        dataType - To naively use FFT, one must use floating point types.
            Due to casting, use either complex64 or complex128.
        iteration num - Algorithm works in log2(Nf) iterations, each iteration
        changes all the sizes (like in FFT)
    Output:
        3d array, with dimensions [N_f/2,N_d1,Nt]
        where N_d1 is the maximal number of bins the dispersion curve travels at
        one output frequency band

    For details, see algorithm 1 in Zackay & Ofek (2014)
    """

    input_dims = input_cube.shape
    output_dims = list(input_dims)

    delta_f = 2 ** (iteration_num) * (f_max - f_min) / float(nfreq)
    d_f = (f_max - f_min) / float(nfreq)
    # the maximum deltaT needed to calculate at the i'th iteration
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1.0 / f_min**2 - 1.0 / (f_min + delta_f) ** 2)
            / (1.0 / f_min**2 - 1.0 / f_max**2)
        )
    )
    logging.debug("deltaT = %.2f", delta_t)
    logging.debug("N_f = %.2f", nfreq / 2.0 ** (iteration_num))
    logging.debug("input_dims = %.2f", input_dims)

    output_dims[0] = output_dims[0] // 2

    output_dims[1] = delta_t + 1
    logging.debug("output_dims: %s", (output_dims,))
    output = np.zeros(output_dims, dtype=dtype)

    # No negative D's are calculated => no shift is needed
    # If you want negative dispersions, this will have to change to
    # 1+deltaT,1+deltaTOld
    # Might want to calculate negative dispersions when using
    # coherent dedispersion, to reduce the number of trial
    # dispersions by a factor of 2 (reducing the complexity
    # of the coherent part of the hybrid)
    shift_output = 0
    shift_input = 0
    times = output_dims[2]
    f_jumps = output_dims[0]

    # For some situations, it is beneficial to play with this correction.
    # When applied to real data, one should carefully analyze and understand
    # the effect of his correction on the pulse he is looking for
    # (especially if convolving with a specific pulse profile)
    if iteration_num > 0:
        correction = d_f / 2.0
    else:
        correction = 0
    for i_f in range(f_jumps):

        f_start = (f_max - f_min) / float(f_jumps) * (i_f) + f_min
        f_end = (f_max - f_min) / float(f_jumps) * (i_f + 1) + f_min
        f_middle = (f_end - f_start) / 2.0 + f_start - correction
        # it turned out in the end, that putting the correction +dF to
        # f_middle_larger (or -dF/2 to f_middle, and +dF/2 to f_middle larger)
        # is less sensitive than doing nothing when dedispersing a coherently
        #  dispersed pulse.
        # The confusing part is that the hitting efficiency is better
        # with the corrections (!?!).
        f_middle_larger = (f_end - f_start) / 2 + f_start + correction
        delta_t_local = int(
            np.ceil(
                (max_dt - 1)
                * (1.0 / f_start**2 - 1.0 / (f_end) ** 2)
                / (1.0 / f_min**2 - 1.0 / f_max**2)
            )
        )

        for i_dt in range(delta_t_local + 1):
            dt_middle = round(
                i_dt
                * (1.0 / f_middle**2 - 1.0 / f_start**2)
                / (1.0 / f_end**2 - 1.0 / f_start**2)
            )
            dt_middle_index = dt_middle + shift_input

            dt_middle_larger = round(
                i_dt
                * (1.0 / f_middle_larger**2 - 1.0 / f_start**2)
                / (1.0 / f_end**2 - 1.0 / f_start**2)
            )

            dt_rest = i_dt - dt_middle_larger
            dt_rest_index = dt_rest + shift_input

            i_t_min = 0

            i_t_max = dt_middle_larger
            output[i_f, i_dt + shift_output, i_t_min:i_t_max] = input_cube[
                2 * i_f, dt_middle_index, i_t_min:i_t_max
            ]

            i_t_min = dt_middle_larger
            i_t_max = times

            output[i_f, i_dt + shift_output, i_t_min:i_t_max] = (
                input_cube[2 * i_f, dt_middle_index, i_t_min:i_t_max]
                + input_cube[
                    2 * i_f + 1,
                    dt_rest_index,
                    i_t_min - dt_middle_larger : i_t_max - dt_middle_larger,
                ]
            )

    return output


def fdmtfft_iteration(
    input_cube: np.ndarray,
    max_dt: int,
    nfreq: int,
    f_min: float,
    f_max: float,
    iteration_num: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Input:
        Input - 3d array, with dimensions [N_f,N_d0,Nt]
        f_min,f_max - are the base-band begin and end frequencies.
            The frequencies can be entered in both MHz and GHz, units are
            factored out in all uses.
        maxDT - the maximal delay (in time bins) of the maximal dispersion.
            Appears in the paper as N_{Delta}
            A typical input is maxDT = N_f
        dataType - To naively use FFT, one must use floating point types.
            Due to casting, use either complex64 or complex128.
        iteration num - Algorithm works in log2(Nf) iterations, each
                        iteration changes all the sizes (like in FFT)
    Output:
        3d array, with dimensions [N_f/2,N_d1,Nt]
        where N_d1 is the maximal number of bins the dispersion curve
        travels at one output frequency band

    For details, see algorithm 2 in Zackay & Ofek (2014)
    """

    input_dims = input_cube.shape
    output_dims = list(input_dims)

    delta_f = 2 ** (iteration_num) * (f_max - f_min) / float(nfreq)
    d_f = (f_max - f_min) / float(nfreq)
    # the maximum deltaT needed to calculate at the i'th iteration
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1.0 / f_min**2 - 1.0 / (f_min + delta_f) ** 2)
            / (1.0 / f_min**2 - 1.0 / f_max**2)
        )
    )
    logging.debug("delta_t: %.2f", delta_t)
    logging.debug("n_f = %.2f", nfreq / 2.0 ** (iteration_num))

    output_dims[1] = output_dims[1] // 2

    output_dims[2] = delta_t + 1
    output = np.zeros(output_dims, dtype)

    # No negative K's are calculated => no shift is needed
    # If you want negative shifts, this will have to change to 1+deltaT,
    # 1+deltaTOld
    shift_output = 0
    shift_input = 0
    times = output_dims[0]
    f_jumps = output_dims[1]

    # see remark about this correction in the FDMT implementation.
    correction = d_f / 2.0

    f_min_fac = 1.0 / f_min**2.0
    delta_t_shift = (
        np.ceil(
            (max_dt - 1)
            * (f_min_fac - 1.0 / (f_min + delta_f / 2.0 + d_f / 2.0) ** 2.0)
            / (f_min_fac - 1.0 / f_max**2.0)
        )
        + 3
    )
    shift_row = np.fft.fft((np.eye(delta_t_shift, times)), axis=1)
    for i_f in range(f_jumps):
        f_start = (f_max - f_min) / float(f_jumps) * (i_f) + f_min
        f_end = (f_max - f_min) / float(f_jumps) * (i_f + 1) + f_min
        f_middle = (f_end - f_start) / 2.0 + f_start - correction
        # correction was removed. see the explanation in FDMT code.
        f_middle_larger = (f_end - f_start) / 2.0 + f_start + correction
        delta_t_local = int(
            np.ceil(
                (max_dt - 1)
                * (1.0 / f_start**2 - 1.0 / (f_end) ** 2)
                / (1.0 / f_min**2 - 1.0 / f_max**2)
            )
        )
        for i_dt in range(delta_t_local + 1):
            dt_middle = round(
                i_dt
                * (1.0 / f_middle**2 - 1.0 / f_start**2)
                / (1.0 / f_end**2 - 1.0 / f_start**2)
            )
            dt_middle_index = dt_middle + shift_input
            dt_middle_larger = round(
                i_dt
                * (1.0 / f_middle_larger**2 - 1.0 / f_start**2)
                / (1.0 / f_end**2 - 1.0 / f_start**2)
            )

            dt_rest = i_dt - dt_middle_larger
            dt_rest_index = dt_rest + shift_input

            output[:, i_f, i_dt + shift_output] = (
                input_cube[:, 2 * i_f, dt_middle_index]
                + input_cube[:, 2 * i_f + 1, dt_rest_index]
                * shift_row[dt_middle_larger, :]
            )

    return output


def coherent_dedispersion(
    raw_signal: np.ndarray,
    dm: float,  # pylint: disable=invalid-name
    f_min: float,
    f_max: float,
    already_ffted: bool = False,
):
    """
    Will perform coherent dedispersion.
    Args:
        raw signal - is assumed to be a one dimensional signal

        dm - is the dispersion measure. units: pc*cm^-3

        f_min - Minimum freq, given in Mhz

        f_max - Maximum freq, given in Mhz

        already_ffted - to reduce complexity, insert fft(raw_signal) instead of
                        raw_signal, and indicate by this flag

    For future improvements:
    1) Signal partition to chunks of length N_d is not applied, and maybe
       it should be.
    2) No use of packing is done, though it is obvious it should be done
       (either in the coherent stage (and take special care of the abs()**2
       operation done by other functions) or in the incoherent stage)

    """
    n_total = len(raw_signal)
    practical_dm = DISPERSION_CONSTANT * dm
    freqs = np.arange(0, f_max - f_min, float(f_max - f_min) / n_total)

    # The added linear term makes the arrival times of the
    # highest frequencies be 0
    exponental = np.e ** (
        -(
            2 * np.pi * complex(0, 1) * practical_dm / (f_min + freqs)
            + 2 * np.pi * complex(0, 1) * practical_dm * freqs / (f_max**2)
        )
    )
    if not already_ffted:
        coherent_dedisp = np.fft.ifft(np.fft.fft(raw_signal) * exponental)
    else:
        coherent_dedisp = np.fft.ifft(raw_signal * exponental)
    return coherent_dedisp


def stft(raw_signal: np.ndarray, block_size: int) -> np.ndarray:
    """
    Raw signal will be divided to blocks, each block will be fourier transformed

    Args:
        raw_signal - raw antenna voltage time series

        block_size - number of bins in each block

    Returns:
        frequency vs. time matrix

    Note: absolute value squared is not performed!
    """
    blocks = np.transpose(
        raw_signal[: len(raw_signal) // block_size * block_size].reshape(
            [len(raw_signal) / block_size, block_size]
        )
    )
    return np.fft.fft(blocks, axis=0)


def hybrid_dedispersion(
    raw_signal: np.ndarray,
    n_p: int,
    dm_max: float,
    f_min: float,
    f_max: float,
    sigma_bound: float = 7,
) -> List:
    """
    Will perform the coherent FDMT hybrid algorithm.
    see algorithm 3 in Zackay & Ofek (2014)
    Args:

        raw_signal - raw antenna voltage time series

        n_p - length of the pulse in time bins, i.e (t_p/\tau), or N_p
              in the paper.

        dm_max - maximal dispersion to scan, in units of pc cm^-3

        f_min,f_max - minimal and maximal frequency band of the signal.
                        - signal is assumed to be base-band sampled

        sigma_bound - The minimal statistical significant to trigger a saving
                      of a result.
                      - You may want to change the storage decision process
                      for your own purposes.

    Returns:
            frequency vs. time matrix

    See example usage in the test function HybridDedispersion_test

    Subjects for future improvement:
    Might want to reduce trial dispersions by a factor of 2 by letting FDMT
    scan negative dispersions.
    Might want to use packing (either in the coherent stage, or in the incoherent stage)
    """
    # ConversionConst is used to convert from Dispersion measure in units of pc*cm^-3
    # to time bins
    conversion_const = (
        DISPERSION_CONSTANT * (1.0 / f_min**2 - 1.0 / f_max**2) * (f_max - f_min)
    )
    n_d = dm_max * conversion_const

    n_coherent = (np.ceil(n_d / (n_p**2))).astype(int)
    logging.debug("Number of coherent iterations: %i", n_coherent)
    ffted_signal = np.fft.fft(raw_signal)

    logging.debug(
        "FDMT parameters: n_p, len(raw_signal)/n_p %s, f_min %.2f, fmax: %.2f, int64",
        [
            n_p,
            len(raw_signal) / n_p,
        ],
        f_min,
        f_max,
    )
    fdmt_normalization = fdmt(
        np.ones([n_p, len(raw_signal) // n_p]), f_min, f_max, n_p, np.int64
    )
    for outer_d in range(n_coherent):

        cur_coherent_d = outer_d * (dm_max / float(n_coherent))
        logging.debug(
            "coherent iteration %i, cur_coherent_d: %.2f", outer_d, cur_coherent_d
        )
        fdmt_input = (
            abs(
                stft(
                    coherent_dedispersion(
                        ffted_signal, cur_coherent_d, f_min, f_max, already_ffted=True
                    ),
                    n_p,
                )
            )
            ** 2
        )
        fdmt_input -= np.mean(fdmt_input)
        std = np.std(fdmt_input)
        fdmt_input /= 0.25 * std
        var = std**2
        fdmt_output = fdmt(fdmt_input, f_min, f_max, n_p, np.int64)
        fdmt_output /= np.sqrt(fdmt_normalization * var + 0.000001)
        if np.max(fdmt_output) > sigma_bound:
            sigma_bound = np.max(fdmt_output)
            logging.debug("Achieved score with: %.2f sigma", sigma_bound)
            res = [cur_coherent_d, fdmt_output.copy()]
    return res


def bit_pack(inputs, dtype: np.dtype, safe: bool = True) -> np.ndarray:
    """
    This function packs different instances of FDMT inputs to one output
    It does that by shifting each instance with a different amount of bits.

    If dataType is complex then there is also packing of half the instances in
    the imaginary field (this is useful for FDMTFFT)

    Not completely debugged yet, beware with the float datatypes.
    """
    if safe:
        raise NotImplementedError("Not debugged yet - Use with caution")
    output = inputs[0].astype(dtype)

    # check the size of the mantissa and the required number of bits
    # for precision in FFT
    # (11 for exp field, 7 for precision, verify with FFT
    # how much is enough)
    n_float_free64 = 18
    # (8 for exp field, 6 for precision, verify with FFT
    #  how much is enough)
    n_float_free32 = 14
    if dtype == np.int64:
        n_bits = 64
        pack_complex = False
    if dtype == np.int32:
        n_bits = 32
        pack_complex = False
    if dtype == np.complex128:
        if len(inputs) % 2 != 0:
            raise RuntimeError(
                "On complex data types the number of inputs should be even!"
            )
        n_bits = 64 - n_float_free64
        pack_complex = True
    if dtype == np.complex64:
        if len(inputs) % 2 != 0:
            raise RuntimeError(
                "On complex data types the number of inputs should be even!"
            )
        n_bits = 32 - n_float_free32
        pack_complex = True

    if not pack_complex:
        n_shift = n_bits / len(inputs)
        n_pack = len(inputs)
    else:
        n_shift = n_bits / (len(inputs) / 2)
        n_pack = len(inputs) // 2
    for i in range(1, n_pack):
        output += inputs[i] * (2 ** (n_shift * i))

    if pack_complex:
        for i in range(n_pack, len(inputs)):
            output += complex(0, 1) * inputs[i] * (2 ** (n_shift * i))

    return output


__Licence__ = """
<OWNER> = Barak Zackay (Weizmann Institute of Science)
<YEAR> = 2014

In the original BSD license, both occurrences of the phrase
"COPYRIGHT HOLDERS AND CONTRIBUTORS" in the disclaimer read
"REGENTS AND CONTRIBUTORS".

Here is the license template:

Copyright (c) 2014, Barak Zackay (Weizmann Institute of Science)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
