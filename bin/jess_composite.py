#!/usr/bin/env python3
"""
Remasters fits/fil files.

Runs a mad filter, fft mad filter, and the a
high pass filter over the data.

Rights out a filterbank.
"""

import argparse
import logging
import os
import textwrap
from typing import Union

import numpy as np
from rich.logging import RichHandler
from rich.progress import track
from your import Writer, Your
from your.formats.filwriter import sigproc_object_from_writer

# from your.utils.math import primes
from your.utils.misc import YourArgparseFormatter

from jess.calculators import get_flatten_to

try:
    import cupy as cp

    BACKEND_GPU = True
except ModuleNotFoundError:

    BACKEND_GPU = False


logger = logging.getLogger()


def get_outfile(file: str, out_file: Union[str, None]) -> str:
    """
    Makes the outfile name by:
    if no str is given -> append _mad to the original file name
    if str is given with an extension other than .fil -> assert error
    if str is given without extention -> add .fil
    """
    if not out_file:
        # if no out file is given, create the string
        path, file_ext = os.path.splitext(file[0])
        out_file = path + "_composite.fil"
        logger.info("No outfile given, writing to %s", out_file)
        return out_file

    # if a file is given, make sure it has a .fil ext
    path, file_ext = os.path.splitext(out_file)
    if file_ext:
        assert file_ext == ".fil", f"I can only write .fil, you gave: {file_ext}!"
        return out_file

    out_file += ".fil"
    return out_file


def clean_cpu(
    yr_input: Your,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    time_median_size: int,
    modes_to_zero: int,
    out_file: str,
    sigproc_object: sigproc_object_from_writer,
) -> None:
    """
    Run the MAD filter on chunks of data without dedispersing it.
    Runs on CPU

     Args:
        yr_input: the your object of the file you want to clean

        sigma: Sigma at which to remove outliers

        gulp: The amount of data to process.

        flatten_to: make this the median out the out data.

        channels_per_subband: the number of channels for each MAD
                              subband

        modes_to_zero: zero dm fft modes to remove

        out_file: name of the file to write out

        sigproc_obj: sigproc object to write out

    Returns
        None
    """

    logging.debug("Using CPU backend")

    # need a bandpass if we do zero_dming, put outside the loop
    if modes_to_zero >= 1:
        bandpass = np.full(
            yr_input.your_header.nchans, fill_value=flatten_to, dtype=np.float32
        )
        no_time_detrend = False
    elif modes_to_zero == 0:
        # preserve zero DM time series
        no_time_detrend = True
    else:
        # This will be median subtraction
        no_time_detrend = False

    total_flag = np.zeros(3)
    n_iter = 0
    for j in track(
        range(0, yr_input.your_header.nspectra, gulp),
        description="Cleaning File",
        transient=True,
    ):
        logging.debug("Cleaning samples starting at %i", j)
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        # cleaned = fft_mad(
        #     cp.asarray(data), sigma=sigma, chans_per_subband=channels_per_subband
        # )
        cleaned, _, mad_percentage = mad_spectra_flat(
            data,
            chans_per_subband=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            time_median_size=time_median_size,
            return_same_dtype=False,
            mask_chans=True,
            no_time_detrend=no_time_detrend,
        )
        cleaned, _, fft_percentage = fft_mad(
            cleaned,
            sigma=sigma,
            chans_per_subband=channels_per_subband,
            time_median_size=1,
            return_same_dtype=False,
        )

        if modes_to_zero == 1:
            logging.debug("Zero DMing: Subtracting Mean")
            cleaned, dm_percentage = zero_dm(cleaned, bandpass, return_same_dtype=False)
        elif modes_to_zero > 1:
            logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
            cleaned, dm_percentage = zero_dm_fft(
                cleaned, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
            )
        else:
            dm_percentage = 0

        n_iter += 1
        total_flag += np.asarray((mad_percentage, fft_percentage, dm_percentage))
        logging.info(
            "mad: %.1f%%, fft: %.1f%%, highpass: %.1f%%, total flagged: %.1f%%",
            mad_percentage,
            fft_percentage,
            dm_percentage,
            mad_percentage + fft_percentage + dm_percentage,
        )
        # Keep full precision until done
        cleaned = to_dtype(cleaned, dtype=yr_input.your_header.dtype)

        sigproc_object.append_spectra(cleaned, out_file)

    n_iter = yr_input.your_header.nspectra / gulp
    total_flag /= n_iter
    logging.info(
        "Full file - mad: %.1f%%, fft: %.1f%%, highpass: %.1f%%, total flagged: %.1f%%",
        total_flag[0],
        total_flag[1],
        total_flag[2],
        total_flag.sum(),
    )


def clean_gpu(
    yr_input: Your,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    time_median_size: int,
    modes_to_zero: int,
    out_file: str,
    sigproc_object: sigproc_object_from_writer,
) -> None:
    """
    Run the MAD filter on chunks of data without dedispersing it.
    Runs on GPU

     Args:
        yr_input: the your object of the file you want to clean

        sigma: Sigma at which to remove outliers

        gulp: The amount of data to process.

        flatten_to: make this the median out the out data.

        channels_per_subband: the number of channels for each MAD
                              subband

        modes_to_zero: zero dm fft modes to remove

        out_file: name of the file to write out

        sigproc_obj: sigproc object to write out

    Returns
        None
    """

    logging.debug("Using GPU backend")

    # need a bandpass if we do zero_dming, put outside the loop
    if modes_to_zero >= 1:
        bandpass = cp.full(
            yr_input.your_header.nchans, fill_value=flatten_to, dtype=cp.float32
        )
        no_time_detrend = False
    elif modes_to_zero == 0:
        # preserve zero DM time series
        no_time_detrend = True
    else:
        # This will be median subtraction
        no_time_detrend = False

    total_flag = cp.zeros(3)
    n_iter = 0
    for j in track(
        range(0, yr_input.your_header.nspectra, gulp),
        description="Cleaning File",
        transient=True,
    ):
        logging.debug("Cleaning samples starting at %i", j)
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        # cleaned = fft_mad(
        #     cp.asarray(data), sigma=sigma, chans_per_subband=channels_per_subband
        # )
        cleaned, _, mad_percentage = mad_spectra_flat(
            cp.asarray(data),
            chans_per_subband=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            time_median_size=time_median_size,
            return_same_dtype=False,
            mask_chans=True,
            no_time_detrend=no_time_detrend,
        )
        cleaned, _, fft_percentage = fft_mad(
            cleaned,
            sigma=sigma,
            chans_per_subband=channels_per_subband,
            time_median_size=1,
            return_same_dtype=False,
        )

        if modes_to_zero == 1:
            logging.debug("Zero DMing: Subtracting Mean")
            cleaned, dm_percentage = zero_dm(cleaned, bandpass, return_same_dtype=False)
        elif modes_to_zero > 1:
            logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
            cleaned, dm_percentage = zero_dm_fft(
                cleaned, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
            )
        else:
            dm_percentage = cp.asarray((0))

        n_iter += 1
        total_flag += cp.asarray((mad_percentage, fft_percentage, dm_percentage))
        logging.info(
            "MAD: %.1f%%, FFT: %.1f%%, Highpass: %.1f%%, Total: %.1f%%",
            mad_percentage,
            fft_percentage,
            dm_percentage,
            mad_percentage + fft_percentage + dm_percentage,
        )
        # Keep full precision until done
        cleaned = to_dtype(cleaned, dtype=yr_input.your_header.dtype)

        sigproc_object.append_spectra(cleaned.get(), out_file)

    total_flag /= n_iter
    logging.info(
        "Full file - MAD: %.1f%%, FFT: %.1f%%, Highpass: %.1f%%, Total: %.1f%%",
        total_flag[0],
        total_flag[1],
        total_flag[2],
        total_flag.sum(),
    )


def clean_dispersion(
    yr_input: Your,
    dispersion_measure: float,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    time_median_size: int,
    modes_to_zero: int,
    out_file: str,
    sigproc_object: sigproc_object_from_writer,
) -> None:
    """
    Run the MAD filter on chunks of data without dedispersing it.

     Args:
        yr_input: the your object of the file you want to clean

        dispersion_measure: The dispersion measure to dedisperse
                            the data to

        sigma: Sigma at which to remove outliers

        gulp: The amount of data to process.

        flatten_to: make this the median out the out data.

        channels_per_subband: the number of channels for each MAD
                              subband

        modes_to_zero: zero dm fft modes to remove

        out_file: name of the file to write out

        sigproc_obj: sigproc object to write out

    Returns
        None
    """
    samples_lost = delay_lost(
        dispersion_measure, yr_input.chan_freqs, yr_input.your_header.tsamp
    )
    logging.debug(
        "dispersion_measure: %f, yr_input.chan_freqs: %s",
        dispersion_measure,
        yr_input.chan_freqs,
    )
    logging.debug("yr_input.your_header.tsamp: %f", yr_input.your_header.tsamp)
    logging.debug("samples_lost: %i", samples_lost)

    # need a bandpass if we do zero_dming, put outside the loop
    if modes_to_zero >= 1:
        bandpass = cp.full(
            yr_input.your_header.nchans, fill_value=flatten_to, dtype=cp.float32
        )

    # add data that can't be dispersed
    # because its at the start
    # if not remove_ends:
    cleaned, _, mad_percentage = mad_spectra_flat(
        cp.asarray(yr_input.get_data(0, samples_lost)),
        chans_per_subband=channels_per_subband,
        sigma=sigma,
        flatten_to=flatten_to,
        time_median_size=time_median_size,
        return_same_dtype=False,
        no_time_detrend=True,
        mask_chans=True,
    )
    cleaned, _, fft_percentage = fft_mad(
        cleaned,
        sigma=sigma,
        chans_per_subband=channels_per_subband,
        time_median_size=1,
        return_same_dtype=False,
    )

    if modes_to_zero == 1:
        logging.debug("Zero DMing: Subtracting Mean")
        cleaned, dm_percentage = zero_dm(cleaned, bandpass, return_same_dtype=False)
    elif modes_to_zero > 1:
        logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
        cleaned, dm_percentage = zero_dm_fft(
            cleaned, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
        )
    else:
        dm_percentage = 0

    logging.info(
        "Start of file - MAD: %.1f%%, FFT: %.1f%%, Highpass: %.1f%%, Total: %.1f%%",
        mad_percentage,
        fft_percentage,
        dm_percentage,
        mad_percentage + fft_percentage + dm_percentage,
    )
    cleaned = to_dtype(cleaned, yr_input.your_header.dtype)
    sigproc_object.append_spectra(cleaned.get(), out_file)

    n_iter = 0
    total_flag = cp.zeros(3)
    # loop through all the data we can dedisperse
    for j in track(
        range(0, yr_input.your_header.nspectra, gulp),
        description="Cleaning File",
        transient=True,
    ):
        logging.debug("Cleaning samples starting at %i", j)

        if 2 * samples_lost + j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, 2 * samples_lost + gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)
        dedisp = dedisperse(
            cp.asarray(data),
            dispersion_measure,
            yr_input.your_header.tsamp,
            yr_input.chan_freqs,
        )
        dedisp[0:-samples_lost, :], _, mad_percentage = mad_spectra_flat(
            dedisp[0:-samples_lost, :],
            chans_per_subband=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            time_median_size=time_median_size,
            no_time_detrend=True,
            return_same_dtype=False,
            mask_chans=True,
        )
        redisip = dedisperse(
            dedisp, -dispersion_measure, yr_input.your_header.tsamp, yr_input.chan_freqs
        )
        redisip, _, fft_percentage = fft_mad(
            redisip,
            sigma=sigma,
            chans_per_subband=channels_per_subband,
            time_median_size=1,
            return_same_dtype=False,
        )

        if modes_to_zero == 1:
            logging.debug("Zero DMing: Subtracting Mean")
            redisip, dm_percentage = zero_dm(redisip, bandpass, return_same_dtype=False)
        elif modes_to_zero > 1:
            logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
            redisip, dm_percentage = zero_dm_fft(
                redisip, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
            )
        else:
            dm_percentage = 0

        n_iter += 1
        total_flag += cp.asarray((mad_percentage, fft_percentage, dm_percentage))
        logging.info(
            "MAD: %.1f%%, FFT: %.1f%%, Highpass: %.1f%%, Total: %.1f%%",
            mad_percentage,
            fft_percentage,
            dm_percentage,
            mad_percentage + fft_percentage + dm_percentage,
        )
        redisip = to_dtype(redisip, yr_input.your_header.dtype)
        sigproc_object.append_spectra(
            redisip[samples_lost:-samples_lost, :].get(), out_file
        )

    # add data that can't be dispersed
    # because its at the end

    # if not remove_ends:
    cleaned, _, mad_percentage = mad_spectra_flat(
        cp.asarray(
            yr_input.get_data(
                yr_input.your_header.nspectra - samples_lost, samples_lost
            )
        ),
        chans_per_subband=channels_per_subband,
        sigma=sigma,
        flatten_to=flatten_to,
        time_median_size=time_median_size,
        no_time_detrend=True,
        return_same_dtype=False,
        mask_chans=True,
    )
    cleaned, _, fft_percentage = fft_mad(
        cleaned,
        sigma=sigma,
        chans_per_subband=channels_per_subband,
        time_median_size=1,
        return_same_dtype=False,
    )

    if modes_to_zero == 1:
        logging.debug("Zero DMing: Subtracting Mean")
        cleaned, dm_percentage = zero_dm(cleaned, bandpass, return_same_dtype=False)
    elif modes_to_zero > 1:
        logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
        cleaned, dm_percentage = zero_dm_fft(
            cleaned, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
        )
    else:
        dm_percentage = 0

    logging.info(
        "Start of file - MAD: %.1f%%, FFT: %.1f%%, Highpass: %.1f%%, Total: %.1f%%",
        mad_percentage,
        fft_percentage,
        dm_percentage,
        mad_percentage + fft_percentage + dm_percentage,
    )
    cleaned = to_dtype(cleaned, yr_input.your_header.dtype)
    sigproc_object.append_spectra(
        cleaned.get(),
        out_file,
    )


def master_cleaner(
    file: str,
    dispersion_measure: float,
    sigma: float = 4,
    gulp: int = 16384,
    flatten_to: Union[int, None] = None,
    channels_per_subband: int = 256,
    time_median_size: int = 32,
    modes_to_zero: int = 1,
    out_file: Union[str, None] = None,
) -> None:
    """
    Loops over a file, dedisperses the data, runs a subbanded mad filter,
    redisperses the data, and then saves to a new file.

    Args:
        file: the file to loop over

        sigma: sigma to remove outliers

        channels_per_subband: the number of channels for each subband

        keep_ends: keep the ends of file that can't be cleaned
                   because they can't be dedispersed
    """

    # fitter = get_fitter(fitter)
    out_file = get_outfile(file, out_file)
    yr_input = Your(file)

    if flatten_to is None:
        flatten_to = get_flatten_to(yr_input.your_header.nbits)

    wrt = Writer(yr_input, outname=out_file)
    sigproc_object = sigproc_object_from_writer(wrt)
    sigproc_object.write_header(out_file)

    if dispersion_measure > 0:
        logging.debug("Cleaning at DM %f", dispersion_measure)
        clean_dispersion(
            yr_input,
            dispersion_measure=dispersion_measure,
            sigma=sigma,
            gulp=gulp,
            flatten_to=flatten_to,
            channels_per_subband=channels_per_subband,
            time_median_size=time_median_size,
            modes_to_zero=modes_to_zero,
            out_file=out_file,
            sigproc_object=sigproc_object,
        )
    else:
        logging.debug("No DM given, cleaning at 0 DM")
        if BACKEND_GPU:
            clean_gpu(
                yr_input=yr_input,
                sigma=sigma,
                gulp=gulp,
                flatten_to=flatten_to,
                channels_per_subband=channels_per_subband,
                time_median_size=time_median_size,
                modes_to_zero=modes_to_zero,
                out_file=out_file,
                sigproc_object=sigproc_object,
            )
        else:
            clean_cpu(
                yr_input=yr_input,
                sigma=sigma,
                gulp=gulp,
                flatten_to=flatten_to,
                channels_per_subband=channels_per_subband,
                time_median_size=time_median_size,
                modes_to_zero=modes_to_zero,
                out_file=out_file,
                sigproc_object=sigproc_object,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mad_filter.py",
        description=textwrap.dedent(
            """Runs a Composite (MAD/FFT/Zero DM) filter
         over subbands of .fits/.fil"""
        ),
        epilog=__doc__,
        formatter_class=YourArgparseFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Set logging to DEBUG",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--file",
        help=".fil or .fits file to process",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-dm",
        "--dispersion_measure",
        help="Dispersion Measure to process the data; if 0, doesn't dedisp",
        type=float,
        default=0,
        required=False,
    )
    parser.add_argument(
        "-sig",
        "--sigma",
        help="Sigma at which to cut data",
        type=float,
        default=4.0,
        required=False,
    )
    parser.add_argument(
        "-chans_per_sub",
        "--channels_per_subband",
        help="Number of channels in each subband",
        type=int,
        default=256,
        required=False,
    )
    parser.add_argument(
        "-time_median_size",
        "--time_median_size",
        help="The length of kernel for median of median and median of MADs in time",
        type=int,
        default=32,
        required=False,
    )
    parser.add_argument(
        "-modes_to_zero",
        "--modes_to_zero",
        help="Number of modes to zero, 0 to preserve zerodm, -1 to median subtract",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "-flatten_to",
        "--flatten_to",
        help="Flatten data to this number. If `None`, sets the data 1/4"
        + " between zero and dtype max",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-g",
        "--gulp",
        help="Number of samples to process at each loop",
        type=int,
        default=16384,
        required=False,
    )
    parser.add_argument(
        "-gpu",
        "--gpu",
        help="Which GPU to use. Default is gpu 0, -1 for cpu",
        type=int,
        default=0,
        required=False,
    )
    # parser.add_argument(
    #    "--remove_ends",
    #    help="keep ends of file that cannot be cleaned",
    #     action="store_true",
    #     required=False,
    # )
    parser.add_argument(
        "-o",
        "--out_file",
        help="output file, default: input filename with _mad appended",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    LOGGING_FORMAT = "%(funcName)s - %(name)s - %(levelname)s - %(message)s"

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format=LOGGING_FORMAT,
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=LOGGING_FORMAT,
            handlers=[RichHandler(rich_tracebacks=True)],
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    from jess.JESS_filters_generic import mad_spectra_flat

    if args.gpu < 0 or not BACKEND_GPU:
        from jess.calculators import to_dtype
        from jess.dispersion import dedisperse, delay_lost
        from jess.JESS_filters import fft_mad, zero_dm, zero_dm_fft

        BACKEND_GPU = False
    else:
        from jess.calculators_cupy import to_dtype
        from jess.dispersion_cupy import dedisperse, delay_lost
        from jess.JESS_filters_cupy import fft_mad, zero_dm, zero_dm_fft

    master_cleaner(
        file=args.file,
        dispersion_measure=args.dispersion_measure,
        sigma=args.sigma,
        gulp=args.gulp,
        flatten_to=args.flatten_to,
        time_median_size=args.time_median_size,
        channels_per_subband=args.channels_per_subband,
        modes_to_zero=args.modes_to_zero,
        # remove_ends=args.remove_ends,
        out_file=args.out_file,
    )
