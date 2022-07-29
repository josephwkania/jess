#!/usr/bin/env python3
"""
This runs two Fourier Filters.
The first filter is fft_mad which removes narrow band RFI.
For details see the docstring for jess.JESS_filters.fft_mad

The second filter is zero_dm_fft which removes low frequency
modes for each time series, reducing the effects of broadband rfi.
See jess.JESS_filters.zero_dm_fft
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
from your.utils.misc import YourArgparseFormatter

try:
    import cupy as xp

    from jess.calculators_cupy import to_dtype
    from jess.JESS_filters_cupy import fft_mad, zero_dm, zero_dm_fft

    BACKEND_GPU = True
except ModuleNotFoundError:
    xp = np

    from jess.calculators import to_dtype
    from jess.JESS_filters import fft_mad, zero_dm, zero_dm_fft

    BACKEND_GPU = False


logger = logging.getLogger()


def get_outfile(file: str, out_file: Union[None, str]) -> str:
    """
    Makes the outfile name by:
    if no str is given -> append _zeroDM to the original file name
    if str is given with an extension other than .fil -> Value error
    if str is given without extension -> add .fil
    """
    if not out_file:
        # if no out file is given, create the string
        path, file_ext = os.path.splitext(file[0])
        out_file = path + "_fft_cleaned.fil"
        logger.info("No outfile given, writing to %s", out_file)
        return out_file

    # if a file is given, make sure it has a .fil ext
    path, file_ext = os.path.splitext(out_file)
    if file_ext:
        if not file_ext == ".fil":
            raise ValueError(f"I can only write .fil, you gave: {file_ext}!")
        return out_file

    out_file += ".fil"
    return out_file


def clean(
    yr_input: Your,
    gulp: int,
    sigproc_object: sigproc_object_from_writer,
    out_file: str,
    sigma: float,
    bad_chans: np.ndarray,
    modes_to_zero: int,
) -> None:
    """
    Loops over the given yr_input file, cleans the data using GPU backend,
    writes to the out_file

    Args:
        yr_input: the your object for the input file

        gulp: amount of data to process each loop

        sigproc_object: the object for the outfile

        out_file: name for the outfile

        sigma: sigma at which to clip narrow band Fourier Components

        bad_chans: channels to be removed by zeroing non-DC components

        modes_to_zero: number of Fourier modes to zero for zero-dm'ing

        outfile: file to write to

    Returns:
        None
    """

    # loop through all the data
    n_iter = 0
    total_flag = xp.zeros(3)
    for j in track(
        range(0, yr_input.your_header.nspectra, gulp),
        description="Cleaning File",
        transient=True,
        refresh_per_second=1,
    ):
        logging.debug("Cleaning samples starting at %i", j)
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        # can't do this with the cupy zero dmer
        # data = np.ma.array(data, mask=np.broadcast_to(mask, data.shape))

        if BACKEND_GPU:
            data = xp.asarray(data)

        # use one bandpass to prevent jumps
        if modes_to_zero > 0:
            logging.debug("Creating bandpass")
            # I had np.ma.mean here before but
            # this isn't a masked array
            bandpass = xp.mean(data, axis=0)

        data, _, fft_percentage = fft_mad(
            data, sigma=sigma, bad_chans=bad_chans, return_same_dtype=False
        )

        if modes_to_zero == 1:
            logging.debug("Zero DMing: Subtracting Mean")
            data, dm_percentage = zero_dm(data, bandpass, return_same_dtype=False)
        elif modes_to_zero > 1:
            logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
            data, dm_percentage = zero_dm_fft(
                data, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
            )
        else:
            dm_percentage = xp.asarray((0))

        sum_flagged = fft_percentage + dm_percentage
        total_flag += xp.asarray((fft_percentage, dm_percentage, sum_flagged))
        n_iter += 1
        logging.info(
            "FFT Flag %.2f%%, Highpass %.2f%%, Total %.2f%%",
            fft_percentage,
            dm_percentage,
            sum_flagged,
        )

        data = to_dtype(data, dtype=yr_input.your_header.dtype)
        if BACKEND_GPU:
            data = data.get()
        sigproc_object.append_spectra(data, out_file)

    total_flag /= n_iter
    logging.info(
        "Total: FFT Flag %.2f%%, Highpass %.2f%%, Total %.2f%%",
        total_flag[0],
        total_flag[1],
        total_flag[2],
    )


def fft_cleaner(
    file: str,
    mask_file: str = None,
    gulp: int = 16384,
    sigma: float = 3,
    modes_to_zero: int = 6,
    out_file: str = None,
) -> None:
    """
    Loops over a file, mad filtering the FFT power,
    then zero-dm'ing using a Fourier filter. Uses one
    bandpass over all the data, as not to create jumps.
    This could be problematic if there are very large
    changes in antenna temp

    Args:
        file: the file to loop over

        mask_file: the file that contains channels to mask

        gulp: amount of data to process each loop

        sigma: sigma at which to clip narrowband Fourier Components

        modes_to_zero: number of Fourier modes to zero for zero-dm'ing

        outfile: file to write to, if not give, appends _zeroDM to the
                 input file name.
    """

    yr_input = Your(file)
    out_file = get_outfile(file, out_file)

    if mask_file is not None:
        bad_chans = np.loadtxt(mask_file, dtype=int)
        logging.debug("Bad Channels: %s", np.array2string(bad_chans))
        mask = np.zeros(yr_input.your_header.nchans, dtype=bool)
        mask[bad_chans] = True
        logging.debug("Masking: %s", np.array2string(mask))
    else:
        bad_chans = None
        mask = np.zeros(yr_input.your_header.nchans, dtype=bool)
        logging.debug("No mask file given")

    wrt = Writer(yr_input, outname=out_file)
    sigproc_object = sigproc_object_from_writer(wrt)
    sigproc_object.write_header(out_file)

    clean(
        yr_input=yr_input,
        gulp=gulp,
        sigproc_object=sigproc_object,
        out_file=out_file,
        sigma=sigma,
        bad_chans=bad_chans,
        modes_to_zero=modes_to_zero,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="zero_dm.py",
        description=textwrap.dedent(
            """Runs two Fourier filters over .fits/.fil,
            allowing the user to give a .bad_chans file"""
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
        "-m",
        "--mask",
        help="Channel Mask to apply",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gulp",
        help="Number of samples to process at each loop",
        type=int,
        default=32768,
        required=False,
    )
    parser.add_argument(
        "-sig",
        "--sigma",
        help="Sigma at which to excises FFT",
        type=float,
        default=3,
        required=False,
    )
    parser.add_argument(
        "-modes_to_zero",
        "--modes_to_zero",
        help="Number of Fourier modes to zero, 0=No filter, 1=mean subtract, etc",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--out_file",
        help="output file, default: input filename with _fft appended",
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

    fft_cleaner(
        file=args.file,
        mask_file=args.mask,
        gulp=args.gulp,
        sigma=args.sigma,
        modes_to_zero=args.modes_to_zero,
        out_file=args.out_file,
    )
