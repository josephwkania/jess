#!/usr/bin/env python3
"""
This runs two Fourier Filters.
The first filter is mad_fft which removes narrow band RFI.
For details see the docstring for jess.JESS_filters.mad_fft

The second filter is zero_dm_fft which removes low frequency
modes for each time series, reducing the effects of broadband rfi.
See jess.JESS_filters.zero_dm_fft
"""

import argparse
import logging
import os
import textwrap

import numpy as np
from rich.logging import RichHandler
from rich.progress import track
from your import Your
from your.formats.filwriter import make_sigproc_object
from your.utils.misc import YourArgparseFormatter

from jess.JESS_filters_cupy import mad_fft, zero_dm_fft

logger = logging.getLogger()


def get_outfile(file: str, out_file: str) -> str:
    """
    Makes the outfile name by:
    if no str is given -> append _zeroDM to the original file name
    if str is given with an extension other than .fil -> Value error
    if str is given without extention  -> add .fil
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
            raise ValueError("I can only write .fil, you gave: %s!" % file_ext)
        return out_file

    out_file += ".fil"
    return out_file


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

        sigma: simga at which to clip narrowband Fourier Components

        modes_to_zero: number of Fourier modes to zero for zero-dm'ing

        outfile: file to write to, if not give, appends _zeroDM to the
                 input file name.
    """

    yr_input = Your(file)
    out_file = get_outfile(file, out_file)

    if mask_file is not None:
        bad_chans = np.loadtxt(mask_file, dtype=int)
        mask = np.zeros(yr_input.your_header.nchans, dtype=bool)
        mask[bad_chans] = True
        logging.debug("Masking %i", mask)
    else:
        bad_chans = None
        mask = np.zeros(yr_input.your_header.nchans, dtype=bool)
        logging.debug("No mask file given")

    sigproc_object = make_sigproc_object(
        rawdatafile=out_file,
        source_name=yr_input.your_header.source_name,
        nchans=yr_input.your_header.nchans,
        foff=yr_input.your_header.foff,  # MHz
        fch1=yr_input.your_header.fch1,  # MHz
        tsamp=yr_input.your_header.tsamp,  # seconds
        tstart=yr_input.your_header.tstart,  # MJD
        # src_raj=yr_input.src_raj,  # HHMMSS.SS
        # src_dej=yr_input.src_dej,  # DDMMSS.SS
        # machine_id=yr_input.your_header.machine_id,
        # nbeams=yr_input.your_header.nbeams,
        # ibeam=yr_input.your_header.ibeam,
        nbits=yr_input.your_header.nbits,
        # nifs=yr_input.your_header.nifs,
        # barycentric=yr_input.your_header.barycentric,
        # pulsarcentric=yr_input.your_header.pulsarcentric,
        # telescope_id=yr_input.your_header.telescope_id,
        # data_type=yr_input.your_header.data_type,
        # az_start=yr_input.your_header.az_start,
        # za_start=yr_input.your_header.za_start,
    )
    sigproc_object.write_header(out_file)
    bandpass = None
    # loop through all the data
    for j in track(range(0, yr_input.your_header.nspectra, gulp)):
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        data = np.ma.array(data, mask=np.broadcast_to(mask, data.shape))
        # use one bandpass to prevent jumps
        if bandpass is None:
            logging.debug("Creating bandpass")
            bandpass = np.ma.mean(data, axis=0)
        data = mad_fft(data, sigma=sigma)
        if modes_to_zero is not None:
            logging.debug("Zero DMing")
            data = zero_dm_fft(data, bandpass, modes_to_zero=modes_to_zero)
        sigproc_object.append_spectra(data, out_file)
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="zero_dm.py",
        description=textwrap.dedent(
            """Runs a zero dm filter over .fits/.fil,
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
        "--gulp",
        help="Number of samples to process at each loop",
        type=int,
        default=32768,
        required=False,
    )
    parser.add_argument(
        "--sigma",
        help="Sigma at which to excises FFT",
        type=float,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--modes_to_zero",
        help="Number of modes to zero",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--out_file",
        help="output file, default: input filename with _zeroDM appended",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    LOGGING_FORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

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
        args.file,
        args.mask,
        args.gulp,
        args.sigma,
        args.modes_to_zero,
        args.out_file,
    )
