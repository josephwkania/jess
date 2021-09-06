#!/usr/bin/env python3
"""
Runs a MAD filter over fits/fil.

Can dedisperse the data to a given DM to
protect bright/narrow pulses.
"""

import argparse
import logging
import os
import textwrap

from rich.logging import RichHandler
from rich.progress import track
from your import Your
from your.formats.filwriter import make_sigproc_object

# from your.utils.math import primes
from your.utils.misc import YourArgparseFormatter

try:
    import cupy as cp
    from jess.calculators_cupy import to_dtype
    from jess.dispersion_cupy import dedisperse, delay_lost
    from jess.JESS_filters_cupy import fft_mad, zero_dm_fft, mad_spectra_flat

    BACKEND_GPU = True
except ModuleNotFoundError:
    from jess.calculators import to_dtype
    from jess.dispersion import dedisperse, delay_lost
    from jess.JESS_filters import fft_mad, zero_dm_fft, mad_spectra_flat

    BACKEND_GPU = False


logger = logging.getLogger()


def get_outfile(file: str, out_file: str) -> str:
    """
    Makes the outfile name by:
    if no str is given -> append _mad to the original file name
    if str is given with an extension other than .fil -> assert error
    if str is given without extention  -> add .fil
    """
    if not out_file:
        # if no out file is given, create the string
        path, file_ext = os.path.splitext(file[0])
        out_file = path + "_remastered.fil"
        logger.info("No outfile given, writing to %s", out_file)
        return out_file

    # if a file is given, make sure it has a .fil ext
    path, file_ext = os.path.splitext(out_file)
    if file_ext:
        assert file_ext == ".fil", "I can only write .fil, you gave: %s!" % file_ext
        return out_file

    out_file += ".fil"
    return out_file


def clean_cpu(
    yr_input: object,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    median_time_kernel: int,
    modes_to_zero: int,
    out_file: str,
    sigproc_object: object,
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

    for j in track(range(0, yr_input.your_header.nspectra, gulp)):
        logging.debug("Cleaning samples starting at %i", j)
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        # cleaned = fft_mad(cp.asarray(data), sigma=sigma, frame=channels_per_subband)
        cleaned = mad_spectra_flat(
            data,
            frame=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            median_time_kernel=median_time_kernel,
            return_same_dtype=False,
        )
        cleaned = fft_mad(
            cleaned, sigma=sigma, frame=channels_per_subband, return_same_dtype=False
        )

        if modes_to_zero > 0:
            cleaned = zero_dm_fft(
                cleaned, modes_to_zero=modes_to_zero, return_same_dtype=False
            )

        # Keep full precision until done
        cleaned = to_dtype(cleaned, dtype=yr_input.your_header.dtype)

        sigproc_object.append_spectra(cleaned, out_file)


def clean_gpu(
    yr_input: object,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    median_time_kernel: int,
    modes_to_zero: int,
    out_file: str,
    sigproc_object: object,
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

    for j in track(range(0, yr_input.your_header.nspectra, gulp)):
        logging.debug("Cleaning samples starting at %i", j)
        if j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)

        # cleaned = fft_mad(cp.asarray(data), sigma=sigma, frame=channels_per_subband)
        cleaned = mad_spectra_flat(
            cp.asarray(data),
            frame=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            median_time_kernel=median_time_kernel,
            return_same_dtype=False,
        )
        cleaned = fft_mad(
            cleaned, sigma=sigma, frame=channels_per_subband, return_same_dtype=False
        )

        if modes_to_zero > 0:
            cleaned = zero_dm_fft(
                cleaned, modes_to_zero=modes_to_zero, return_same_dtype=False
            )

        # Keep full precision until done
        cleaned = to_dtype(cleaned, dtype=yr_input.your_header.dtype)

        sigproc_object.append_spectra(cleaned.get(), out_file)


def clean_dispersion(
    yr_input: object,
    dispersion_measure: float,
    sigma: float,
    gulp: int,
    flatten_to: int,
    channels_per_subband: int,
    median_time_kernel: int,
    out_file: str,
    sigproc_object: object,
) -> None:
    """
    Run the MAD filter on chunks of data without dedispersing it.

     Args:
        yr_input: the your object of the file you want to clean

        dispersion_measure: The dispersion measure to dedisperse
                            the data to

        sigma: Sigma at which to remove outliers

        gulp: The amount of data to process.

        latten_to: make this the median out the out data.

        channels_per_subband: the number of channels for each MAD
                              subband

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

    # add data that can't be dispersed
    # because its at the start
    # if not remove_ends:
    cleaned = mad_spectra_flat(
        cp.asarray(yr_input.get_data(0, samples_lost)),
        frame=channels_per_subband,
        sigma=sigma,
        flatten_to=flatten_to,
        median_time_kernel=median_time_kernel,
    )
    sigproc_object.append_spectra(cleaned.get(), out_file)

    # loop through all the data we can dedisperse
    for j in track(range(0, yr_input.your_header.nspectra, gulp)):
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
        dedisp[0:-samples_lost, :] = mad_spectra_flat(
            dedisp[0:-samples_lost, :],
            frame=channels_per_subband,
            sigma=sigma,
            flatten_to=flatten_to,
            median_time_kernel=median_time_kernel,
        )
        redisip = dedisperse(
            dedisp, -dispersion_measure, yr_input.your_header.tsamp, yr_input.chan_freqs
        )
        redisip = redisip.astype(yr_input.your_header.dtype)
        sigproc_object.append_spectra(
            redisip[samples_lost:-samples_lost, :].get(), out_file
        )

    # add data that can't be dispersed
    # because its at the end

    # if not remove_ends:
    cleaned = mad_spectra_flat(
        cp.asarray(
            yr_input.get_data(
                yr_input.your_header.nspectra - samples_lost, samples_lost
            )
        ),
        frame=channels_per_subband,
        sigma=sigma,
        flatten_to=flatten_to,
        median_time_kernel=median_time_kernel,
    )
    sigproc_object.append_spectra(
        cleaned.get(),
        out_file,
    )


def master_cleaner(
    file: str,
    dispersion_measure: float,
    sigma: float = 3,
    gulp: int = 16384,
    flatten_to: int = 64,
    channels_per_subband: int = 256,
    median_time_kernel: int = 0,
    modes_to_zero: int = 6,
    out_file: str = None,
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

    if dispersion_measure > 0:
        raise NotImplementedError("This isn't working quite right yet")
        logging.debug("Cleaning at DM %f", dispersion_measure)
        clean_dispersion(
            yr_input,
            dispersion_measure=dispersion_measure,
            sigma=sigma,
            gulp=gulp,
            flatten_to=flatten_to,
            channels_per_subband=channels_per_subband,
            median_time_kernel=median_time_kernel,
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
                median_time_kernel=median_time_kernel,
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
                median_time_kernel=median_time_kernel,
                modes_to_zero=modes_to_zero,
                out_file=out_file,
                sigproc_object=sigproc_object,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mad_filter.py",
        description=textwrap.dedent(
            """Runs a Medain Absolute Deviation (MAD) filter
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
        default=3.0,
        required=False,
    )
    parser.add_argument(
        "-channels_per_subband",
        "--channels_per_subband",
        help="Number of channels in each subband",
        type=int,
        default=256,
        required=False,
    )
    parser.add_argument(
        "-median_time_kernel",
        "---median_time_kernel",
        help="The length of kernel for median of median and median of MADs in time",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--modes_to_zero",
        help="Number of modes to zero",
        type=int,
        default=6,
        required=False,
    )
    parser.add_argument(
        "-flatten_to",
        "--flatten_to",
        help="Flatten data to this number",
        type=int,
        default=64,
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

    master_cleaner(
        file=args.file,
        dispersion_measure=args.dispersion_measure,
        sigma=args.sigma,
        gulp=args.gulp,
        flatten_to=args.flatten_to,
        median_time_kernel=args.median_time_kernel,
        channels_per_subband=args.channels_per_subband,
        modes_to_zero=args.modes_to_zero,
        # remove_ends=args.remove_ends,
        out_file=args.out_file,
    )