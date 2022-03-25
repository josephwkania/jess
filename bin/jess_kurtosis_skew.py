#!/usr/bin/env python3
"""
Runs a Kurtosis and Skew filter over a .fits/.fil file
writing out a .fil
"""

import argparse
import logging
import os
import textwrap
from typing import Tuple, Union

from rich.logging import RichHandler
from rich.progress import track
from your import Writer, Your
from your.formats.filwriter import sigproc_object_from_writer
from your.utils.misc import YourArgparseFormatter

from jess.JESS_filters_generic import kurtosis_and_skew
from jess.scipy_cupy.stats import iqr_med

try:
    import cupy as xp

    from jess.calculators_cupy import flattner_median, to_dtype

    BACKEND_GPU = True

except ModuleNotFoundError:
    import numpy as xp

    from jess.calculators import flattner_median, to_dtype

    BACKEND_GPU = False


def get_outfile(file: str, out_file: str) -> str:
    """
    Makes the outfile name by:
    if no str is given -> append _mad to the original file name
    if str is given with an extension other than .fil -> assert error
    if str is given without extention -> add .fil
    """
    if not out_file:
        # if no out file is given, create the string
        path, file_ext = os.path.splitext(file[0])
        out_file = path + "_gauss.fil"
        logging.info("No outfile given, writing to %s", out_file)
        return out_file

    # if a file is given, make sure it has a .fil ext
    path, file_ext = os.path.splitext(out_file)
    if file_ext:
        assert file_ext == ".fil", f"I can only write .fil, you gave: {file_ext}!"
        return out_file

    out_file += ".fil"
    return out_file


def clean(
    yr_input: Your,
    samples_per_block: int,
    no_time_detrend: bool,
    winsorize_args: Union[Tuple, None],
    sigma: float,
    flatten_to: int,
    gulp: int,
    out_file: str,
    sigproc_object: sigproc_object_from_writer,
) -> None:
    """
    Run a kurtosis and skew filter on chunks of data `gulp` long.
    This wraps jess_filters_cupy.kurtosis_and_skew

     Args:
        yr_input: the your object of the file you want to clean

        samples_per_block: Number of time samples for each block

        winsorize_args: (std, channeles_per_fit) to winsorize m2

        sigma: Sigma at which to remove outliers

        gulp: The amount of data to process.

        flatten_to: make this the median out the out data.

        channels_per_subband: the number of channels for each MAD
                              subband

        out_file: name of the file to write out

        sigproc_obj: sigproc object to write out

    Returns:
        None
    """
    mask_chans: bool = True

    n_iter = 0
    total_flag = xp.zeros(3)
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

        if BACKEND_GPU:
            data = xp.asarray(data)

        _, mask, mask_percentage = kurtosis_and_skew(
            dynamic_spectra=data,
            samples_per_block=samples_per_block,
            sigma=sigma,
            winsorize_args=winsorize_args,
            nan_policy=None,
        )

        data = data.astype(xp.float32)
        data[mask] = xp.nan
        data, time_series = flattner_median(
            data, flatten_to=flatten_to, return_time_series=True
        )

        if mask_chans:
            means = xp.nanmean(data, axis=0)
            chan_noise, chan_mid = iqr_med(means, scale="normal", nan_policy="omit")
            chan_mask = xp.abs(means - chan_mid) > sigma * chan_noise
            mask += chan_mask
            chan_mask_percent = 100 * chan_mask.mean()

        mask_percentage_total = 100 * mask.mean()
        n_iter += 1
        total_flag += xp.asarray(
            [mask_percentage, chan_mask_percent, mask_percentage_total]
        )
        logging.info(
            "Gauss Flag %.2f%%, Chan Flag %.2f%%, Total %.2f%%",
            mask_percentage,
            chan_mask_percent,
            mask_percentage_total,
        )

        data[mask] = flatten_to

        if no_time_detrend:
            time_series -= xp.median(time_series)
            data += time_series[:, None]

        data = to_dtype(data, dtype=yr_input.your_header.dtype)

        if BACKEND_GPU:
            data = data.get()
        sigproc_object.append_spectra(data, out_file)

    total_flag /= n_iter
    logging.info(
        "Total: Gauss Flag %.2f%%, Chan Flag %.2f%%, Total %.2f%%",
        total_flag[0],
        total_flag[1],
        total_flag[2],
    )


def clean_fast(
    yr_input: Your,
    sigma: float,
    samples_per_block: int,
    winsorize_args: Union[Tuple, None],
    gulp: int,
    out_file: str,
    sigproc_object: sigproc_object_from_writer,
) -> xp.ndarray:
    """
    A striped down version clean that does not do the
    detrending in time and frequency.
    """

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

        if BACKEND_GPU:
            data = xp.asarray(data)

        _, mask, mask_percentage = kurtosis_and_skew(
            data,
            samples_per_block=samples_per_block,
            detrend=None,  # (median_fitter, 50),
            sigma=sigma,
            winsorize_args=winsorize_args,
            nan_policy=None,
        )

        means = xp.mean(data, axis=0)
        medians = xp.median(data, axis=0).astype(yr_input.your_header.dtype)
        diff = means - medians
        chan_noise, chan_mid = iqr_med(diff, scale="normal", nan_policy=None)
        chan_mask = xp.abs(diff - chan_mid) > sigma * chan_noise
        chan_mask_percent = chan_mask.mean()
        mask += chan_mask

        mask_percentage_total = 100 * mask.mean()
        n_iter += 1
        total_flag += xp.asarray(
            [mask_percentage, chan_mask_percent, mask_percentage_total]
        )
        logging.info(
            "Gauss Flag %.2f%%, Chan Flag %.2f%%, Total %.2f%%",
            mask_percentage,
            chan_mask_percent,
            mask_percentage_total,
        )

        # https://stackoverflow.com/a/61485863
        xp.putmask(data, mask, medians)

        if BACKEND_GPU:
            data = data.get()
        sigproc_object.append_spectra(data, out_file)

    total_flag /= n_iter
    logging.info(
        "Total: Gauss Flag %.2f%%, Chan Flag %.2f%%, Total %.2f%%",
        total_flag[0],
        total_flag[1],
        total_flag[2],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="kurtosis_skew_filter.py",
        description=textwrap.dedent(
            """A Kurtosis and Skew filter
         on a .fits/.fil"""
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
        "-sig",
        "--sigma",
        help="Sigma at which to cut data",
        type=float,
        default=4.0,
        required=False,
    )
    parser.add_argument(
        "-spb",
        "--samples_per_block",
        help="Number of channels in each subband",
        type=int,
        default=64,
        required=False,
    )
    parser.add_argument(
        "-flatten_to",
        "--flatten_to",
        help="Flatten data to this number (Only used for mad_specta_flat)",
        type=int,
        default=64,
        required=False,
    )
    parser.add_argument(
        "-no_td",
        "--no_time_detrend",
        help="No time series detrend (for low DM sources)",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-winsorize_args",
        "--winsorize_args",
        help="""Winsorize 2nd moment along freq axis (std, chan_per_fit);
        if None, don't Winsorize""",
        nargs="+",
        type=float,
        default=(5, 40),
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
        "-fast",
        "--fast",
        help="Run the fast version that does a less robust filtering/deterend",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        help="output file, default: input filename with _gauss appended",
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

    outfile = get_outfile(file=args.file, out_file=args.out_file)
    yrinput = Your(args.file)
    wrt = Writer(yrinput, outname=outfile)
    sigproc_obj = sigproc_object_from_writer(wrt)
    sigproc_obj.write_header(outfile)
    if args.winsorize_args[0] == -1:
        WINSORIZE = None
    else:
        WINSORIZE = args.winsorize_args

    if args.fast:
        clean_fast(
            yr_input=yrinput,
            samples_per_block=args.samples_per_block,
            winsorize_args=None,
            sigma=args.sigma,
            gulp=args.gulp,
            out_file=outfile,
            sigproc_object=sigproc_obj,
        )
    else:
        clean(
            yr_input=yrinput,
            samples_per_block=args.samples_per_block,
            no_time_detrend=args.no_time_detrend,
            winsorize_args=WINSORIZE,
            flatten_to=args.flatten_to,
            sigma=args.sigma,
            gulp=args.gulp,
            out_file=outfile,
            sigproc_object=sigproc_obj,
        )
