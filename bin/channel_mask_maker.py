#!/usr/bin/env python3
"""
Makes a channel mask for a given .fits/.fil

This is a cli wrapper for jess.channel_masks.channel_masker
"""

import argparse
import logging
import os
import textwrap

import numpy as np
from rich.logging import RichHandler
from scipy.signal import decimate
from your import Your
from your.utils.misc import YourArgparseFormatter

from jess.channel_masks import channel_masker

logger = logging.getLogger()


def file_saver(mask: np.ndarray, out_file: str) -> None:
    """
    Saves the mask to a file,
    """
    all_chans = np.array(range(0, len(mask)), dtype=int)

    np.savetxt(out_file, all_chans[mask], fmt="%d", delimiter=" ", newline=" ")


def channel_mask_maker(
    file: str,
    test: str,
    sigma: float = 4.0,
    start: int = 0,
    nspec: int = 65536,
    decimation_factor: int = None,
    fitter: str = "median_fitter",
    chans_per_fit: int = 40,
    flagger="z_score_flagger",
    flag_above: bool = True,
    flag_below: bool = True,
    out_file: str = None,
) -> None:
    """
    Reads data from the given file, does a statistical test,
    flags outlying channels.
    """

    yr_obj = Your(file)

    if yr_obj.your_header.nspectra < nspec + start:
        logger.warning(
            "Asked for %i spectra, starting %i, file has %i: using full file",
            nspec,
            start,
            yr_obj.your_header.nspectra,
        )
        start = 0
        nspec = yr_obj.your_header.nspectra
    dynamic_spectra = yr_obj.get_data(start, nspec)
    if decimation_factor is not None:
        dynamic_spectra = decimate(dynamic_spectra, decimation_factor, axis=0)

    mask = channel_masker(
        dynamic_spectra,
        test=test,
        sigma=sigma,
        fitter=fitter,
        chans_per_fit=chans_per_fit,
        flagger=flagger,
        flag_above=flag_above,
        flag_below=flag_below,
        show_plots=False,
    )

    if out_file is None:
        out_file = os.path.basename(file[0]).split(".")[0] + ".bad_chans"

    file_saver(mask, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="channel_mask_maker.py",
        description=textwrap.dedent(
            """Makes a list of channels to mask from .fits/.fil"""
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
        "-t",
        "--test",
        help="test to flag on, see epilog for list",
        type=str,
        default="90-10",
        required=False,
    )
    parser.add_argument(
        "-sig",
        "--sigma",
        help="Sigma at which to cut data, (z_score_flagger only)",
        type=float,
        default=4.0,
        required=False,
    )
    parser.add_argument(
        "-fitter",
        "--fitter",
        help="""fitter to detrend,
        [bspline_fitter, cheb_fitter, median_fitter(median), poly_fitter]""",
        type=str,
        default="median_fitter",
        required=False,
    )
    parser.add_argument(
        "-chans_per_fit",
        "--chans_per_fit",
        help="Number of channels for each fit degree of freedom.",
        type=int,
        default=40,
        required=False,
    )
    parser.add_argument(
        "-flagger",
        "--flagger",
        help="""Flagger to remove outliers,
        [z_score_flagger, dbscan_flagger]""",
        type=str,
        default="z_score_flagger",
        required=False,
    )
    parser.add_argument(
        "-nspec",
        "--num_spectra",
        help=textwrap.dedent(
            """Number of samples to process, starting at first sample."
                            (-1 for the whole file)"""
        ),
        type=int,
        default=65536,
        required=False,
    )
    parser.add_argument(
        "-deci",
        "--decimation_factor",
        help="Decimate in time by this factor",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-start",
        "--start",
        help="Spectra to start.",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "-flag_above",
        "--flag_above",
        help="""Flag values that are above median+sigma*stand dev.
            (z_score_flagger only)""",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "-flag_below",
        "--flag_below",
        help="""Flag values that are below median-sigma*stand dev.
            (z_score_flagger only)""",
        type=bool,
        default=True,
        required=False,
    )
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

    channel_mask_maker(
        args.file,
        args.test,
        args.sigma,
        args.start,
        args.num_spectra,
        args.decimation_factor,
        args.fitter,
        args.chans_per_fit,
        args.flagger,
        args.flag_below,
        args.flag_above,
        args.out_file,
    )
