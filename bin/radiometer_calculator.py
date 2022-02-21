#!/usr/bin/env python3
"""
Calculates radiometer noise for a give search mode data file
Calculated from raw sampling time to 2**max_boxcar_width
"""

import argparse
import logging
from builtins import ValueError

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from scipy import signal
from your import Your

from jess.scipy_cupy.stats import median_abs_deviation


def get_timeseries(input_file, block_size=2**14, nspectra=-1, max_boxcar_width=8):
    """
    Makes a zero DM timeseries for a given file
    args:
    input file - file to read
    block_size - number of specta to loop over
    return:
    timeseries
    """
    yr = Your(input_file)

    if nspectra == -1:
        end = yr.your_header.nspectra
    else:
        end = 2**nspectra
    timeseries = cp.zeros(end, dtype=np.float64)
    if 2**max_boxcar_width > 2 * end:
        raise ValueError(
            f"""2*number samples to be processed:
        {2*end} is shorter than the max boxcar width of {2**max_boxcar_width}"""
        )

    for j in track(
        np.arange(0, end, block_size), description="Getting Timeseries", transient=True
    ):
        if j + block_size > end:
            block_size = end - j
        timeseries[j : j + block_size] = cp.array(
            signal.detrend(yr.get_data(j, block_size)).mean(axis=1)
        )

    return timeseries, yr.your_header.native_tsamp


def get_stds(
    input_file: str,
    max_boxcar_width: int = 8,
    number_spectra: int = 14,
    headless: bool = False,
) -> dict:
    """
    Computes the Standard Deviations of the 0DM timeseries

    args:
    input_file - the search mode file to calculate the standard deviations
    max_boxcar_width - largest boxcar will be 2**max_boxvar_width
    headless - if True, don't print to terminmal or show plot
    """
    timeseries, tsamp = get_timeseries(
        input_file, nspectra=number_spectra, max_boxcar_width=max_boxcar_width
    )
    # timeseries = cp.array(np.random.normal(0, 1, len(timeseries)))
    # can use the above to test

    powers_of_two = np.arange(1, max_boxcar_width + 1, 1)
    stds = cp.zeros(len(powers_of_two) + 1, dtype=np.float64)
    mads = cp.zeros(len(powers_of_two) + 1, dtype=np.float64)
    stds[0] = cp.std(timeseries)
    mads[0] = median_abs_deviation(timeseries, scale="normal")

    for j, k in enumerate(powers_of_two):
        kernal = cp.array(signal.boxcar(2**k) / 2**k)
        conv = cp.convolve(timeseries, kernal, "valid")
        # conv = signal.detrend(conv.get())
        stds[j + 1] = cp.std(conv)
        mads[j + 1] = median_abs_deviation(conv, scale="normal")

    widths = 2 ** np.insert(powers_of_two, [0], 0)
    stds = stds.get()  # Don't need cupy
    mads = mads.get()

    stds_dic = {width: [std, mad] for width, std, mad in zip(widths, stds, mads)}
    logging.debug("Boxcarwidths: std-dev mad %f", stds_dic)

    if not headless:
        console = Console()
        table = Table(show_header=True, header_style="bold red", box=box.DOUBLE_EDGE)
        table.add_column("Boxcar Width", justify="right")
        table.add_column("Stand. Dev")
        table.add_column("Median Abs Dev")
        for width, std, mad in zip(widths, stds, mads):
            table.add_row(f"{width}", f"{std:.4e}", f"{mad:.4e}")
        console.print(table)

        fig, axs = plt.subplots(2, constrained_layout=True)
        times = widths * tsamp
        fig.suptitle("Observed Vs Ideal Radiometer Noise")
        axs[0].plot(times, stds, "-.", label="Stand Dev")
        axs[0].plot(times, mads, "*", label="Median Abs Dev")
        # axs[0].xaxis.tick_top()
        axs[0].plot(widths * tsamp, mads[0] / np.sqrt(widths), label="Gauss Noise")
        axs[0].set_ylabel("Stand. Dev.")
        axs[0].set_xlabel("Boxcar Width [Second]")
        axs[0].legend()

        axs[1].set_xscale("log", basex=2)
        axs[1].set_yscale("log", basey=2)
        axs[1].plot(widths, stds, "-.", label="Stand Dev")
        axs[1].plot(widths, mads, "*", label="Median Abs Dev")
        axs[1].plot(widths, mads[0] / np.sqrt(widths), label="Guass Noise")
        axs[1].set_ylabel("Stand. Dev.")
        axs[1].set_xlabel("Boxcar Width [Sample]")

        plt.show()

    return stds_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_header.py",
        description="Show Radiometer noise for the zero DM time series",
        epilog=__doc__,
    )
    parser.add_argument(
        "-f",
        "--files",
        help="psrfits or filterbank files to process",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-max",
        "--max_boxcar_width",
        help="Log2 of the max width of the boxcar in number of samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-nspectra",
        "--number_spectra",
        help="Log2 of the number of samples to process. [Default = -1 = entire file]",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--headless",
        help="Don't print info to console or show images",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")

    values = parser.parse_args()
    LOGGING_FORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if values.verbose:
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
    get_stds(
        values.files, values.max_boxcar_width, values.number_spectra, values.headless
    )
