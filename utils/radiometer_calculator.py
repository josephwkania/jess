#!/usr/bin/env python3
"""
Calcualtes radiometer noise for a give search mode data file
Caculated from raw sampling time to 2**max_boxcar_width
"""

import argparse
import logging
import cupy as cp
import numpy as np
from rich.logging import RichHandler
from rich.progress import track
from scipy import signal, stats
import sys
from your import Your


def get_timeseries(input_file, block_size=2 ** 14):
    """
    Makes a zero DM timeseries for a given file
    args:
    input file - file to read
    block_size - number of specta to loop over
    return:
    timeseries
    """
    yr = Your(input_file)
    timeseries = cp.zeros(yr.your_header.nspectra, dtype=np.float64)

    for j in track(np.arange(0, yr.your_header.nspectra, block_size)):
        if j + block_size > yr.your_header.nspectra:
            block_size = yr.your_header.nspectra - j
        timeseries[j : j + block_size] = cp.array(yr.get_data(j, block_size)).mean(
            axis=1
        )

    return timeseries


def get_stds(input_file, max_boxcar_width):
    """
    Computes the Standard Deviations of the 0DM timeseries

    args:
    input_file - the search mode file to calculate the standard diviations
    max_boxcar_width - largest boxcar will be 2**max_boxvar_width
    """
    timeseries = get_timeseries(input_file)

    if len(timeseries) < 2 ** max_boxcar_width:
        logging.error(
            f"The file length of {len(timeseries)} is shorter than the max boxcar width of {2**max_boxcar_width}"
        )
        sys.exit()

    powers_of_two = np.arange(1, max_boxcar_width, 1)
    stds = cp.zeros(len(powers_of_two) + 1, dtype=np.float64)
    stds[0] = cp.std(timeseries)
    for j, k in enumerate(powers_of_two):
        kernal = cp.array(signal.boxcar(2 ** k) / 2 ** k)
        stds[j + 1] = cp.std(cp.convolve(timeseries, kernal, "valid"))
    stds = stds.get()  # Don't need cupy

    logging.debug(f"Boxcar widths: {2**np.insert(powers_of_two,[0],0)}")
    logging.debug(f"STDs: {stds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_header.py",
        description="Read header from psrfits/filterbank files and print the unified header",
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
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")

    values = parser.parse_args()
    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if values.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format=logging_format,
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=logging_format,
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    get_stds(values.files, values.max_boxcar_width)
