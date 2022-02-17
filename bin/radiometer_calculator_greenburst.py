#!/usr/bin/env python3
"""
Calcualtes radiometer noise for a give search mode data file
Caculated from raw sampling time to 2**max_boxcar_width
"""

import argparse
import logging

import numpy as np
from scipy import signal
from your import Your


def get_timeseries(input_file, block_size=2**14):
    """
    Makes a zero DM timeseries for a given file
    args:
    input file - file to read
    block_size - number of specta to loop over
    return:
    timeseries
    """
    yr = Your(input_file)
    timeseries = np.zeros(yr.your_header.nspectra, dtype=np.float64)

    for j in np.arange(0, yr.your_header.nspectra, block_size):
        if j + block_size > yr.your_header.nspectra:
            block_size = yr.your_header.nspectra - j
        timeseries[j : j + block_size] = np.array(yr.get_data(j, block_size)).mean(
            axis=1
        )

    return timeseries


def get_stds(input_file, headless):
    """
    Computes the Standard Deviations of the 0DM timeseries

    args:
    input_file - the search mode file to calculate the standard diviations
    max_boxcar_width = 7 largest boxcar will be 2**max_boxvar_width
    headless - if True, don't print to terminmal or show plot
    """
    timeseries = get_timeseries(input_file)
    # timeseries = cp.array(np.random.normal(0, 1, len(timeseries)))
    # can use the above to test

    assert (
        len(timeseries) < 2**7
    ), f"""The file length of
    {len(timeseries)} is shorter than the max boxcar width of {2**7}"""

    # powers_of_two = [2, 4, 7] # get boxcars for 2^0, 2^2, 2^4, 2^7
    powers_of_two = np.arange(1, 7, 1)  # get boxcars for 2**[0,1, ... , 7]
    stds = np.zeros(len(powers_of_two) + 1, dtype=np.float64)
    stds[0] = np.std(timeseries)
    for j, k in enumerate(powers_of_two):
        kernal = np.array(signal.boxcar(2**k) / 2**k)
        stds[j + 1] = np.std(np.convolve(timeseries, kernal, "valid"))

    widths = 2 ** np.insert(powers_of_two, [0], 0)

    stds_dic = {j: k for j, k in zip(widths, stds)}
    logging.debug(f"Boxcarwidths: stad dev {stds_dic}")

    return stds_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_header.py",
        description="""Read header from psrfits/filterbank files
         and print the unified header""",
    )
    parser.add_argument(
        "-f",
        "--files",
        help="psrfits or filterbank files to process",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--headless",
        help="Don't print info to console or show images",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")

    values = parser.parse_args()
    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    get_stds(values.files, values.headless)
