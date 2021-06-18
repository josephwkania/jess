#!/usr/bin/env python3
"""
Makes a channel mask for a given .fits/.fil
"""
import argparse
import logging
import textwrap

import numpy as np
from rich.logging import RichHandler
from scipy import stats
from your import Your
from your.utils.misc import YourArgparseFormatter

from jess.calculators import shannon_entropy, preprocess
from jess.fitters import bspline_fitter, cheb_fitter, poly_fitter

logger = logging.getLogger()


def get_fitter(fitter: str) -> object:
    """
    Get the fitter object for a given string

    Args:
        fitter: string with the selection of cheb_fitter, poly_fitter, or bspline_fitter

    return:
        corresponding fitter object
    """
    if fitter == "cheb_fitter":
        return cheb_fitter
    if fitter == "poly_fitter":
        return poly_fitter
    if fitter == "bspline_fitter":
        return bspline_fitter

    raise ValueError(f"You didn't give a valid fitter type! (Given {fitter})")


def stat_test(data: np.ndarray, which_test: str) -> np.ndarray:
    """
    Runs the statistical tests
    Should have the same tests as rfi_viewer.py
    """
    which_test = which_test.lower()
    if which_test == "98-2":
        top_quant, bottom_quant = np.quantile(data, [0.98, 0.02], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "91-9":
        top_quant, bottom_quant = np.quantile(data, [0.91, 0.09], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "90-10":
        top_quant, bottom_quant = np.quantile(data, [0.90, 0.01], axis=0)
        test = top_quant - bottom_quant
    elif which_test == "75-25":
        test = stats.iqr(data, axis=0)
    elif which_test == "anderson-darling":
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        # self.hor_test_p = np.zeros(num_samps)
        for ichan in range(0, num_freq):
            test[ichan], _, _ = stats.anderson(data[:, ichan], dist="norm")
    elif which_test == "d'angostino":
        test, _ = stats.normaltest(data, axis=1)
    elif which_test == "jarque-bera":
        _, num_freq = data.shape
        if num_freq < 2000:
            logging.warning(
                "Jarque-Bera requires > 2000 points, given %i channels", num_freq
            )
        test = np.zeros(num_freq)
        for ichan in range(0, num_freq):
            test[ichan], _ = stats.jarque_bera(
                data[:, ichan],
            )
    elif which_test == "kurtosis":
        test = stats.kurtosis(data, axis=0)
    elif which_test == "lilliefors":
        # I don't take into account the change of dof when calculating the p_value
        # The test stattic is the same as statsmodels lilliefors
        num_freq, num_samps = data.shape
        test = np.zeros(num_freq)
        data_0, _ = preprocess(data)
        for j in range(0, num_freq):
            test[j], _ = stats.kstest(data_0[j, :], "norm")
    elif which_test == "mad":
        test = stats.median_abs_deviation(data, axis=0)
    elif which_test == "mean":
        test = np.mean(data, axis=0)
    elif which_test == "midhing":
        top_quant, bottom_quant = np.quantile(data, [0.75, 0.25], axis=0)
        test = (bottom_quant + top_quant) / 2.0
    elif which_test == "shannon_entropy":
        test = shannon_entropy(data, axis=0)
    elif which_test == "shapiro_wilk":
        _, num_freq = data.shape
        test = np.zeros(num_freq)
        # test_p = np.zeros(num_freq)
        for k in range(0, num_freq):
            test, _ = stats.shapiro(data[:, k])
    elif which_test == "skew":
        test = stats.skew(data, axis=0)
    elif which_test == "stand-dev":
        test = np.std(data, axis=0)
    elif which_test == "trimean":
        top_quant, middle_quant, bottom_quant = np.quantile(
            data, [0.75, 0.50, 0.25], axis=0
        )
        test = (top_quant + 2.0 * middle_quant + bottom_quant) / 4.0
    else:
        raise ValueError(f"You gave {which_test}, which is not avaliable.")

    return test


def file_saver(mask: list[bool], out_file: str):
    pass


def channel_masker(
    file: str,
    sigma: float = 3.0,
    start: int = 0,
    nspec: int = 65536,
    fitter: str = "bspline_fitter",
    out_file: str = None,
):
    """
    Reads data from the given file, does a statistical test,
    flags outlying channels.
    """
    fitter = get_fitter(fitter)

    yr_obj = Your(file)

    if yr_obj.your_header.nspectra > nspec + start:
        logger.warning(
            "Asked for %i spectra, starting %i, file has %i: using full file",
            nspec,
            start,
            yr_obj.your_header.nspectra,
        )
        start = 0
        nspec = yr_obj.your_header.nspectra

    dynamic_spectra = yr_obj.get_data(start, nspec)

    test_values = stat_test(dynamic_spectra)

    fitter(test_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=".py",
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
        help="test to flag on, [cheb_fitter, poly_fitter(default), bspline_fitter]",
        type=str,
        default="90-10",
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
        "--fitter",
        help="fitter to you, [cheb_fitter, poly_fitter(default), bspline_fitter]",
        type=str,
        default="bspline_fitter",
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
        "-start",
        "--start",
        help="Spectra to start.",
        type=int,
        default=0,
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

    channel_masker(
        args.file,
        args.sigma,
        args.start,
        args.nspec,
        args.fitter,
        args.out_file,
    )
