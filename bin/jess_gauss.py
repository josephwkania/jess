#!/usr/bin/env python3
"""
Runs a Kurtosis and Skew filter over a .fits/.fil file
writing out a .fil
"""

import argparse
import logging
import os
import textwrap
from typing import Callable, Tuple, Union

from rich.logging import RichHandler
from rich.progress import track
from your import Writer, Your
from your.formats.filwriter import sigproc_object_from_writer
from your.utils.misc import YourArgparseFormatter

from jess.calculators import get_flatten_to

try:
    import cupy as xp

    BACKEND_GPU = True

except ModuleNotFoundError:

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
    modes_to_zero: int,
    test: Callable,
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

        modes_to_zero: number of Fourier modes to zero in the highpass

        test: Gaussianity test to use

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

    if modes_to_zero >= 1:
        bandpass = xp.full(
            yr_input.your_header.nchans, fill_value=flatten_to, dtype=xp.float32
        )

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

        _, mask, mask_percentage = test(
            dynamic_spectra=data,
            detrend=None,
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

        data[mask] = flatten_to

        if modes_to_zero < 0:
            time_series -= xp.median(time_series)
            data += time_series[:, None]
        elif modes_to_zero == 1:
            logging.debug("Zero DMing: Subtracting Mean")
            data, dm_percentage = zero_dm(data, bandpass, return_same_dtype=False)
        elif modes_to_zero > 1:
            logging.debug("High Pass filtering: removing %i modes", modes_to_zero)
            data, dm_percentage = zero_dm_fft(
                data, bandpass, modes_to_zero=modes_to_zero, return_same_dtype=False
            )
        else:
            dm_percentage = 0

        total_flag += dm_percentage
        logging.info(
            "Gauss Flag %.2f%%, Chan Flag %.2f%%, Highpass %.2f%%, Total %.2f%%",
            mask_percentage,
            chan_mask_percent,
            dm_percentage,
            mask_percentage_total,
        )

        data = to_dtype(data, dtype=yr_input.your_header.dtype)

        if BACKEND_GPU:
            data = data.get()
        sigproc_object.append_spectra(data, out_file)

    total_flag /= n_iter
    logging.info(
        "Total: Gauss Flag %.2f%%, Chan Flag %.2f%%,  Highpass %.2f%%, Total %.2f%%",
        total_flag[0],
        total_flag[1],
        dm_percentage,
        total_flag[2],
    )


def clean_fast(
    yr_input: Your,
    sigma: float,
    samples_per_block: int,
    test: Callable,
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

        _, mask, mask_percentage = test(
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
        chan_mask_percent = 100 * chan_mask.mean()
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
        help="Flatten data to this number. If `None`, sets the data 1/4"
        + " between zero and dtype max",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-mtz",
        "--modes_to_zero",
        help="""Number of Modes to zero; -1 perserve time seres; 0 subtracted mean;
        >1 number of Fourier Modes to remove""",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "-test",
        "--test",
        help="Test to use, [kurtosis_and_skew, jaqrue_bera, dagostino]",
        type=str,
        default="kurtosis_and_skew",
        required=False,
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
        "-gpu",
        "--gpu",
        help="Which GPU to use. Default is gpu 0, -1 for cpu",
        type=int,
        default=0,
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

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    from jess.JESS_filters_generic import dagostino, jarque_bera, kurtosis_and_skew
    from jess.scipy_cupy.stats import iqr_med

    if args.gpu < 0 or not BACKEND_GPU:
        import numpy as xp

        from jess.calculators import flattner_median, to_dtype
        from jess.JESS_filters import zero_dm, zero_dm_fft

        BACKEND_GPU = False
    else:
        from jess.calculators_cupy import flattner_median, to_dtype
        from jess.JESS_filters_cupy import zero_dm, zero_dm_fft

    if args.flatten_to is None:
        FLATTEN_TO = get_flatten_to(yrinput.your_header.nbits)
    else:
        FLATTEN_TO = args.flatten_to

    if args.winsorize_args[0] == -1:
        WINSORIZE = None
    else:
        WINSORIZE = args.winsorize_args

    test_str = args.test.casefold()
    if test_str == "kurtosis_and_skew":
        _test = kurtosis_and_skew
    elif test_str == "jarque_bera":
        _test = jarque_bera
    elif test_str == "dagostino":
        _test = dagostino
    else:
        raise NotImplementedError(f"Test {test_str} not available")

    if args.fast:
        clean_fast(
            yr_input=yrinput,
            samples_per_block=args.samples_per_block,
            test=_test,
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
            modes_to_zero=args.modes_to_zero,
            test=_test,
            winsorize_args=WINSORIZE,
            flatten_to=FLATTEN_TO,
            sigma=args.sigma,
            gulp=args.gulp,
            out_file=outfile,
            sigproc_object=sigproc_obj,
        )
