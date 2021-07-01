#!/usr/bin/env python3
"""
Runs a MAD filter over dispersed data.
"""

import argparse
import logging
import os
import textwrap

# import psutil
from rich.logging import RichHandler
from rich.progress import track
from your import Your
from your.formats.filwriter import make_sigproc_object

# from your.utils.math import primes
from your.utils.misc import YourArgparseFormatter

from jess.dispersion import dedisperse, delay_lost
from jess.fitters import get_fitter  # bspline_fitter, cheb_fitter, poly_fitter
from jess.JESS_filters import mad_spectra

# from your.utils.rfi import sk_sg_filter
# from your.writer import Writer


logger = logging.getLogger()

"""
class JessWriter(Writer):
    def __init__(
        self,
        your_object,
        dm,
        sigma,
        channels_per_subband,
        remove_ends,
        nstart=0,
        nsamp=None,
        c_min=None,
        c_max=None,
        npoln=1,
        outdir=None,
        outname=None,
        flag_rfi=False,
        progress=True,
        spectral_kurtosis_sigma=4,
        savgol_frequency_window=15,
        savgol_sigma=4,
        gulp=None,
        zero_dm_subt=False,
        time_decimation_factor=1,
        frequency_decimation_factor=1,
        replacement_policy="mean",
    ):

        super().__init__(
            your_object,
            nstart=nstart,
            nsamp=nsamp,
            c_min=c_min,
            c_max=c_max,
            npoln=npoln,
            outdir=outdir,
            outname=outname,
            flag_rfi=flag_rfi,
            progress=progress,
            spectral_kurtosis_sigma=spectral_kurtosis_sigma,
            savgol_frequency_window=savgol_frequency_window,
            savgol_sigma=savgol_sigma,
            gulp=gulp,
            zero_dm_subt=zero_dm_subt,
            time_decimation_factor=time_decimation_factor,
            frequency_decimation_factor=frequency_decimation_factor,
            replacement_policy=replacement_policy,
        )

        self.dm = dm
        self.sigma = sigma
        self.channels_per_subband = channels_per_subband
        self.remove_ends = remove_ends

        # self.your_object = your_object
        self.samples_lost = delay_lost(
            self.dm, your_object.chan_freqs, your_object.tsamp
        )

    def clean_data(self: object, data: np.ndarray) -> np.ndarray:
        dedispersed = dedisperse(
            data,
            self.dm,
            self.your_object.tsamp,
            self.your_object.chan_freqs,
        )
        dedispersed[0 : -self.samples_lost, :] = spectral_mad(
            dedispersed[0 : -self.samples_lost, :],
            frame=self.channels_per_subband,
            sigma=self.sigma,
        )
        redispersed = dedisperse(
            dedispersed,
            -self.dm,
            self.your_object.tsamp,
            self.your_object.chan_freqs,
        )

        return redispersed

    def get_data_to_write(self: object, start_sample: int, nsamp: int):

        # Read data to self.data, selects channels
        # Optionally perform RFI filtering and zero-DM subtraction
        # Args:
        #    start_sample (int): Start sample number to read from
        #    nsamp (int): Number of samples to read

        proposed_end = start_sample + nsamp + 2 * self.samples_lost
        if proposed_end > self.your_object.your_header.nspectra:
            nsamp = self.your_object.your_header.nspectra
            # something I don't know
        data = self.your_object.get_data(
            start_sample, nsamp + 2 * self.samples_lost, npoln=self.npoln
        )

        data = self.your_object.get_data(
            start_sample, nsamp + 2 * self.samples_lost, npoln=self.npoln
        )

        if self.npoln == 1:
            data = np.expand_dims(data, 1)

        # shape of data is (nt, npoln, nf)
        data = data[:, :, self.chan_min : self.chan_max]
        if self.flag_rfi:
            for i in range(data.shape[1]):
                data_to_flag = data[:, i, :]
                mask = sk_sg_filter(
                    data_to_flag,
                    self.your_object,
                    self.sk_sig,
                    self.sg_fw,
                    self.sg_sig,
                )

                if self.replacement_policy == "mean":
                    fill_value = np.mean(data_to_flag[:, ~mask])
                elif self.replacement_policy == "median":
                    fill_value = np.median(data_to_flag[:, ~mask])
                else:
                    fill_value = 0

                if self.your_object.your_header.nbits < 32:
                    fill_value = np.around(fill_value).astype(
                        self.your_object.your_header.dtype
                    )

                data[:, i, mask] = fill_value

        shape = data.shape

        for i in range(data.shape[1]):
            cleaned = self.clean_data(data[:, i, :])

            if start_sample == 0 and not self.remove_ends:
                data_clean = np.zeros([nsamp, shape[1], shape[2]], dtype=data.dtype)
                data_clean[:, i, :] = cleaned[: -2 * self.samples_lost]
            elif (
                start_sample + nsamp == self.your_object.your_header.nspectra
                and not self.remove_ends
            ):
                print("In final if")
                data_clean = np.zeros([nsamp, shape[1], shape[2]], dtype=data.dtype)
                data_clean[:, i, :] = cleaned[2 * self.samples_lost :]
            else:
                data_clean = np.zeros([nsamp, shape[1], shape[2]], dtype=data.dtype)
                data_clean[:, i, :] = cleaned[self.samples_lost : -self.samples_lost]
        data = data_clean
        print(data.shape)

        if self.zero_dm_subt:
            if self.npoln > 1:
                raise NotImplementedError(
                    "0-DM subtraction is implemented only for 1 output pol."
                )

            logger.debug("Subtracting 0-DM time series from the data.")
            min_value = np.iinfo(self.your_object.your_header.dtype).min
            max_value = np.iinfo(self.your_object.your_header.dtype).max
            nt, npoln, nf = data.shape
            ts = data.mean(-1)
            bp = data.mean(0).squeeze()

            for channel in range(nf):
                data[:, :, channel] = np.clip(
                    data[:, :, channel].astype("float32") - ts + bp[channel],
                    min_value,
                    max_value,
                )

        data = data.astype(self.your_object.your_header.dtype)

        # shape of data is (nt, npoln, nf)
        self.data = data
"""

"""
def get_gulp_size(your: object) -> int:

    # This function calculates the number of samples that
    # will fit the data and the mask while using 80% of
    # the avaliable ram.

    # Args:
    #     Your object: for the file you are intersted in loading

    # returns:
    #     number of samples

    available_ram = 0.80 * psutil.virtual_memory()[1]  # in bytes
    # take up 80% of the avaliable memory
    bytes_per_spectra = your.your_header.nchans * (your.your_header.nbits + 64) / 8
    # the masks are 64 bit

    return np.floor(available_ram / bytes_per_spectra).astype(int)
"""


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
        out_file = path + "_MAD.fil"
        logger.info("No outfile given, writing to %s", out_file)
        return out_file

    # if a file is given, make sure it has a .fil ext
    path, file_ext = os.path.splitext(out_file)
    if file_ext:
        assert file_ext == ".fil", "I can only write .fil, you gave: %s!" % file_ext
        return out_file

    out_file += ".fil"
    return out_file


def mad_cleaner(
    file: str,
    dispersion_measure: float,
    sigma: float = 3,
    gulp: int = 16384,
    fitter: str = "poly_fitter",
    channels_per_subband: int = 256,
    remove_ends: bool = False,
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

    """
    if out_file:
        path, file_ext = os.path.splitext(out_file[0])
    else:
        path, file_ext = os.path.splitext(file[0])
        out_file = path + "_mad_cleaned" + "." + file_ext


    original_yr_input = Your(file)
    wr = JessWriter(
        original_yr_input,ff
        dm=dispersion_measure,
        sigma=sigma,
        channels_per_subband=channels_per_subband,
        remove_ends=remove_ends,
    )

    if file_ext == ".fits":
        wr.to_fits(npsub=4032)
    elif file_ext == ".fil":
        wr.to_fil()
    else:
        raise ValueError(f"Tried file extention {file_ext}, which I can't write")
    """

    fitter = get_fitter(fitter)
    out_file = get_outfile(file, out_file)

    yr_input = Your(file)
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

    # add data that can't be dispersed
    # because its at the start
    if not remove_ends:
        sigproc_object.append_spectra(yr_input.get_data(0, samples_lost), out_file)

    # loop through all the data we can dedisperse
    for j in track(range(0, yr_input.your_header.nspectra, gulp)):

        if 2 * samples_lost + j + gulp < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, 2 * samples_lost + gulp)
        else:
            data = yr_input.get_data(j, yr_input.your_header.nspectra - j)
        dedisp = dedisperse(
            data, dispersion_measure, yr_input.your_header.tsamp, yr_input.chan_freqs
        )
        dedisp[0:-samples_lost, :] = mad_spectra(
            dedisp[0:-samples_lost, :], frame=channels_per_subband, sigma=sigma
        )
        redisip = dedisperse(
            dedisp, -dispersion_measure, yr_input.your_header.tsamp, yr_input.chan_freqs
        )
        redisip = redisip.astype(yr_input.your_header.dtype)
        sigproc_object.append_spectra(redisip[samples_lost:-samples_lost, :], out_file)

    # add data that can't be dispersed
    # because its at the end

    if not remove_ends:
        sigproc_object.append_spectra(
            yr_input.get_data(
                yr_input.your_header.nspectra - samples_lost, samples_lost
            ),
            out_file,
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
        help="Dispersion Measure to process the data",
        type=float,
        required=True,
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
        "--channels_per_subband",
        help="Number of channels in each subband",
        type=int,
        default=256,
        required=False,
    )
    parser.add_argument(
        "--fitter",
        help="fitter to you, [cheb_fitter, poly_fitter(default), bspline_fitter]",
        type=str,
        default="poly_fitter",
        required=False,
    )
    parser.add_argument(
        "--gulp",
        help="Number of samples to process at each loop",
        type=int,
        default=16384,
        required=False,
    )
    parser.add_argument(
        "--remove_ends",
        help="keep ends of file that cannot be cleaned",
        action="store_true",
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

    mad_cleaner(
        args.file,
        args.dispersion_measure,
        args.sigma,
        args.gulp,
        args.fitter,
        args.channels_per_subband,
        args.remove_ends,
        args.out_file,
    )
