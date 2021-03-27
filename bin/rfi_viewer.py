#!/usr/bin/env python3
import argparse
import logging
import os
import textwrap
from tkinter import *
from tkinter import filedialog

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from your import Your
from your.utils.astro import dedisperse, calc_dispersion_delays
from your.utils.misc import YourArgparseFormatter


# based on https://steemit.com/utopian-io/@hadif66/tutorial-embeding-scipy-matplotlib-with-tkinter-to-work-on-images-in-a-gui-framework


class Paint(Frame):
    """
    Class for plotting object
    """

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None, dm=0):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # Creation of init_window
        # set widget title
        self.master.title("your_viewer")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH)  # , expand=1)
        self.create_widgets()
        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        self.dm = dm
        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        # added "file" to our menu
        menu.add_cascade(label="File", menu=file)

    def create_widgets(self):
        """
        Create all the user buttons
        """
        # which file to load
        self.browse = Button(self)
        self.browse["text"] = "Browse file"
        self.browse["command"] = self.load_file
        self.browse.grid(row=0, column=0)

        # move image foward to next gulp of data
        self.next = Button(self)
        self.next["text"] = "Next Gulp"
        self.next["command"] = self.next_gulp
        self.next.grid(row=0, column=1)

        # move image back to previous gulp of data
        self.prev = Button(self)
        self.prev["text"] = "Prevous Gulp"
        self.prev["command"] = self.prev_gulp
        self.prev.grid(row=0, column=3)

        # save figure
        self.prev = Button(self)
        self.prev["text"] = "Save Fig"
        self.prev["command"] = self.save_figure
        self.prev.grid(row=0, column=4)

        # Stat test to use
        self.tests = ["D'Angostino", "IQR", "Kurtosis", "MAD", "Skew", "Stand. Dev."]
        self.which_test = StringVar(self)
        self.test = OptionMenu(self, self.which_test, *self.tests)
        self.which_test.set("IQR")
        self.test.grid(row=0, column=5)

        self.which_test.trace("w", self.update_plot)

    def table_print(self, dic):
        """
        Prints out data using rich.Table

        Inputs:
        dic --  dictionary containing data file meta data to be printed
        """

        console = Console()
        table = Table(show_header=True, header_style="bold red", box=box.DOUBLE_EDGE)
        table.add_column("Parameter", justify="right")
        table.add_column("Value")
        for key, item in dic.items():
            table.add_row(key, f"{item}")
        console.print(table)

    def get_header(self):
        """
        Gets meta data from data file and give the data to nice_print() to print to user
        """
        dic = vars(self.your_obj.your_header)
        dic["tsamp"] = self.your_obj.your_header.tsamp
        dic["nchans"] = self.your_obj.your_header.nchans
        dic["foff"] = self.your_obj.your_header.foff
        dic["nspectra"] = self.your_obj.your_header.nspectra
        self.table_print(dic)

    def load_file(self, file_name=[""], start_samp=0, gulp_size=1024, chan_std=False):
        """
        Loads data from a file:

        Inputs:
        file_name -- name or list of files to load, if none given user must use gui to give file
        start_samp -- sample number where to start show the file, defaults to the beginning of the file
        gulp_size -- amount of data to show at a given time
        """
        self.start_samp = start_samp
        self.gulp_size = gulp_size
        self.chan_std = chan_std

        if len(file_name) == 0:
            file_name = filedialog.askopenfilename(
                filetypes=(("fits/fil files", "*.fil *.fits"), ("All files", "*.*"))
            )

        logging.info(f"Reading file {file_name}.")
        self.your_obj = Your(file_name)
        self.master.title(self.your_obj.your_header.basename)
        logging.info(f"Printing Header parameters")
        self.get_header()
        if self.dm != 0:
            self.dispersion_delays = calc_dispersion_delays(
                self.dm, self.your_obj.chan_freqs
            )
            max_delay = np.max(np.abs(self.dispersion_delays))
            if max_delay > self.gulp_size * self.your_obj.your_header.native_tsamp:
                logging.warning(
                    f"Maximum dispersion delay for DM ({self.dm}) = {max_delay:.2f}s is greater than "
                    f"the input gulp size {self.gulp_size*self.your_obj.your_header.native_tsamp}s. Pulses may not be "
                    f"dedispersed completely."
                )
                logging.warning(
                    f"Use gulp size of {int(max_delay//self.your_obj.your_header.native_tsamp):0d} to "
                    f"dedisperse completely."
                )
        self.read_data()

        # create three plots, for ax1=time_series, ax2=dynamic spectra, ax4=bandpass
        self.gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[1, 4, 1],
            height_ratios=[1, 4, 1],
            wspace=0.02,
            hspace=0.03,
        )
        ax1 = plt.subplot(self.gs[0, 1])  # timeseries
        ax2 = plt.subplot(self.gs[1, 1])  #  dynamic spectra
        self.ax3 = plt.subplot(self.gs[0, 2])  # histogram
        self.ax3.xaxis.tick_top()
        self.ax3.yaxis.tick_right()
        ax4 = plt.subplot(self.gs[1, 2])  # bandpass
        self.ax5 = plt.subplot(
            self.gs[1, 0]
        )  # verticle test.  Needs to be self. so I can self.ax6.legend()
        self.ax6 = plt.subplot(self.gs[2, 1])
        ax2.axis("off")
        ax1.set_xticks([])
        ax4.set_yticks([])

        # get the min and max image values to that we can see the typical values well
        self.vmax = min(np.max(self.data), np.median(self.data) + 5 * np.std(self.data))
        self.vmin = max(np.min(self.data), np.median(self.data) - 5 * np.std(self.data))
        self.im_ft = ax2.imshow(
            self.data, aspect="auto", vmin=self.vmin, vmax=self.vmax
        )

        # make bandpass
        bp_std = np.std(self.data, axis=1)
        bp_y = np.linspace(self.your_obj.your_header.nchans, 0, len(self.bandpass))
        (self.im_bandpass,) = ax4.plot(self.bandpass, bp_y, label="Bandpass")
        if self.chan_std:
            self.im_bp_fill = ax4.fill_betweenx(
                x1=self.bandpass - bp_std,
                x2=self.bandpass + bp_std,
                y=bp_y,
                interpolate=False,
                alpha=0.25,
                color="r",
                label="1 STD",
            )
            ax4.legend()
        ax4.set_ylim([-1, len(self.bandpass) + 1])
        ax4.set_xlabel("Avg. Arb. Flux")

        # make time series
        ax4.set_xlabel("<Arb. Flux>")
        (self.im_time,) = ax1.plot(self.time_series, label="Timeseries")
        ax1.set_xlim(-1, len(self.time_series + 1))
        ax1.set_ylabel("<Arb. Flux>")

        # plt.colorbar(self.im_ft, orientation="vertical", pad=0.01, aspect=30)

        # ax = self.im_ft.axes
        self.ax6.set_xlabel("Time [sec]")
        self.ax5.set_ylabel("Frequency [MHz]")
        self.ax5.set_yticks(np.linspace(0, self.your_obj.your_header.nchans, 8))
        yticks = [
            str(int(j))
            for j in np.flip(
                np.linspace(
                    self.your_obj.chan_freqs[0], self.your_obj.chan_freqs[-1], 8
                )
            )
        ]
        self.ax5.set_yticklabels(yticks)
        self.set_x_axis()

        # Make histogram
        self.ax3.hist(self.data.ravel(), bins=52, density=True)

        # show stat tests
        self.stat_test()
        (self.im_test_ver,) = self.ax5.plot(
            self.ver_test, bp_y, label=f"{self.which_test.get()}"
        )
        self.ax5.set_ylim([-1, len(self.bandpass) + 1])
        self.ax5.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        (self.im_test_hor,) = self.ax6.plot(
            self.hor_test, label=f"{self.which_test.get()}"
        )
        self.ax6.set_xlim([-1, len(self.time_series) + 1])
        self.ax6.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.im_ft.figure, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def client_exit(self):
        """
        exits the plotter
        """
        exit()

    def next_gulp(self):
        """
        Moves the images to the next gulp of data
        """
        self.start_samp += self.gulp_size
        # check if there is a enough data to fill plt
        proposed_end = self.start_samp + self.gulp_size
        if proposed_end > self.your_obj.your_header.nspectra:
            self.start_samp = self.start_samp - (
                proposed_end - self.your_obj.your_header.nspectra
            )
            logging.info("End of file.")
        self.update_plot()

    def prev_gulp(self):
        """
        Movies the images to the prevous gulp of data
        """
        # check if new start samp is in the file
        if (self.start_samp - self.gulp_size) >= 0:
            self.start_samp -= self.gulp_size
        self.update_plot()

    def update_plot(
        self, *args
    ):  # added *args to make self.which_test.trace("w", self.update_plot) happy
        self.read_data()
        self.set_x_axis()
        self.im_ft.set_data(self.data)
        self.im_bandpass.set_xdata(self.bandpass)
        self.ax3.cla()  # https://stackoverflow.com/questions/53258160/update-an-embedded-matplotlib-plot-in-a-pyqt5-gui-with-toolbar
        self.ax3.hist(self.data.ravel(), bins=52, density=True)
        if self.chan_std:
            self.fill_bp()
        self.im_bandpass.axes.set_xlim(
            np.min(self.bandpass) * 0.97, np.max(self.bandpass) * 1.03
        )
        self.im_time.set_ydata(np.mean(self.data, axis=0))
        self.im_time.axes.set_ylim(
            np.min(self.time_series) * 0.97, np.max(self.time_series) * 1.03
        )

        self.stat_test()
        self.im_test_ver.set_xdata(self.ver_test)
        self.im_test_ver.axes.set_xlim(
            np.min(self.ver_test)
            - 0.03 * (np.max(self.ver_test) - np.min(self.ver_test)),
            np.max(self.ver_test) * 1.03,
        )
        # print(self.im_test_ver.label)
        self.im_test_ver.set_label(f"{self.which_test.get()}")
        self.ax5.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.im_test_hor.set_ydata(self.hor_test)
        self.im_test_hor.axes.set_ylim(
            np.min(self.hor_test)
            - 0.03 * (np.max(self.hor_test) - np.min(self.hor_test)),
            np.max(self.hor_test) * 1.03,
        )
        self.im_test_hor.set_label(f"{self.which_test.get()}")
        self.ax6.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.canvas.draw()

    def fill_bp(self):
        self.im_bp_fill.remove()
        bp_std = np.std(self.data, axis=1)
        bp_y = self.im_bandpass.get_ydata()
        self.im_bp_fill = self.im_bandpass.axes.fill_betweenx(
            x1=self.bandpass - bp_std,
            x2=self.bandpass + bp_std,
            y=bp_y,
            interpolate=False,
            alpha=0.25,
            color="r",
        )

    def read_data(self):
        """
        Read data from the psr seach data file
        Returns:
        data -- a 2D array of frequency time plts
        """
        ts = self.start_samp * self.your_obj.your_header.tsamp
        te = (self.start_samp + self.gulp_size) * self.your_obj.your_header.tsamp
        self.data = self.your_obj.get_data(self.start_samp, self.gulp_size).T
        if self.dm != 0:
            logging.info(f"Dedispersing data at DM: {self.dm}")
            self.data = dedisperse(
                self.data.copy(),
                self.dm,
                self.your_obj.native_tsamp,
                delays=self.dispersion_delays,
            )
        self.bandpass = np.mean(self.data, axis=1)
        self.time_series = np.mean(self.data, axis=0)
        logging.info(
            f"Displaying {self.gulp_size} samples from sample {self.start_samp} i.e {ts:.2f}-{te:.2f}s - gulp mean: "
            f"{np.mean(self.data):.3f}, std: {np.std(self.data):.3f}"
        )

    def set_x_axis(self):
        """
        sets x axis labels in the correct location
        """
        ax = self.im_ft.axes
        xticks = ax.get_xticks()
        logging.debug(f"x-axis ticks are {xticks}")
        xtick_labels = (xticks + self.start_samp) * self.your_obj.your_header.tsamp
        logging.debug(f"Setting x-axis tick labels to {xtick_labels}")
        ax.set_xticklabels([f"{j:.2f}" for j in xtick_labels])

    def save_figure(self):
        """
        Saves the canvas image
        """
        img_name = (
            os.path.splitext(os.path.basename(self.your_obj.your_header.filename))[0]
            + f"_samp_{self.start_samp}_{self.start_samp + self.gulp_size}.png"
        )
        logging.info(f"Saving figure: {img_name}")
        self.im_ft.figure.savefig(img_name, dpi=300)
        logging.info(f"Saved figure: {img_name}")

    def stat_test(self):
        """
        Runs the statistical tests
        ["D'Angostino", "IQR", "Kurtosis", "MAD", "Skew", "Stand. Dev."]
        """
        if self.which_test.get() == "D'Angostino":
            self.ver_test, self.ver_test_p = stats.normaltest(self.data, axis=1)
            self.hor_test, self.hor_test_p = stats.normaltest(self.data, axis=0)
            # TODO plot p values
        elif self.which_test.get() == "IQR":
            self.ver_test = stats.iqr(self.data, axis=1)
            self.hor_test = stats.iqr(self.data, axis=0)
        elif self.which_test.get() == "Kurtosis":
            self.ver_test = stats.kurtosis(self.data, axis=1)
            self.hor_test = stats.kurtosis(self.data, axis=0)
        elif self.which_test.get() == "MAD":
            self.ver_test = stats.median_abs_deviation(self.data, axis=1)
            self.hor_test = stats.median_abs_deviation(self.data, axis=0)
        elif self.which_test.get() == "Skew":
            self.ver_test = stats.skew(self.data, axis=1)
            self.hor_test = stats.skew(self.data, axis=0)
        elif self.which_test.get() == "Stand. Dev.":
            self.ver_test = np.std(self.data, axis=1)
            self.hor_test = np.std(self.data, axis=0)


if __name__ == "__main__":
    logger = logging.getLogger()
    logging_format = (
        "%(asctime)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        prog="your_viewer.py",
        description="Read psrfits/filterbank files and show the data",
        formatter_class=YourArgparseFormatter,
        epilog=textwrap.dedent(
            """\
            This script can be used to visualize the data (Frequency-Time, bandpass and time series). It also reports some basic statistics of the data. 
            """
        ),
    )
    parser.add_argument(
        "-f",
        "--files",
        help="Fits or filterbank files to view.",
        required=False,
        default=[""],
        nargs="+",
    )
    parser.add_argument(
        "-s", "--start", help="Start index", type=float, required=False, default=0
    )
    parser.add_argument(
        "-g", "--gulp", help="Gulp size", type=int, required=False, default=4096
    )
    parser.add_argument(
        "-e",
        "--chan_std",
        help="Show 1 standard devation per channel in bandpass",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--display",
        help="Display size for the plot",
        type=int,
        nargs=2,
        required=False,
        metavar=("width", "height"),
        default=[1920, 1080],
    )
    parser.add_argument(
        "-dm",
        "--dm",
        help="DM to dedisperse the data",
        type=float,
        required=False,
        default=0,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    values = parser.parse_args()

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

    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.INFO)

    # root window created.
    root = Tk()
    root.geometry(f"{values.display[0]}x{values.display[1]}")
    # creation of an instance
    app = Paint(root, dm=values.dm)
    app.load_file(
        values.files, values.start, values.gulp, values.chan_std
    )  # load file with user params
    root.mainloop()
