#!/usr/bin/env python3
"""
Takes from Dynamic Spectra from filterbank/fits files
and displays in a GUI.
Shows time series above spectra and bandpass to the right.
Shows user defined statistical test
    on the left: per channel
    on the bottom: per time sample

Arguments:
    --files: files to get data

    --start: first sample to show

    --gulp: How many saples to show

    --chan_std: show 1 std for each channel

    --display: display size of GUI

    --dm: dispersion measure to dedisperse

    -subtract/--bandpass_subtract: subtract a polynomial bandpass fit

Key Binds:
    Left Arrow: Move the previous gulp

    Right Arrow: Move the the next gulp
"""
import argparse
import logging
import os
from builtins import range
from tkinter import (
    BOTH,
    TOP,
    Button,
    Frame,
    Menu,
    OptionMenu,
    StringVar,
    Tk,
    filedialog,
)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy import stats
from your import Your
from your.utils.astro import calc_dispersion_delays, dedisperse
from your.utils.misc import YourArgparseFormatter

from jess.calculators import preprocess, shannon_entropy
from jess.fitters import bspline_fitter

# from urllib.parse import non_hierarchical


# based on
# https://steemit.com/utopian-io/@hadif66/tutorial-embeding-scipy-matplotlib-with-tkinter-to-work-on-images-in-a-gui-framework


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

        # Bind left and right keys to move data chunk
        self.master.bind("<Left>", lambda event: self.prev_gulp())
        self.master.bind("<Right>", lambda event: self.next_gulp())

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

        self.ver_test_p = None
        self.hor_test_p = None

    def create_widgets(self):
        """
        Create all the user buttons
        """
        # which file to load
        self.browse = Button(self)
        self.browse["text"] = "Browse file"
        self.browse["command"] = self.load_file
        self.browse.grid(row=0, column=0)

        # save figure
        self.prev = Button(self)
        self.prev["text"] = "Save Fig"
        self.prev["command"] = self.save_figure
        self.prev.grid(row=0, column=1)

        # move image back to previous gulp of data
        self.prev = Button(self)
        self.prev["text"] = "Previous Gulp"
        self.prev["command"] = self.prev_gulp
        self.prev.grid(row=0, column=2)

        # move image forward to next gulp of data
        self.next = Button(self)
        self.next["text"] = "Next Gulp"
        self.next["command"] = self.next_gulp
        self.next.grid(row=0, column=3)

        # Stat test to use
        self.tests = [
            "98-2",
            "91-9",
            "90-10: Interdecile",
            "75-25: IQR",
            "Anderson-Darling",
            "D'Angostino",
            "Jarque-Bera",
            "Kurtosis",
            "Lilliefors",
            "MAD",
            "Midhing",
            "Shannon Entropy",
            "Shapiro Wilk",
            "Skew",
            "Stand. Dev.",
            "Trimean",
        ]
        self.which_test = StringVar(self)
        self.test = OptionMenu(self, self.which_test, *self.tests)
        self.which_test.set("D'Angostino")  # "75-25: IQR")
        self.test.grid(row=0, column=4)

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
        Gets meta data from data file and give the data to nice_print()
        to print to user
        """
        dic = vars(self.your_obj.your_header)
        dic["tsamp"] = self.your_obj.your_header.tsamp
        dic["nchans"] = self.your_obj.your_header.nchans
        dic["foff"] = self.your_obj.your_header.foff
        dic["nspectra"] = self.your_obj.your_header.nspectra
        self.table_print(dic)

    def load_file(
        self,
        file_name=None,
        start_samp=0,
        gulp_size=4096,
        chan_std=False,
        mask_file=None,
        bandpass_subtract=False,
    ):
        """
        Loads data from a file:

        Inputs:
        file_name -- name or list of files to load,
                    if none given user must use gui to give file
        start_samp -- sample number where to start show the file,
                      defaults to the beginning of the file
        gulp_size -- amount of data to show at a given time

        bandpass_subtract -- subtract a polynomial fit of the bandpass
        """
        self.start_samp = start_samp
        self.gulp_size = gulp_size
        self.chan_std = chan_std

        if file_name is None:
            file_name = filedialog.askopenfilename(
                filetypes=(("fits/fil files", "*.fil *.fits"), ("All files", "*.*"))
            )

        logging.info(f"Reading file {file_name}.")
        self.your_obj = Your(file_name)
        self.master.title(self.your_obj.your_header.basename)
        if bandpass_subtract:
            iinfo = np.iinfo(self.your_obj.your_header.dtype)
            self.min = iinfo.min
            self.max = iinfo.max
            self.subtract = True
        else:
            self.subtract = False

        if mask_file is not None:
            bad_chans = np.loadtxt(mask_file, dtype=int)
            self.mask = np.zeros(self.your_obj.your_header.nchans, dtype=bool)
            self.mask[bad_chans] = True
            logging.debug("Masking %i", self.mask)
        else:
            self.mask = np.zeros(self.your_obj.your_header.nchans, dtype=bool)

        logging.info("Printing Header parameters")
        self.get_header()
        if self.dm != 0:
            self.dispersion_delays = calc_dispersion_delays(
                self.dm, self.your_obj.chan_freqs
            )
            max_delay = np.max(np.abs(self.dispersion_delays))
            if max_delay > self.gulp_size * self.your_obj.your_header.native_tsamp:
                logging.warning(
                    f"Maximum dispersion delay for DM ({self.dm}) ="
                    f" {max_delay:.2f}s is greater than the input gulp size "
                    f"{self.gulp_size*self.your_obj.your_header.native_tsamp}"
                    f"s. Pulses may not be dedispersed completely."
                )
                logging.warning(
                    f"Use gulp size of "
                    f"{int(max_delay//self.your_obj.your_header.native_tsamp):0d}"
                    f" to dedisperse completely."
                )
        self.read_data()

        # create 6 plots, for ax1=time_series, ax2=dynamic spectra,
        # ax3= histogram, ax4=bandpass, ax5 = vertical test,
        # ax6 = horizontal test
        gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[1, 4, 1],
            height_ratios=[1, 4, 1],
            wspace=0.02,
            hspace=0.03,
        )
        ax1 = plt.subplot(gs[0, 1])  # timeseries
        ax2 = plt.subplot(gs[1, 1])  # dynamic spectra
        self.ax3 = plt.subplot(gs[0, 2])  # histogram
        self.ax3.xaxis.tick_top()
        self.ax3.yaxis.tick_right()
        ax4 = plt.subplot(gs[1, 2])  # bandpass
        self.ax5 = plt.subplot(
            gs[1, 0]
        )  # vertical test. Needs to be self. so I can self.ax6.legend()
        self.ax6 = plt.subplot(gs[2, 1])
        ax2.axis("off")
        ax1.set_xticks([])
        ax4.set_yticks([])

        # get the min and max image values so that
        # we can see the typical values well
        median = np.ma.median(self.data)
        std = np.ma.std(self.data)
        self.vmax = min(np.ma.max(self.data), median + 4 * std)
        self.vmin = max(np.ma.min(self.data), median - 4 * std)
        self.im_ft = ax2.imshow(
            self.data,
            aspect="auto",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="none",
        )

        # make bandpass
        bp_std = np.std(self.data, axis=1)
        bp_y = np.linspace(self.your_obj.your_header.nchans, 0, len(self.bandpass))
        # ax4.set_ylabel("Bandpass")
        # ax4.yaxis.set_label_position("right")
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
        else:
            pass
            ax4.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        ax4.set_ylim([-1, len(self.bandpass) + 1])
        ax4.set_xlabel("Avg. Arb. Flux")
        # ax4.set_title("Bandpass", rotation='vertical', x=1.2, y=.25)

        # make time series
        ax4.set_xlabel("<Arb. Flux>")
        (self.im_time,) = ax1.plot(self.time_series, label="Timeseries")
        ax1.set_xlim(-1, len(self.time_series + 1))
        ax1.set_ylabel("<Arb. Flux>")
        # ax1.set_label("Time Series")
        ax1.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        # ax1.set_title("Time Series", y=1.0, pad=-14)
        # ax1.legend(handletextpad=0, handlelength=0, framealpha=0.4)

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

        # Make histogram
        self.ax3.hist(self.data.compressed(), bins=52, density=True, label="Hist")
        self.ax3.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        # show stat tests
        which_test = self.stat_test()
        if True:  # self.hor_test_p is None:
            (self.im_test_ver,) = self.ax5.plot(self.ver_test, bp_y, label=which_test)
            (self.im_test_hor,) = self.ax6.plot(self.hor_test, label=which_test)
            self.ax5.legend(handletextpad=0, handlelength=0, framealpha=0.4)
            self.ax6.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        else:
            (self.im_test_ver,) = self.ax5.plot(self.ver_test, bp_y, label=which_test)
            _ = self.ax5.plot(self.ver_test_p, bp_y, label=self.test_label)
            # self.im_test_ver.set_legend(which_test, "p-value")
            (self.im_test_hor,) = self.ax6.plot(self.hor_test, label=which_test)
            _ = self.ax6.plot(self.hor_test_p, label=self.test_label)
            # self.im_test_hor.set_label(which_test, "p-value")
            self.ax5.legend(handletextpad=0, handlelength=0.3, framealpha=0.4)
            self.ax6.legend(handletextpad=0, handlelength=0.3, framealpha=0.4)
        self.ax5.set_ylim([-1, len(self.bandpass) + 1])

        self.ax6.set_xlim([-1, len(self.time_series) + 1])

        self.set_x_axis()

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
        Movies the images to the previous gulp of data
        """
        # check if new start samp is in the file
        if (self.start_samp - self.gulp_size) >= 0:
            self.start_samp -= self.gulp_size
        self.update_plot()

    def update_plot(self, *args):
        """
        Redraws the plots when something is changed
        """
        # added *args to make self.which_test.trace("w", self.update_plot)
        #  happy
        self.read_data()
        self.set_x_axis()
        self.im_ft.set_data(self.data)
        self.im_bandpass.set_xdata(self.bandpass)
        self.ax3.cla()
        # https://stackoverflow.com/questions/53258160/update-an-embedded-matplotlib-plot-in-a-pyqt5-gui-with-toolbar
        self.ax3.hist(self.data.ravel(), bins=52, density=True)
        if self.chan_std:
            self.fill_bp()
        self.im_bandpass.axes.relim()
        self.im_bandpass.axes.autoscale(axis="x")
        self.im_time.set_ydata(np.ma.mean(self.data, axis=0))
        self.im_time.axes.relim()
        self.im_time.axes.autoscale(axis="y")

        which_test = self.stat_test()

        self.im_test_ver.set_xdata(self.ver_test)
        self.im_test_ver.axes.relim()
        self.im_test_ver.axes.autoscale(axis="x")

        self.im_test_ver.set_label(f"{which_test}")
        self.ax5.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.im_test_hor.set_ydata(self.hor_test)
        self.im_test_hor.axes.relim()
        self.im_test_hor.axes.autoscale(axis="y")
        self.im_test_hor.set_label(f"{which_test}")
        self.ax6.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.canvas.draw()

    def fill_bp(self):
        """
        Adds each channel's standard deviations to bandpass plot
        """
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
        Read data from the psr search data file
        Returns:
        data -- a 2D array of frequency time plts
        """
        ts = self.start_samp * self.your_obj.your_header.tsamp
        te = (self.start_samp + self.gulp_size) * self.your_obj.your_header.tsamp
        data_chunk = self.your_obj.get_data(self.start_samp, self.gulp_size).T
        if self.dm != 0:
            logging.info(f"Dedispersing data at DM: {self.dm}")
            data_chunk = dedisperse(
                data_chunk.copy(),
                self.dm,
                self.your_obj.native_tsamp,
                delays=self.dispersion_delays,
            )
        if self.mask is None:
            mask = np.broadcast_to(False, data_chunk.shape)
            self.data = np.ma.array(data_chunk, mask=mask)
        else:
            mask = np.broadcast_to(self.mask[:, None], data_chunk.shape)
            self.data = np.ma.array(data_chunk, mask=mask)
        if self.subtract:
            bandpass = bspline_fitter(np.median(self.data, axis=1))
            # fit data to median bandpass
            np.clip(bandpass, self.min, self.max, out=bandpass)
            # make sure the fit is numerically possable
            self.data = self.data - bandpass[:, None]

            # attempt to return the correct data type,
            # most values are close to zero
            # add get clipped, causeing dynamic range problems
            # diff = np.clip(self.data - bandpass[:, None], self.min, self.max)
            # self.data = diff #diff.astype(self.your_obj.your_header.dtype)

        self.bandpass = np.ma.mean(self.data, axis=1)
        self.time_series = np.ma.mean(self.data, axis=0)
        logging.info(
            f"Displaying {self.gulp_size} samples from sample "
            f"{self.start_samp} i.e {ts:.2f}-{te:.2f}s - gulp mean: "
            f"{np.ma.mean(self.data):.3f}, std: {np.ma.std(self.data):.3f}"
        )

    def set_x_axis(self):
        """
        sets x axis labels in the correct location
        """
        ax = self.im_test_hor.axes
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
        which_test = self.which_test.get()
        # mask = np.broadcast_to(self.mask[:,None],  self.data.shape)
        # self.data = self.data[mask]
        if which_test == "98-2":
            top_quant, bottom_quant = np.quantile(self.data, [0.98, 0.02], axis=1)
            self.ver_test = (top_quant - bottom_quant) / 2.0
            top_quant, bottom_quant = np.quantile(
                self.data[~self.mask], [0.98, 0.02], axis=0
            )
            self.hor_test = (top_quant - bottom_quant) / 2.0
        elif which_test == "91-9":
            top_quant, bottom_quant = np.quantile(self.data, [0.91, 0.09], axis=1)
            self.ver_test = (top_quant - bottom_quant) / 2.0
            top_quant, bottom_quant = np.quantile(
                self.data[~self.mask], [0.91, 0.09], axis=0
            )
            self.hor_test = (top_quant - bottom_quant) / 2.0
        elif which_test == "90-10: Interdecile":
            top_quant, bottom_quant = np.quantile(self.data, [0.90, 0.10], axis=1)
            self.ver_test = (top_quant - bottom_quant) / 2.0
            top_quant, bottom_quant = np.quantile(
                self.data[~self.mask], [0.90, 0.10], axis=0
            )
            self.hor_test = (top_quant - bottom_quant) / 2.0
        elif which_test == "75-25: IQR":
            self.ver_test = stats.iqr(self.data, axis=1)
            self.hor_test = stats.iqr(self.data[~self.mask], axis=0)
        elif which_test == "Anderson-Darling":
            num_freq, num_samps = self.data.shape
            self.ver_test = np.zeros(num_freq)
            # self.ver_test_p = np.zeros(num_freq)
            self.hor_test = np.zeros(num_samps)
            # self.hor_test_p = np.zeros(num_samps)
            for j in range(0, num_freq):
                self.ver_test[j], _, _ = stats.anderson(self.data[j, :], dist="norm")
            for k in range(0, num_samps):
                self.hor_test[k], _, _ = stats.anderson(
                    self.data[~self.mask][:, k], dist="norm"
                )
        elif which_test == "D'Angostino":
            self.ver_test, self.ver_test_p = stats.normaltest(self.data, axis=1)
            self.hor_test, self.hor_test_p = stats.normaltest(
                self.data[~self.mask], axis=0
            )
            # self.test_label = "p-value"
        elif which_test == "Jarque-Bera":
            num_freq, num_samps = self.data.shape
            if num_freq < 2000:
                logging.warning(
                    "Jarque-Bera requires > 2000 points, given %i channels", num_freq
                )
            if num_samps < 2000:
                logging.warning(
                    "Jarque-Bera requires > 2000 points, given %i samples", num_samps
                )
            self.ver_test = np.zeros(num_freq)
            self.ver_test_p = np.zeros(num_freq)
            self.hor_test = np.zeros(num_samps)
            self.hor_test_p = np.zeros(num_samps)
            for j in range(0, num_freq):
                self.ver_test[j], self.ver_test_p[j] = stats.jarque_bera(
                    self.data[j, :],
                )
            for k in range(0, num_samps):
                self.hor_test[k], self.hor_test_p[k] = stats.jarque_bera(
                    self.data[~self.mask][:, k]
                )
        elif which_test == "Kurtosis":
            self.ver_test = stats.kurtosis(self.data, axis=1)
            self.hor_test = stats.kurtosis(self.data[~self.mask], axis=0)
        elif which_test == "Lilliefors":
            # I don't take into account the change of dof when calculating the p_value
            # The test stattic is the same as statsmodels lilliefors
            num_freq, num_samps = self.data.shape
            self.ver_test = np.zeros(num_freq)
            self.ver_test_p = np.zeros(num_freq)
            self.hor_test = np.zeros(num_samps)
            self.hor_test_p = np.zeros(num_samps)
            data_0, data_1 = preprocess(self.data)
            for j in range(0, num_freq):
                self.ver_test[j], self.ver_test_p[j] = stats.kstest(
                    data_0[j, :], "norm"
                )
            for k in range(0, num_samps):
                print(k)
                self.hor_test[k], self.hor_test_p[k] = stats.kstest(
                    data_1[~self.mask][:, k], "norm"
                )
        elif which_test == "MAD":
            self.ver_test = stats.median_abs_deviation(self.data, axis=1)
            self.hor_test = stats.median_abs_deviation(self.data[~self.mask], axis=0)
        elif which_test == "Midhing":
            top_quant, bottom_quant = np.quantile(self.data, [0.75, 0.25], axis=1)
            self.ver_test = (top_quant + bottom_quant) / 2.0
            top_quant, bottom_quant = np.quantile(
                self.data[~self.mask], [0.75, 0.25], axis=0
            )
            self.hor_test = (bottom_quant + top_quant) / 2.0
        elif which_test == "Shannon Entropy":
            self.ver_test = shannon_entropy(self.data, axis=1)
            self.hor_test = shannon_entropy(self.data[~self.mask], axis=0)
        elif which_test == "Shapiro Wilk":
            num_freq, num_samps = self.data.shape
            self.ver_test = np.zeros(num_freq)
            self.ver_test_p = np.zeros(num_freq)
            self.hor_test = np.zeros(num_samps)
            self.hor_test_p = np.zeros(num_samps)
            data_0, data_1 = preprocess(self.data)
            for ichan in range(0, num_freq):
                self.ver_test[ichan], self.ver_test_p[ichan] = stats.shapiro(
                    data_0[ichan, :],
                )
            for isamp in range(0, num_samps):
                self.hor_test[isamp], self.hor_test_p[isamp] = stats.shapiro(
                    data_1[~self.mask][:, isamp]
                )
        elif which_test == "Skew":
            self.ver_test = stats.skew(self.data, axis=1)
            self.hor_test = stats.skew(self.data[~self.mask], axis=0)
        elif which_test == "Stand. Dev.":
            self.ver_test = np.std(self.data, axis=1)
            self.hor_test = np.std(self.data[~self.mask], axis=0)
            # self.ver_test_p = None
            # self.hor_test_p = None
        elif which_test == "Trimean":
            top_quant, middle_quant, bottom_quant = np.quantile(
                self.data, [0.75, 0.50, 0.25], axis=1
            )
            self.ver_test = (bottom_quant + 2.0 * middle_quant + top_quant) / 4.0
            top_quant, middle_quant, bottom_quant = np.quantile(
                self.data[~self.mask], [0.75, 0.50, 0.25], axis=0
            )
            self.hor_test = (bottom_quant + 2.0 * middle_quant + top_quant) / 4.0
        else:
            raise ValueError(f"You gave {which_test}, which is not avaliable.")

        self.ver_test[self.mask] = np.nan  # mask values that are channel masked
        return which_test


if __name__ == "__main__":
    logger = logging.getLogger()
    logging_format = (
        "%(asctime)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        prog="your_viewer.py",
        description="Read psrfits/filterbank files and show the data",
        formatter_class=YourArgparseFormatter,
        epilog=__doc__,
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
        help="Show 1 standard deviation per channel in bandpass",
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
    parser.add_argument(
        "-m",
        "--mask",
        help="Channel Mask to apply",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-subtract",
        "--bandpass_subtract",
        help="subtract a polynomial fitted bandpass",
        required=False,
        default=False,
        action="store_true",
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
        values.files,
        values.start,
        values.gulp,
        values.chan_std,
        values.mask,
        values.bandpass_subtract,
    )  # load file with user params
    root.mainloop()
