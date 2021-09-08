#!/usr/bin/env python3
"""  # noqa: W605
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

Arrangement

### \################\  \################\ ###
### \Time Series ax01\  \Time Series ax02\ ###
### \################\  \################\ ###

\##\ \###############\  \################\ \##\
\B \ \ Dynamic       \  \Dynamic         \ \B \
\a \ \ Spectra       \  \Spectra         \ \a \
\n \ \ ax11          \  \Masked          \ \n \
\d \ \               \  \ax12            \ \d \
\10\ \               \  \                \ \23\
\##\ \###############\  \################\ \##\

\##\ \###############\  \################\ \##\
\T \ \ Test blocks   \  \ Mask           \ \M \
\e \ \ ax21          \  \ ax22           \ \a \
\s \ \               \  \                \ \s \
\t \ \ Test          \  \                \ \k \
\  \ \               \  \                \ \23\
\20\ \###############\  \################\ \##\

### \################\  \################\ ###
### \Test ax31       \  \Mask % ax32     \ ###
### \################\  |################\ ###
"""
import argparse
import logging
import os
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
from your import Your
from your.utils.astro import calc_dispersion_delays, dedisperse
from your.utils.misc import YourArgparseFormatter

from jess.fitters import bspline_fitter
from jess.JESS_filters import (
    dagostino_time,
    iqr_time,
    kurtosis_time,
    mad_time,
    skew_time,
)

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
        self.master.title("mask_viewer")

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

        # move image foward to next gulp of data
        self.next = Button(self)
        self.next["text"] = "Next Gulp"
        self.next["command"] = self.next_gulp
        self.next.grid(row=0, column=3)

        # Stat test to use
        # self.tests = ["D'Angostino", "IQR", "Kurtosis", "MAD", "Skew", "Stand. Dev."]
        self.tests = [
            "Anderson-Darling",
            "D'Angostino",
            "Jarque-Bera",
            "KS",
            "Kurtosis",
            "Shapiro Wilk",
            "Skew",
        ]
        self.which_test = StringVar(self)
        self.test = OptionMenu(self, self.which_test, *self.tests)
        self.which_test.set("IQR")
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

    def make_canvas(self):
        """
        Makes the canvas, sets the axes, populates
        the plots with the inital images.
        """

        # create a 4x4 grid
        #  4 images in the center, surrounded by parameter plots
        # ax12=masl, ax10 = vertical test, ax21 = horizontal test
        gs = gridspec.GridSpec(
            4,
            4,
            width_ratios=[1, 4, 4, 1],
            height_ratios=[1, 4, 4, 1],
            wspace=0.02,
            hspace=0.03,
        )
        # plt.rcParams['axes.titley'] = 1.0
        # plt.rcParams['axes.titlepad'] = -14
        ax01 = plt.subplot(gs[0, 1])  # timeseries
        # ax01.set_title("Time Series", position=(0.5, 0.3))
        ax02 = plt.subplot(gs[0, 2])  # cleaned Time Series
        self.ax10 = plt.subplot(gs[1, 0])  # bandpass
        ax11 = plt.subplot(gs[1, 1])  # dynamic spectra
        ax12 = plt.subplot(gs[1, 2])  # cleaned dynamic spectra
        ax13 = plt.subplot(gs[1, 3])  # cleaned bandpass

        self.ax20 = plt.subplot(gs[2, 0])  # channel test values
        self.ax21 = plt.subplot(gs[2, 1])  # 2D test values
        ax22 = plt.subplot(gs[2, 2])  # mask
        ax23 = plt.subplot(gs[2, 3])  # channel mask percentage

        self.ax31 = plt.subplot(gs[3, 1])  # time test vales
        ax32 = plt.subplot(gs[3, 2])  # masked percentage

        # self.ax22.xaxis.tick_top()
        # self.ax22.yaxis.tick_right()

        ax01.set_xticks([])
        ax02.get_xaxis().set_visible(False)
        ax02.yaxis.tick_right()  # set up plot labels

        self.ax10.xaxis.tick_top()
        ax11.get_xaxis().set_visible(False)
        ax11.get_yaxis().set_visible(False)
        ax12.get_xaxis().set_visible(False)
        ax12.get_yaxis().set_visible(False)
        ax13.xaxis.tick_top()
        ax13.get_yaxis().set_visible(False)

        self.ax21.get_xaxis().set_visible(False)
        self.ax21.get_yaxis().set_visible(False)
        ax22.get_xaxis().set_visible(False)
        ax22.get_yaxis().set_visible(False)
        ax23.yaxis.tick_right()
        ax23.get_yaxis().set_visible(False)

        ax32.yaxis.tick_right()

        # get the min and max image values so that
        # we can see the typical values well
        median = np.median(self.data)
        std = np.std(self.data)
        self.vmax = min(np.max(self.data), median + 4 * std)
        self.vmin = max(np.min(self.data), median - 4 * std)
        self.im_ft = ax11.imshow(
            self.data, aspect="auto", vmin=self.vmin, vmax=self.vmax
        )

        # make bandpass
        bp_std = np.std(self.data, axis=1)
        bp_y = np.linspace(self.your_obj.your_header.nchans, 0, len(self.bandpass))
        (self.im_bandpass,) = self.ax10.plot(self.bandpass, bp_y, label="Bandpass")
        if self.chan_std:
            self.im_bp_fill = self.ax10.fill_betweenx(
                x1=self.bandpass - bp_std,
                x2=self.bandpass + bp_std,
                y=bp_y,
                interpolate=False,
                alpha=0.25,
                color="r",
                label="1 STD",
            )
            self.ax10.legend()
        else:
            pass
            # ax12.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        self.ax10.set_ylim([-1, len(self.bandpass) + 1])
        # self.ax10.set_xlabel("Avg. Arb. Flux")
        # ax12.set_title("Bandpass", rotation='vertical', x=1.2, y=.25)

        # make time series
        # ax12.set_xlabel("<Arb. Flux>")
        (self.im_time,) = ax01.plot(self.time_series, label="Timeseries")
        ax01.set_xlim(-1, len(self.time_series + 1))
        ax01.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        # ax01.set_ylabel("<Arb. Flux>")
        # ax01.set_title("Time Series", y=1.0, pad=-14)
        # ax01.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        # plt.colorbar(self.im_ft, orientation="vertical", pad=0.01, aspect=30)

        # ax = self.im_ft.axes
        self.ax10.set_ylabel("Frequency [MHz]")
        self.ax10.set_yticks(np.linspace(0, self.your_obj.your_header.nchans, 8))
        yticks_freq = [
            str(int(j))
            for j in np.flip(
                np.linspace(
                    self.your_obj.chan_freqs[0], self.your_obj.chan_freqs[-1], 8
                )
            )
        ]
        self.ax10.set_yticklabels(yticks_freq)
        which_test = self.which_test.get()
        # Make histogram
        # self.ax22.hist(self.data.ravel(), bins=52, density=True)

        # show stat tests
        self.stat_test()
        # (self.im_test_ver,) = self.ax10.plot(
        #    self.ver_test, bp_y, label=f"{self.which_test.get()}"
        # )
        (self.im_time_clean,) = ax02.plot(
            self.data_masked.mean(axis=0), label="Timeseries"
        )
        ax02.set_xlim([0, self.gulp_size])
        ax02.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.ax10.set_ylim([-1, len(self.bandpass) + 1])
        self.ax10.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        self.im_ft_masked = ax12.imshow(
            self.data_masked,
            aspect="auto",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="none",
        )
        # interpolation="none" stops large amount
        # of white space from being shown
        ax12.text(
            0.05,
            0.95,
            "Cleaned",
            verticalalignment="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.1",
                "alpha": 0.4,
            },
        )
        (self.im_bandpass_clean,) = ax13.plot(
            np.ma.mean(self.data_masked, axis=1), bp_y, label="Bandpass"
        )
        ax13.set_ylim([-1, len(self.bandpass) + 1])
        ax13.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        test_median = np.median(self.test_values)
        test_std = np.std(self.test_values)
        self.test_vmax = min(np.max(self.test_values), test_median + 4 * test_std)
        self.test_vmin = max(np.min(self.test_values), test_median - 4 * test_std)

        self.im_test_values = self.ax21.imshow(
            self.test_values, vmin=self.test_vmin, vmax=self.test_vmax, aspect="auto"
        )
        self.ax21.text(
            0.05,
            0.95,
            f"{which_test}",
            verticalalignment="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.1",
                "alpha": 0.4,
            },
        )
        # self.ax21.set_bbox(dic(handletextpad=0, handlelength=0, framealpha=0.4))

        (self.im_test_ver,) = self.ax20.plot(
            self.test_values.mean(axis=1), bp_y, label=f"{which_test}"
        )
        self.ax20.set_ylim([0, self.your_obj.your_header.nchans])
        self.ax20.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        self.ax20.set_yticks(np.linspace(0, self.your_obj.your_header.nchans, 8))
        yticks_freq = [
            str(int(j)) for j in np.linspace(self.your_obj.your_header.nchans, 0, 8)
        ]
        self.ax20.set_yticklabels(yticks_freq)
        self.ax20.set_ylabel("Channel")

        self.ax21.set_xlim([-1, len(self.time_series) + 1])
        # self.ax21.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.im_mask = ax22.imshow(
            self.mask,
            aspect="auto",
            interpolation="none",  # , vmin=self.vmin, vmax=self.vmax
        )
        ax22.text(
            0.05,
            0.95,
            "Mask",
            verticalalignment="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.1",
                "alpha": 0.4,
            },
        )
        (self.im_mask_chan_frac,) = ax23.plot(
            self.mask.mean(axis=1), bp_y, label="Frac."
        )
        # ax13.set_title("Mask Fraction")
        # # puts in the wrong place
        ax23.set_ylim([0, self.your_obj.your_header.nchans])
        ax23.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        (self.im_test_hor,) = self.ax31.plot(
            self.test_values.mean(axis=0), label=f"{which_test}"
        )
        self.ax31.set_xlim([0, self.gulp_size])
        self.ax31.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        (self.im_mask_time_frac,) = ax32.plot(self.mask.mean(axis=0), label="Frac.")
        self.ax31.set_xlabel("Time [sec]")

        ax32.set_xlim(-1, len(self.time_series + 1))
        ax32.legend(handletextpad=0, handlelength=0, framealpha=0.4)
        ax32.set_xlabel("Sample")

        self.set_x_axis()

    def load_file(
        self,
        file_name=[""],
        start_samp=0,
        gulp_size=4096,
        chan_std=False,
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

        if file_name == [""]:
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

        self.make_canvas()

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

    def update_plot(self, *args):
        """
        Redraws the plots when something is changed
        """
        # added *args to make self.which_test.trace("w", self.update_plot)
        # happy
        self.read_data()
        self.set_x_axis()
        self.im_ft.set_data(self.data)
        self.im_bandpass.set_xdata(self.bandpass)
        # self.ax22.cla()
        # https://stackoverflow.com/questions/53258160/update-an-embedded-matplotlib-plot-in-a-pyqt5-gui-with-toolbar
        # self.ax22.hist(self.data.ravel(), bins=52, density=True)
        if self.chan_std:
            self.fill_bp()
        self.im_bandpass.axes.relim()
        self.im_bandpass.axes.autoscale(axis="x")
        self.im_time.set_ydata(np.mean(self.data, axis=0))
        self.im_time.axes.relim()
        self.im_time.axes.autoscale(axis="y")

        self.stat_test()
        which_test = self.which_test.get()
        self.im_test_values.set_data(self.test_values)

        test_median = np.median(self.test_values)
        test_std = np.std(self.test_values)
        self.test_vmax = min(np.max(self.test_values), test_median + 4 * test_std)
        self.test_vmin = max(np.min(self.test_values), test_median - 4 * test_std)
        self.im_test_values.set_clim(vmin=self.test_vmin, vmax=self.test_vmax)
        self.ax21.texts[-1].set_text(f"{which_test}")

        self.im_mask.set_data(self.mask)
        self.im_ft_masked.set_data(self.data_masked)

        self.im_bandpass_clean.set_xdata(self.data_masked.mean(axis=1))
        self.im_bandpass_clean.axes.relim()
        self.im_bandpass_clean.axes.autoscale(axis="x")
        self.im_time_clean.set_ydata(self.data_masked.mean(axis=0))
        self.im_time_clean.axes.relim()
        self.im_time_clean.axes.autoscale(axis="y")

        self.im_mask_chan_frac.set_xdata(self.mask.mean(axis=1))
        self.im_mask_chan_frac.axes.relim()
        self.im_mask_chan_frac.axes.autoscale(axis="x")

        self.im_mask_time_frac.set_ydata(self.mask.mean(axis=0))
        self.im_mask_time_frac.axes.relim()
        self.im_mask_time_frac.axes.autoscale(axis="y")

        self.im_test_ver.set_xdata(self.ver_test)
        self.im_test_ver.axes.relim()
        self.im_test_ver.axes.autoscale(axis="x")

        self.im_test_ver.set_label(f"{which_test}")
        self.ax20.legend(handletextpad=0, handlelength=0, framealpha=0.4)

        self.im_test_hor.set_ydata(self.hor_test)
        self.im_test_hor.axes.relim()
        self.im_test_hor.axes.autoscale(axis="y")
        self.im_test_hor.set_label(f"{which_test}")
        self.ax31.legend(handletextpad=0, handlelength=0, framealpha=0.4)

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
        self.data = self.your_obj.get_data(self.start_samp, self.gulp_size).T
        if self.dm != 0:
            logging.info(f"Dedispersing data at DM: {self.dm}")
            self.data = dedisperse(
                self.data.copy(),
                self.dm,
                self.your_obj.native_tsamp,
                delays=self.dispersion_delays,
            )
        if self.subtract:
            bandpass = bspline_fitter(np.median(self.data, axis=1))
            # fit data to median bandpass
            np.clip(bandpass, self.min, self.max, out=bandpass)
            # make sure the fit is nummerically possable
            self.data = self.data - bandpass[:, None]

            # attempt to return the correct data type,
            # most values are close to zero
            # add get clipped, causeing dynamic range problems
            # diff = np.clip(self.data - bandpass[:, None], self.min, self.max)
            # self.data = diff #diff.astype(self.your_obj.your_header.dtype)

        self.bandpass = np.mean(self.data, axis=1)
        self.time_series = np.mean(self.data, axis=0)
        logging.info(
            f"Displaying {self.gulp_size} samples from sample "
            f"{self.start_samp} i.e {ts:.2f}-{te:.2f}s - gulp mean: "
            f"{np.mean(self.data):.3f}, std: {np.std(self.data):.3f}"
        )

    def set_x_axis(self):
        """
        sets x axis labels in the correct location
        """
        ax_left = self.im_test_hor.axes
        ax_right = self.im_mask_time_frac.axes
        xticks_left = ax_left.get_xticks()
        logging.debug(f"x-axis ticks are {xticks_left}")
        xtick_labels_left = (
            xticks_left + self.start_samp
        ) * self.your_obj.your_header.tsamp
        xtick_labels_right = xticks_left + self.start_samp
        logging.debug(f"Setting x-axis tick labels to {xtick_labels_left}")
        ax_left.set_xticklabels([f"{j:.2f}" for j in xtick_labels_left])
        ax_right.set_xticklabels([f"{int(j)}" for j in xtick_labels_right])

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
            # self.ver_test, self.ver_test_p = stats.normaltest(self.data, axis=1)
            # self.hor_test, self.hor_test_p = stats.normaltest(self.data, axis=0)
            mask, test_values = dagostino_time(self.data.T, return_values=True)
            # TODO plot p values
        elif self.which_test.get() == "IQR":
            mask, test_values = iqr_time(self.data.T, return_values=True)
        elif self.which_test.get() == "Kurtosis":
            # self.ver_test = stats.kurtosis(self.data, axis=1)
            # self.hor_test = stats.kurtosis(self.data, axis=0)
            mask, test_values = kurtosis_time(self.data.T, return_values=True)

        elif self.which_test.get() == "MAD":
            # self.ver_test = stats.median_abs_deviation(self.data, axis=1)
            # self.hor_test = stats.median_abs_deviation(self.data, axis=0)
            mask, test_values = mad_time(self.data.T, return_values=True)
        elif self.which_test.get() == "Skew":
            # self.ver_test = stats.skew(self.data, axis=1)
            # self.hor_test = stats.skew(self.data, axis=0)
            mask, test_values = skew_time(self.data.T, return_values=True)
        elif self.which_test.get() == "Stand. Dev.":
            self.ver_test = np.std(self.data, axis=1)
            self.hor_test = np.std(self.data, axis=0)
        self.mask, self.test_values = mask.T, test_values.T
        self.data_masked = np.ma.array(self.data, mask=self.mask)
        self.ver_test = self.test_values.mean(axis=1)
        self.hor_test = self.test_values.mean(axis=0)


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
        "-no_subtract",
        "--no_bandpass_subtract",
        help="Don't subtract a polynomial fitted bandpass",
        required=False,
        default=True,
        action="store_true",
    )
    # bandpass subtraction help the time series not go wild
    # when block are removed
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
        values.no_bandpass_subtract,
    )  # load file with user params
    root.mainloop()
