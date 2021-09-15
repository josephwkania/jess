#!/usr/bin/env python3
"""
Tests for dedispersion.py
"""

import numpy as np

from jess.dispersion import calc_dispersion_delays, dedisperse, delay_lost

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201


class TestDedisperson:
    """
    Class for dedispersion tests
    """

    def setup_class(self):
        """
        Make some frequencies
        """
        self.tsamp = 0.1
        self.time = self.tsamp * np.arange(0, 3)
        self.dm = 100
        self.freq = 1 / np.sqrt(
            1 / 1400 ** 2 - 1000 * self.time / (4148808.0 * self.dm)
        )

        self.disperesed = np.zeros((3, 3), dtype=int)
        np.fill_diagonal(self.disperesed, 1)
        self.disperesed = np.flip(self.disperesed, axis=1)
        self.delays = np.array([0, 1, 2])
        self.dedisp = np.zeros((3, 3), dtype=int)
        self.dedisp[2] = 1

    def test_dedisperse_no_freq(self):
        """
        Make a dispersed array, dedisperse
        Give delays
        """

        assert np.array_equal(
            dedisperse(self.disperesed, dm=0, tsamp=1, delays=self.delays), self.dedisp
        )

    def test_dedisperse(self):
        """
        Make a dispersed array, dedisperse
        Give frequencies
        """
        assert np.array_equal(
            dedisperse(
                self.disperesed, dm=self.dm, tsamp=self.tsamp, chan_freqs=self.freq
            ),
            self.dedisp,
        )

    def test_delay_lost(self):
        """
        test if delay lost return correct number

        Delays should be one less then that length
        of the array
        """
        print(delay_lost(self.dm, self.freq, tsamp=self.tsamp))
        assert len(self.freq) - 1 == delay_lost(self.dm, self.freq, tsamp=self.tsamp)

    def test_calc_dispersion_delays(self):
        """
        Make some dispersion delays
        """
        assert np.allclose(
            calc_dispersion_delays(self.dm, chan_freqs=self.freq), self.time
        )
