#!/usr/bin/env python3
"""
Test for JESS_filters
"""

import numpy as np

import jess.JESS_filters as Jf


# class TestRunFilter:
#     """
#     Tests for run_filter
#     """

#     def test_run_filter_anserson():
#         """
#         Run anderson
#         """


class TestCentralLimit:
    """
    Test for outliers
    """

    def setup_class(self):
        """
        Make the array with some outliers
        """
        self.rand = np.random.normal(size=1024 * 512).reshape(1024, 512)
        self.rand[200, 200] += 8
        self.rand[300, 300] -= 8
        self.rand[244, 244] += 30
        self.rand[333, 333] -= 30

        self.window = 4

    def test_central_limit_masker(self):
        """
        Flag point above and below
        """
        should_mask = np.zeros((1024, 512), dtype=bool)
        should_mask[200, 200] += 8
        should_mask[300, 300] -= 8
        should_mask[244, 244] += 30
        should_mask[333, 333] -= 30
        should_mask = np.repeat(should_mask, self.window, axis=0)
        mask = Jf.central_limit_masker(self.rand, window=self.window, sigma=6)

        assert np.array_equal(should_mask, mask)

    def test_central_limit_masker_no_lower(self):
        """
        Don't flag points below
        """
        should_mask = np.zeros((1024, 512), dtype=bool)
        should_mask[200, 200] += 8
        # should_mask[300, 300] -= 8
        should_mask[244, 244] += 30
        # should_mask[333, 333] -= 30
        should_mask = np.repeat(should_mask, self.window, axis=0)
        mask = Jf.central_limit_masker(
            self.rand, window=self.window, remove_lower=False, sigma=6
        )

        assert np.array_equal(should_mask, mask)
