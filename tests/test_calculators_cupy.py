#!/usr/bin/env python3
"""
Tests for calculator_cupy.py
"""
try:
    import cupy as cp

    CUPY_AVAL = True
except ModuleNotFoundError:
    CUPY_AVAL = False

import pytest

import jess.calculators_cupy as calc


@pytest.mark.skipif(not CUPY_AVAL, reason="cupy not available.")
def test_to_dtype():
    """
    Create some random data and turn it into uint8
    """
    random = cp.random.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = cp.around(random)
    random_8 = cp.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert cp.array_equal(random_8, calc.to_dtype(random, cp.uint8))
