#!/usr/bin/env python3
"""
Tests for calculator_cupy.py
"""

import cupy as cp

import jess.calculators_cupy as calc


def test_to_dtype():
    random = cp.random.normal(scale=12, size=512 * 512).reshape(512, 512)
    random_8 = cp.around(random)
    random_8 = cp.clip(random_8, 0, 255)
    random_8 = random_8.astype("uint8")
    assert cp.array_equal(random_8, calc.to_dtype(random, cp.uint8))
