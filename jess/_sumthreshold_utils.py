#!/usr/bin/env python3
"""
Utils for Sumthreshold
"""

from typing import Dict, List

import numpy as np
from numba import njit, prange
from scipy import ndimage
from scipy.optimize import curve_fit


def binary_mask_dilation(
    mask: np.ndarray, struct_size_0: int, struct_size_1: int
) -> np.ndarray:
    """
    Dilates the mask.

    Args:
        mask - Original mask

        struct_size_0 - Dilation parameter

        struct_size_1 - Dilation parameter

    Return
        dilated mask

    Note:
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/mitigation/sum_threshold.py
    """
    struct = np.ones((struct_size_0, struct_size_1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=struct, iterations=2)


def blob_mitigation(
    dynamic_spectra: np.ndarray,
    baseline: np.ndarray,
    line_mask: np.ndarray,
    threshold: float,
    n_iter: int,
) -> np.ndarray:
    """
    The function to identify the blob RFI

    Args:
        dynamic_spectra - 2D array to clean

        baseline - Estimated baseline of the input data

        line_mask - Band mask

        threshold - First threshold value of the sumthreshold algorithm

    Returns:
        blob RFI mask

    Note:
        From http://zmtt.bao.ac.cn/GPPS/RFI/
    """
    valid_index = np.where(~line_mask)[0]
    valid_data = dynamic_spectra - baseline
    valid_data = valid_data[:, valid_index]
    point_mask_temp = run_sumthreshold_arpls(valid_data, n_iter=n_iter, chi_i=threshold)
    point_mask = np.zeros_like(dynamic_spectra, dtype=bool)
    point_mask[:, valid_index] = point_mask_temp

    return point_mask


def _gaussian_filter(
    dynamic_spectra_pad: np.ndarray,
    n_spectra: int,
    n_chan: int,
    mask_pad: np.ndarray,
    mask: np.ndarray,
    dynamic_filtered: np.ndarray,
    dynamic_filtered_2: np.ndarray,
    kernel_0: np.ndarray,
    kernel_1: np.ndarray,
    window_size_1: int,
    window_size_0: int,
):
    """
    Apply Guassain filter to Dysnamic spectra

    Args:
        dynamic_spectra_pad - Dynamic spectra padded to include kernel sizes

        n_spectra - Number of points along axis 1

        c_chan - Number of points along axis 0

        mask_pad - Mask to exclude, padded with kernel sizes

        mask - Mask to exclude

        dynamic_filtered - Array to put filtered data

        dynamic_filtered - Array to put filtered data, 2nd iteration

        kernel_0 - Kernel for axis 0

        kernel_1 - Kernel for axis 1

        window_size_1 - Kernel window size for axis 1

        window_size_0 - Kernel window size for axis 0

    Returns:
        Guassian filtere 2D ndarray

    Note:
        based on
        https://github.com/cosmo-ethz/seek/blob/master/seek/utils/filter.py
    """
    # pylint: disable=not-an-iterable, invalid-name
    # pylingt can't see that prange is iterable
    window_0_half = window_size_0 // 2
    window_1_half = window_size_1 // 2
    for i in prange((window_0_half), n_spectra + (window_0_half)):
        for j in prange((window_1_half), n_chan + (window_1_half)):
            if mask[i - window_0_half, j - window_1_half]:
                dynamic_filtered[i, j] = 0  # V[i-n2, j-m2]
            else:
                val = np.sum(
                    (
                        mask_pad[i - window_0_half : i + window_0_half + 1, j]
                        * dynamic_spectra_pad[
                            i - window_0_half : i + window_0_half + 1, j
                        ]
                        * kernel_0
                    )
                )
                dynamic_filtered[i, j] = val / np.sum(
                    mask_pad[i - window_0_half : i + window_0_half + 1, j] * kernel_0
                )

    for k in prange((window_1_half), n_chan + (window_1_half)):
        for w in prange((window_0_half), n_spectra + (window_0_half)):
            if mask[w - window_0_half, k - window_1_half]:
                dynamic_filtered_2[w, k] = 0  # V[i2-k, j2-m2]
            else:
                val = np.sum(
                    (
                        mask_pad[w, k - window_1_half : k + window_1_half + 1]
                        * dynamic_filtered[w, k - window_1_half : k + window_1_half + 1]
                        * kernel_1
                    )
                )
                dynamic_filtered_2[w, k] = val / np.sum(
                    mask_pad[w, k - window_1_half : k + window_1_half + 1] * kernel_1
                )
    return dynamic_filtered_2


def gaussian_filter(
    dynamic_spectra: np.ndarray,
    mask: np.ndarray,
    window_size_1: int = 40,
    window_size_0: int = 20,
    sigma_m: float = 0.5,
    sigma_n: float = 0.5,
) -> np.ndarray:
    """
    Applies a gaussian filter (smoothing) to the given two dimesnional array
    taking into account masked values

    Args:
        dynamic_spectra - the value array to be smoothed

        mask -  boolean array defining masked values

        window_size_1 -  kernel window size in axis=1

        window_size_0 -  kernel window size in axis=0

        sigma_m -  kernel sigma in axis=1

        sigma_n - kernel sigma in axis=0

    Returns:
        Filtered array

    Notes:
        based on
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/utils/filter.py
    """
    n_spectra, n_chan = dynamic_spectra.shape
    dynamic_spectra_pad = np.zeros(
        (n_spectra + window_size_0, n_chan + window_size_1), dtype=dynamic_spectra.dtype
    )
    dynamic_spectra_pad[
        window_size_0 // 2 : -window_size_0 // 2,
        window_size_1 // 2 : -window_size_1 // 2,
    ] = dynamic_spectra[:]

    mask_pad = np.zeros(
        (n_spectra + window_size_0, n_chan + window_size_1), dtype=np.bool
    )
    mask_pad[
        window_size_0 // 2 : -window_size_0 // 2,
        window_size_1 // 2 : -window_size_1 // 2,
    ] = ~mask[:]

    dynamic_filtered = np.zeros((n_spectra + window_size_0, n_chan + window_size_1))
    dynamic_filtered_2 = np.zeros((n_spectra + window_size_0, n_chan + window_size_1))

    def twod_guass(
        array_n: np.ndarray, array_m: np.ndarray, sigma_n: float, sigma_m: float
    ) -> np.ndarray:
        """
        Calculate the two dimsional smoothing Guassian

        Args:
            array_n - array along axis=0

            array_m - array along axis=1

            sigma_n - width along axis=0

            sigma_m - widht along axis=1

        Returns:
           2D guassian
        """
        return np.exp(
            -(array_n ** 2) / (2 * sigma_n ** 2) - array_m ** 2 / (2 * sigma_m ** 2)
        )

    array_n = np.arange(-window_size_0 / 2, window_size_0 / 2 + 1)
    array_m = np.arange(-window_size_1 / 2, window_size_1 / 2 + 1)
    kernel_0 = twod_guass(array_n, 0, sigma_n=sigma_n, sigma_m=sigma_m).T
    kernel_1 = twod_guass(0, array_m, sigma_n=sigma_n, sigma_m=sigma_m).T

    dynamic_filtered = _gaussian_filter(
        dynamic_spectra_pad,
        n_spectra,
        n_chan,
        mask_pad,
        mask,
        dynamic_filtered,
        dynamic_filtered_2,
        kernel_0,
        kernel_1,
        window_size_1,
        window_size_0,
    )

    dynamic_filtered = dynamic_filtered[
        window_size_0 // 2 : -window_size_0 // 2,
        window_size_1 // 2 : -window_size_1 // 2,
    ]
    dynamic_filtered[mask] = dynamic_spectra[mask]

    return dynamic_filtered


def get_di_kwargs(struct_size_0: int = 3, struct_size_1: int = 3) -> Dict:
    """
    Creates a dict with the dilation keywords.

    Args:
        struct_size_0 - Struct size in axis=0

        struct_size_1 - Struct size in axis=1

    Return:
        dictionary with the dilation keywords

    Note:
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/mitigation/sum_threshold.py
    """
    return dict(struct_size_0=struct_size_0, struct_size_1=struct_size_1)


def get_sm_kwargs(
    kernel_m: int = 20,
    kernel_n: int = 40,
    sigma_m: float = 15,
    sigma_n: float = 7.5,
) -> Dict:
    """
    Creates a dict with the smoothing keywords.

    Args:
        kernel_m - Kernel window size in axis=1

        kernel_n - Kernel window size in axis=0

        sigma_m - Kernel sigma in axis=1

        sigma_n - Kernel sigma in axis=0

    Returns:
        dictionary with the smoothing keywords

    Notes:
        From
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/mitigation/sum_threshold.py
    """
    return dict(M=kernel_m, N=kernel_n, sigma_m=sigma_m, sigma_n=sigma_n)


def ksigma(dynamic_spectra: np.ndarray) -> np.ndarray:
    """
    The automatic parameter setup based on the K sigma criterion

    Args:
        dynamic_spectra - data calculate

    Return:
        popt - the estimated standard deviation of the input data

    Note:
        From http://zmtt.bao.ac.cn/GPPS/RFI/
    """
    med = np.median(dynamic_spectra)
    hist_result = np.histogram(dynamic_spectra, bins=50, density=True)
    x_val = (hist_result[1][1:] + hist_result[1][:-1]) / 2

    def gaus(x_val: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Guass function to use for fit
        """
        return np.exp(-((x_val - med) ** 2) / (2 * sigma ** 2)) / (
            np.sqrt(2 * np.pi) * sigma
        )

    popt, *_ = curve_fit(gaus, x_val, hist_result[0], p0=[1])
    return popt


def normalize(dynamic_spectra: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Simple normalization of standing waves: subtracting the median over time
    for each frequency.

    Args:
        dynamic_spectra - Spectra chunk to subtract

        mask - Mask to ignore

    Returns:
        normalized data

    Notes:
        From
        https://cosmo-gitlab.phys.ethz.ch/cosmo_public/seek/-/blob/master/seek/mitigation/sum_threshold.py

        Changed so that time is on the vertical axis
    """
    median = np.ma.median(np.ma.MaskedArray(dynamic_spectra, mask), axis=1)
    data = np.abs(dynamic_spectra - median[:, None])
    return data.data


@njit
def _sumthreshold(
    dynamic_spectra: np.ndarray,
    mask: np.ndarray,
    n_iter: int,
    chi: float,
    ds0: int,
    ds1: int,
) -> np.ndarray:
    """
    The operation of summing and thresholding.

    Args:
        dynamic_spectra - Dynamic_spectra to sumthreshold

        mask - Original mask

        n_iter - Number of iterations

        chi - Thresholding criteria

        ds0 - Dimension of the first axis

        ds1 - Dimension of the second axis

    Output:
        2D bool mask

    Note:
        Based on
        https://github.com/cosmo-ethz/seek/blob/master/seek/mitigation/sum_threshold.py
    """
    # pylint: disable=invalid-name
    tmp_mask = mask.copy()

    for x in range(ds0):
        power_sum = 0.0
        cnt = 0

        for ii in range(0, n_iter):
            if not mask[x, ii]:
                power_sum += dynamic_spectra[x, ii]
                cnt += 1

        for y in range(n_iter, ds1):
            if power_sum > chi * cnt:
                for ii2 in range(0, n_iter):
                    tmp_mask[x, y - ii2 - 1] = True

            if not mask[x, y]:
                power_sum += dynamic_spectra[x, y]
                cnt += 1

            if mask[x, y - n_iter] != 1:
                power_sum -= dynamic_spectra[x, y - n_iter]
                cnt -= 1

    return tmp_mask


def run_sumthreshold(
    dynamic_spectra: np.ndarray,
    init_mask: np.ndarray,
    eta: float,
    n_iter: int,
    chi_i: List[float],
    sm_kwargs: Dict,
) -> np.ndarray:
    """
    Perform one SumThreshold operation: sum the un-masked data after
    subtracting a smooth background and threshold it.

    Args:
        dynamic_spectra - Dynamic Spectra to make mask

        init_mask - Initial mask

        eta - Number that scales the chi value for each iteration

        M - Number of iterations

        chi - Thresholding criteria

        sm_kwargs - Smoothing keyword

    Returns:
        SumThreshold mask

    Note:
        based on
        https://github.com/cosmo-ethz/seek/blob/master/seek/mitigation/sum_threshold.py
    """

    smoothed_dynamic = gaussian_filter(dynamic_spectra, init_mask, **sm_kwargs)
    res = dynamic_spectra - smoothed_dynamic

    st_mask = init_mask.copy()

    for i_iter, chi in zip(n_iter, chi_i):
        chi = chi / eta
        if i_iter == 1:
            st_mask = st_mask | (chi <= res)
        else:
            st_mask = _sumthreshold(res, st_mask, i_iter, chi, *res.shape)
            st_mask = _sumthreshold(res.T, st_mask.T, i_iter, chi, *res.T.shape).T

    # if plotting:
    #     sum_threshold_utils.plot_steps(
    #         data, st_mask, smoothed_data, res, "%s (%s)" % (eta, chi_i)
    #     )

    return st_mask


def run_sumthreshold_arpls(
    detrend_dynamic: np.ndarray, n_iter: int, chi_i: List[float]
):
    """
    A function to call sumthreshold for a list of threshold value

    Args:
        detrend_dynamic - Difference of the data and the estimated baseline

        chi_1 - First threshold value

    Returns:
        ArPLS SumThredshold mask

    Note:
        based on
        http://zmtt.bao.ac.cn/GPPS/RFI/
    """
    # use first threshold value to compute the whole list of threshold
    # for sumthreshold algorithm

    if len(detrend_dynamic.shape) == 1:
        detrend_dynamic = detrend_dynamic[:, np.newaxis]
    st_mask = np.full(detrend_dynamic.shape, False)

    for i_iter, chi in zip(n_iter, chi_i):
        if i_iter == 1:
            st_mask = st_mask | (chi <= detrend_dynamic)
        else:
            if detrend_dynamic.shape[-1] != 1:
                st_mask = _sumthreshold(
                    detrend_dynamic, st_mask, i_iter, chi, *detrend_dynamic.shape
                )

            st_mask = _sumthreshold(
                detrend_dynamic.T, st_mask.T, i_iter, chi, *detrend_dynamic.T.shape
            ).T

    return st_mask
