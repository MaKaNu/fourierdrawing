"""Fourier Module.
    The fourier module provides functions to calculatee fourier series components.
    
    implements following functions:

    - calc_components_freq
    - calc_components
    - calc_n_fourier_components
"""
from typing import List
import numpy as np

from fourier_drawing.data import load_data, normalize, cartesian2polar


def calc_components_freq(point_array: np.ndarray, freq: int) -> np.complex128:
    """_summary_

    Args:
        point_array (np.ndarray): _description_
        freq (int): _description_

    Returns:
        np.ndarray: _description_
    """
    integral = []
    num_points = point_array.shape[0]
    for t in range(num_points):
        p_cmp = complex(point_array[t, 0], point_array[t, 1])
        integral.append(
            p_cmp * np.exp(-freq * 2 * np.pi * 1j * t / num_points) / num_points
        )
    return sum(integral)


def calc_components(point_array: np.ndarray, num_components: int) -> np.ndarray:
    """_summary_

    Args:
        point_array (np.ndarray): _description_
        num_components (int): _description_

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if not num_components % 2 == 0:
        raise ValueError("num_components should be even!")
    comps = [calc_components_freq(point_array, freq=0)]
    for n in range(1, num_components // 2 + 1):
        comps.append(calc_components_freq(point_array, freq=n))
        comps.append(calc_components_freq(point_array, freq=-n))
    return np.array(comps)


def calc_n_fourier_components(
    point_array: np.ndarray, n=100, normal=True
) -> np.ndarray:
    """calculate number n fourier set components

    Args:
        point_array (np.ndarray): provides the datapoints as nested lists
        n (int, optional): number of calculated components plus constant. Defaults to 100.
        normal (bool, optional): defines if the data should be normalized. Defaults to True.

    Returns:
        np.ndarray: Array of polar coordinates for n components
    """
    if normal:
        point_array = normalize(point_array)
    cartesian_components = calc_components(point_array, n)
    return cartesian2polar(cartesian_components)


if __name__ == "__main__":
    data = load_data("rabbit")
    calc_n_fourier_components(data)
