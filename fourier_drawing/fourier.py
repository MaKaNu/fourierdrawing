"""Fourier Module.
    The fourier module provides functions to calculatee fourier series components.
    
    implements following functions:

    - normalize
    - calc_components_freq
    - calc_components
    - cartesian2polar
    - calc_n_fourier_components
"""
import numpy as np

from fourier_drawing.data import load_data


def normalize(point_array: np.ndarray, min_v=-1, max_v=1) -> np.ndarray:
    """_summary_

    Args:
        point_array (np.ndarray): _description_
        min_v (int, optional): _description_. Defaults to -1.
        max_v (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    x = point_array[:, 0]
    y = point_array[:, 1]
    x_norm = (max_v - min_v) * ((x - x.min()) / (x.max() - x.min())) + min_v
    y_norm = (max_v - min_v) * ((y - y.min()) / (y.max() - y.min())) + min_v
    return np.array([x_norm, y_norm]).T


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
    for n in range(1, num_components // 2):
        comps.append(calc_components_freq(point_array, freq=n))
        comps.append(calc_components_freq(point_array, freq=-n))
    return np.array(comps)


def cartesian2polar(
    points: np.complex128 | np.ndarray,
) -> np.ndarray[np.float64]:
    """_summary_

    Args:
        points (np.complex128 | np.ndarray): _description_

    Returns:
        np.ndarray[np.float64]: _description_
    """
    magnitude = np.abs(points)
    angles = np.angle(points, deg=True)
    return np.array([magnitude, angles]).T


def calc_n_fourier_components(dataset="rabbit", n=100, normal=True):
    """_summary_

    Args:
        n (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    point_array = load_data(dataset)
    if normal:
        point_array = normalize(point_array)
    cartesian_components = calc_components(point_array, n)
    return cartesian2polar(cartesian_components)


if __name__ == "__main__":
    calc_n_fourier_components(dataset="ellipse")
