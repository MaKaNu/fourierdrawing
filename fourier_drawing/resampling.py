import numpy as np
from scipy.interpolate import interp1d

def resample_contour(point_array: np.ndarray, num_points=100) -> np.ndarray:
    x = point_array[:,0]
    y = point_array[:,1]
    cumulative_distance = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    cumulative_distance = np.insert(cumulative_distance, 0, 0)

    interp_func_x = interp1d(cumulative_distance, x, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(cumulative_distance, y, kind='linear', fill_value='extrapolate')

    new_cumulative_distance = np.linspace(0, cumulative_distance[-1], num_points)

    resampled_x = interp_func_x(new_cumulative_distance)
    resampled_y = interp_func_y(new_cumulative_distance)


    return np.array([resampled_x,resampled_y]).T


