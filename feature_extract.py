from dask import delayed, multiprocessing
from functools import partial
from os import path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn import linear_model

import math
import numpy as np
import pandas as pd
import sys

def main():
    """
    $ python3 feature_extract.py CatalinaVars.csv curves/ test.csv
    """
    data_file = sys.argv[1]
    curves_dir = sys.argv[2]
    output_file = sys.argv[3]

    data = pd.read_csv(data_file)

    extract_func = partial(extract_with_curve, curves_dir)
    new_data = data.apply(extract_func, axis=1)

    new_data.to_csv(output_file, index=False)

def extract_with_curve(curves_dir, data):
    curve_file = get_curve_file_name(data["Numerical_ID"])
    curve_path = path.join(curves_dir, curve_file)

    if path.exists(curve_path):
        curve = pd.read_csv(curve_path)

        return extract_features(data, curve)
    else:
        new_data = data.copy()

        new_data["lt"] = np.nan
        new_data["mr"] = np.nan
        new_data["ms"] = np.nan
        new_data["b1std"] = np.nan
        new_data["lc_rms"] = np.nan
        new_data["lc_flux_asymmetry"] = np.nan
        new_data["sm_phase_rms"] = np.nan
        new_data["periodicity"] = np.nan

        return new_data

def get_curve_file_name(star_id):
    return "%s.txt" % star_id

def extract_features(data, light_curve):
    new_data = data.copy()
    period = data["Period_(days)"]

    times = light_curve.as_matrix(["MJD"])
    magnitudes = light_curve.as_matrix(["Mag"])

    phase_times = phase_subtract(times, period)

    sm_phase_times, sm_phase_magnitudes = smooth_curve(phase_times, magnitudes)

    lt = linear_trend(times, magnitudes)
    mr = magnitude_ratio(magnitudes)
    ms = maximum_slope(times, magnitudes)
    b1std = beyond_1std(magnitudes)
    lc_rms = root_mean_square(magnitudes)
    lc_flux_asymmetry = light_curve_flux_asymmetry(magnitudes, lc_rms)
    sm_phase_rms = root_mean_square(sm_phase_magnitudes)
    periodicity = periodicity_metric(lc_rms, sm_phase_rms)

    new_data["lt"] = lt
    new_data["mr"] = mr
    new_data["ms"] = ms
    new_data["b1std"] = b1std
    new_data["lc_rms"] = lc_rms
    new_data["lc_flux_asymmetry"] = lc_flux_asymmetry
    new_data["sm_phase_rms"] = sm_phase_rms
    new_data["periodicity"] = periodicity

    return new_data

def linear_trend(times, magnitudes):
    model = linear_model.LinearRegression()
    model.fit(times, magnitudes)

    return model.coef_[0][0]

def magnitude_ratio(magnitudes):
    median = np.median(magnitudes)

    points_above_median = magnitudes[magnitudes > median].size

    return points_above_median / float(magnitudes.size)

def maximum_slope(times, magnitudes):
    max_slope = None

    for i in range(len(times) - 1):
        mag_diff = magnitudes[i + 1][0] - magnitudes[i][0]
        time_diff = times[i + 1][0] - times[i][0]

        if time_diff != 0:
            slope = mag_diff / time_diff

            if max_slope is None or slope > max_slope:
                max_slope = slope

    return max_slope

def beyond_1std(magnitudes):
    std = np.std(magnitudes)
    mean = np.mean(magnitudes)

    gt_1_std = magnitudes[abs(magnitudes - mean) > std].size

    return gt_1_std / magnitudes.size

def phase_subtract(times, period):
    phase_times = np.array([t % period for t in times])

    return phase_times

def smooth_curve(times, magnitudes):
    N = len(times)
    x = times[:,0]
    y = magnitudes[:,0]

    # https://stackoverflow.com/questions/28855928/python-smoothing-data
    smoothed_times = np.linspace(np.min(x), np.max(x), 1000)

    # interpolate + smooth
    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 101, 3
    smoothed_magnitudes = savgol_filter(itp(smoothed_times), window_size, poly_order)

    return (smoothed_times, smoothed_magnitudes)

def periodicity_metric(light_curve_rms, sm_phase_rms):
    return (sm_phase_rms ** 2) / (light_curve_rms ** 2)

def light_curve_flux_asymmetry(magnitudes, light_curve_rms):
    d_med = np.median(magnitudes)

    data_len = len(magnitudes)
    index_10p = int(round(data_len * 0.1))
    top_10 = data_len - index_10p - 1
    bottom_10 = index_10p
    d_10p = magnitudes[top_10:-1] + magnitudes[0:bottom_10]

    mean_d_10p = np.mean(d_10p)

    return (mean_d_10p - d_med) / light_curve_rms

def root_mean_square(xs):
    """
    Returns the root mean square of the given data.

    x_rms = sqrt((1/n) * (x_1 ^ 2 + x_2 ^ 2 + ...))

    Parameters
    ----------
    xs : numpy.ndarray
        The data to take the root mean square of.

    Returns
    -------
    rms : numpy.float64
        The root mean square of the given data.
    """
    squares = xs ** 2
    sum_squares = np.sum(squares)

    rms = math.sqrt((len(xs) ** -1) * sum_squares)
    return rms

if __name__ == "__main__":
    main()
