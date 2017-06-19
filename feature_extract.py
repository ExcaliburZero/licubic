from dask import delayed, multiprocessing
from os import path
from sklearn import linear_model
import math
import numpy as np
import pandas as pd
import sys

def main():
    """
    $ python3 feature_extract.py CatalinaVars.csv test.csv
    """
    data_file = sys.argv[1]
    output_file = sys.argv[2]

    data = pd.read_csv(data_file)

    new_data = extract_partition(data)

    new_data.to_csv(output_file, index=False)

def extract_with_curve(data):
    curve_file = get_curve_file_name(data["Numerical_ID"])

    if path.exists(curve_file):
        curve = pd.read_csv(curve_file)

        return extract_features(data, curve)
    else:
        new_data = data.copy()

        new_data["lt"] = np.nan
        new_data["mr"] = np.nan
        new_data["ms"] = np.nan
        new_data["b1std"] = np.nan

        return new_data

def get_curve_file_name(star_id):
    return "curves/%s.txt" % star_id

def extract_features(data, light_curve):
    new_data = data.copy()
    #period = data["period"]

    times = light_curve.as_matrix(["MJD"])
    magnitudes = light_curve.as_matrix(["Mag"])

    lt = linear_trend(times, magnitudes)
    mr = magnitude_ratio(magnitudes)
    ms = maximum_slope(times, magnitudes)
    b1std = beyond_1std(magnitudes)

    new_data["lt"] = lt
    new_data["mr"] = mr
    new_data["ms"] = ms
    new_data["b1std"] = b1std

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

if __name__ == "__main__":
    main()
