import numpy as np
import unittest

from light_curve_features.feature_extract import *

TEST_CURVES = [
    (
        np.array([[0.0], [10.9], [12.3], [15.7], [19.2]]),
        np.array([[14.0], [14.5], [14.65], [14.4], [13.9]])
    ),
    (
        np.array([[1.3], [2.7], [3.1], [4.6], [4.9], [5.1], [6.9], [10.2]]),
        np.array([[15.0], [15.1], [14.9], [15.3], [15.2], [15.4], [14.9], [15.0]])
    ),
    (
        np.array([[0.1], [9.8]]),
        np.array([[14.1], [15.9]]),
    ),
    (
        np.array([[0.1], [1.2], [1.9], [2.1], [2.4], [2.7], [2.9], [3.33], [3.42], [3.7], [4.2], [4.5], [4.9]]),
        np.array([[14.0], [14.1], [14.3], [15.2], [16.3], [15.1], [14.2], [14.0], [13.8], [13.5], [13.9], [14.1], [14.0]])
    )
]

TIMES = 0
MAGNITUDES = 1

def try_expected(self, curves, expected_results, test):
    """
    Runs the given test function against the given curves and compares the
    results to the given expected results.
    """
    for i in range(len(curves)):
        curve = curves[i]

        expected = expected_results[i]
        actual = test(curve)

        self.assertEqual(expected, actual)

class TestFeatureExtract(unittest.TestCase):
    """
    The tests defined here are regression tests used primarily to ensure that
    changes to the functions do not lead to changes in the program results.

    Each of the expected results here were caclulated using the existing code,
    and thus they are only as correct as the implementation they were calulated
    using.

    Small changes in the calculated values may be acceptable depending on the
    surrounding circumstances and nature of the change.
    """

    def test_linear_trend(self):
        expected_results = [
            0.004550040931330576,
            -0.0039444027047332722,
            0.18556701030927839,
            -0.14333805216641998
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]
            return linear_trend(times, magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_magnitude_ratio(self):
        expected_results = [
            0.4,
            0.5,
            0.5,
            0.38461538461538464
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return magnitude_ratio(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_maximum_slope(self):
        expected_results = [
            0.10714285714285737,
            1.0000000000000089,
            0.18556701030927841,
            4.4999999999999893
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]
            return maximum_slope(times, magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_beyond_1std(self):
        expected_results = [
            0.4,
            0.5,
            0.0,
            0.3076923076923077
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return beyond_1std(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_r_cor_bor(self):
        expected_results = [
            0.0,
            0.0,
            0.0,
            0.07692307692307693
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return r_cor_bor(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_median_absolute_deviation(self):
        expected_results = [
            0.25,
            0.14999999999999947,
            0.90000000000000036,
            0.19999999999999929
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return median_absolute_deviation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)
