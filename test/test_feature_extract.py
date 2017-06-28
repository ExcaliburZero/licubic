import numpy as np
import unittest

from light_curve_features import feature_extract

TEST_CURVES = [
    (
        np.array([[0.0], [10.9], [12.3], [15.7], [19.2]]),
        np.array([[14.0], [14.5], [14.65], [14.4], [13.9]]),
        np.array([[0.01], [0.009], [0.02], [0.01], [0.01]])
    ),
    (
        np.array([[1.3], [2.7], [3.1], [4.6], [4.9], [5.1], [6.9], [10.2]]),
        np.array([[15.0], [15.1], [14.9], [15.3], [15.2], [15.4], [14.9], [15.0]]),
        np.array([[0.02], [0.01], [0.0093], [0.0089], [0.01], [0.015], [0.019], [0.02]])
    ),
    (
        np.array([[0.1], [9.8]]),
        np.array([[14.1], [15.9]]),
        np.array([[0.01], [0.02]])
    ),
    (
        np.array([[0.1], [1.2], [1.9], [2.1], [2.4], [2.7], [2.9], [3.33], [3.42], [3.7], [4.2], [4.5], [4.9]]),
        np.array([[14.0], [14.1], [14.3], [15.2], [16.3], [15.1], [14.2], [14.0], [13.8], [13.5], [13.9], [14.1], [14.0]]),
        np.array([[0.01], [0.01], [0.02], [0.0198], [0.012], [0.013], [0.01], [0.02], [0.0098], [0.01], [0.03], [0.01], [0.02]])
    )
]

TIMES = 0
MAGNITUDES = 1
ERRORS = 2

def try_expected(self, curves, expected_results, test):
    """
    Runs the given test function against the given curves and compares the
    results to the given expected results.
    """
    for i in range(len(curves)):
        curve = curves[i]

        expected = expected_results[i]
        actual = test(curve)

        if np.isnan(expected):
            self.assertTrue(np.isnan(actual))
        else:
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
            return feature_extract.linear_trend(times, magnitudes)

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
            return feature_extract.magnitude_ratio(magnitudes)

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
            return feature_extract.maximum_slope(times, magnitudes)

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
            return feature_extract.beyond_1std(magnitudes)

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
            return feature_extract.r_cor_bor(magnitudes)

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
            return feature_extract.median_absolute_deviation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_median_buffer_range_percentage(self):
        expected_results = [
            1.0,
            1.0,
            1.0,
            0.9230769230769231
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.median_buffer_range_percentage(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_percent_amplitude(self):
        expected_results = [
            0.25,
            0.34999999999999964,
            0.90000000000000036,
            2.2000000000000011
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.percent_amplitude(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_total_variation(self):
        expected_results = [
            0.28000000000000008,
            0.20000000000000018,
            0.90000000000000036,
            0.44615384615384623
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.total_variation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_quadratic_variation(self):
        expected_results = [
            0.11700000000000003,
            0.065000000000000072,
            1.6200000000000012,
            0.36153846153846192
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.quadratic_variation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_periodicity_metric(self):
        expected_results = [
            1.0029310383570502,
            0.9911085682696126,
            0.9976110027920946,
            0.9737421892363514
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]

            period = 0.5
            phase_times = feature_extract.phase_fold(times, period)

            sm_phase_times, sm_phase_magnitudes = feature_extract.smooth_curve(phase_times, magnitudes)

            lc_rms = feature_extract.root_mean_square(magnitudes)
            sm_phase_rms = feature_extract.root_mean_square(sm_phase_magnitudes)

            return feature_extract.periodicity_metric(lc_rms, sm_phase_rms)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_light_curve_flux_asymmetry(self):
        expected_results = [
            np.nan,
            np.nan,
            np.nan,
            -0.0069616264413517003
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            lc_rms = feature_extract.root_mean_square(magnitudes)

            return feature_extract.light_curve_flux_asymmetry(magnitudes, lc_rms)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_root_mean_square(self):
        expected_results = [
            14.29295280898947,
            15.100993344810137,
            15.026975743641833,
            14.364459346167273
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.root_mean_square(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_chi_2_test(self):
        expected_results = [
            5823845.9725812068,
            9882084.1746215299,
            659891.24999999988,
            14702747.809515834
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            errors = curve[ERRORS]

            return feature_extract.chi_2_test(magnitudes, errors)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_interquartile_range(self):
        expected_results = [
            -0.59999999999999964,
            -1.2999999999999989
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extract.interquartile_range(magnitudes)

        try_expected(self, [TEST_CURVES[0]] + [TEST_CURVES[3]], expected_results, test)
