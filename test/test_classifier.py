import numpy as np
import pandas as pd
import unittest

from licubic import classifier

np.random.seed(42)
TEST_CURVES = [
    np.array([
        np.arange(0, 10, 0.5),
        np.sin(np.arange(0, 10, 0.5)) + np.random.normal(0.00, 0.50, 20),
        np.random.normal(0.03, 0.01, 20)
    ]).transpose()

    for _ in range(50)
]

TEST_LABELS = [str(x % 3) for x in range(len(TEST_CURVES))]

N_PREDICTABLE = 100
TEST_CURVES_PREDICTABLE = [
    np.array([
        np.arange(0, 50, 0.5),
        np.sin(np.arange(0, 50, 0.5)) + np.random.normal(5.00, 0.20, 100),
        np.random.normal(0.03, 0.01, 100)
    ]).transpose()
    for _ in range(N_PREDICTABLE)
] + [
    np.array([
        np.arange(0, 50, 0.5),
        np.arange(0, 50, 0.5) + np.random.normal(0.00, 0.20, 100),
        np.random.normal(0.03, 0.01, 100)
    ]).transpose()
    for _ in range(N_PREDICTABLE)
] + [
    np.array([
        np.arange(0, 50, 0.5),
        np.arange(0, -50, -0.5) + np.cos(np.arange(0, 50, 0.5)) + np.random.normal(0.00, 0.20, 100),
        np.random.normal(0.03, 0.01, 100)
    ]).transpose()
    for _ in range(N_PREDICTABLE)
]

TEST_LABELS_PREDICTABLE = ["A" for _ in range(N_PREDICTABLE)] + ["B" for _ in range(N_PREDICTABLE)] + ["C" for _ in range(N_PREDICTABLE)]

class TestClassifier(unittest.TestCase):

    def test_train(self):
        model = classifier.LICUBIC()

        model.train(TEST_CURVES, TEST_LABELS)

        expected = True
        actual = model.trained
        self.assertEqual(actual, expected)

    def test_train_predict(self):
        model = classifier.LICUBIC()

        model.train(TEST_CURVES, TEST_LABELS)

        y = model.predict(TEST_CURVES)

        self.assertEqual(y.shape, (len(TEST_CURVES),))

    def test_train_predictable(self):
        model = classifier.LICUBIC()

        model.train(TEST_CURVES_PREDICTABLE, TEST_LABELS_PREDICTABLE)

        y_pred = model.predict(TEST_CURVES_PREDICTABLE)

        print(model.selected_features)
        model.write_confusion_matrix(TEST_LABELS_PREDICTABLE, y_pred, "cm.png")
        model.write_feature_matrix("site/feature_matrix.html")

        self.assertEqual(list(y_pred), TEST_LABELS_PREDICTABLE)
