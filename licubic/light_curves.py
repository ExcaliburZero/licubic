import numpy as np
import pandas as pd

class LightCurveCleaner(object):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return [clean_and_sort(curve) for curve in X]

def clean_and_sort(curve):
    times = curve[:,0]

    _, clean_ind = np.unique(times, return_index=True)

    return curve[clean_ind]

def load_curves_from_file(curves_file):
    curves = pd.read_csv(curves_file)
    curves.columns = ["id", "time", "mag", "magerr", "ra", "dec"]

    ids = np.array(curves["id"].unique())

    groups = curves.groupby(by="id")

    columns = ["time", "mag", "magerr"]
    curves_list = [groups.get_group(i).as_matrix(columns) for i in ids]

    return ids, curves_list

def remove_scarce_classes(min_instances, X, y):
    class_frequencies = np.unique(y, return_counts=True)

    removed_classes = []
    for class_name, count in zip(class_frequencies[0], class_frequencies[1]):
        if count < min_instances:
            removed_classes.append(class_name)

    indexes = np.where(~np.isin(y, removed_classes))[0]

    X_t = [X[i] for i in range(len(X)) if i in indexes]
    y_t = y[indexes]

    return X_t, y_t

class ScarceClassRemover(object):

    def __init__(self, min_instances):
        self.min_instances = min_instances

        self.removed_classes = []

    def fit(self, X, y):
        class_frequencies = np.unique(y, return_counts=True)
        print(class_frequencies)

        for class_name, count in zip(class_frequencies[0], class_frequencies[1]):
            if count < self.min_instances:
                self.removed_classes.append(class_name)

        return self

    def transform(self, X, y):
        indexes = ~y.isin(self.removed_classes)

        return X[indexes], y[indexes]
