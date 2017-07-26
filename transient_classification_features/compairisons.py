from dask import delayed

import itertools
import numpy as np
import sklearn.ensemble
import sklearn.model_selection

def feature_matrix(category_col, features_cols, data):
    categories = data[category_col].unique()

    combinations = list(itertools.combinations_with_replacement(categories, 2))

    print(combinations)

    calculations = []
    for (a, b) in combinations:
        calc = delayed(compute_cell)(category_col, features_cols, data, a, b)
        calculations.append(calc)

    #cells = [compute_cell(category_col, features_cols, data, a, b) for (a, b) in combinations]
    cells = delayed(calculations).compute()

    matrix = {}
    for i in range(len(cells)):
        matrix[combinations[i]] = cells[i]

    return matrix

def compute_cell(category_col, features_cols, data, a, b):
    if a == b:
        return a_against_all(category_col, features_cols, data, a)
    else:
        return a_against_b(category_col, features_cols, data, a, b)

def a_against_all(category_col, features_cols, data, a):
    new_data = data.copy()

    is_a_col = "is_a"
    new_data[is_a_col] = new_data[category_col].map(lambda x: x == a)

    features_and_score = calculate_features(is_a_col, features_cols, new_data)

    print(a + " vs ~" + a)

    return features_and_score

def a_against_b(category_col, features_cols, data, a, b):
    new_data = data[data[category_col].isin([a, b])]

    features_and_score = calculate_features(category_col, features_cols, new_data)

    print(a + " vs " + b)

    return features_and_score

def calculate_features(category_col, features_cols, data):
    test_size = 0.2
    n_best = 3
    random_state = 42

    X = data.as_matrix(features_cols)
    y = data[category_col]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", random_state=random_state)
    rf.fit(X_train, y_train)

    #score = rf.score(X_test, y_test)

    importances = rf.feature_importances_
    feature_indexes = np.argsort(importances)[::-1][:n_best]

    best_features = np.array(features_cols)[feature_indexes]

    best_features_info = list(zip(best_features, importances[feature_indexes]))

    X_2 = data.as_matrix(best_features)
    y_2 = data[category_col]

    X_train_2, X_test_2, y_train_2, y_test_2 = sklearn.model_selection.train_test_split(
        X_2, y_2, test_size=test_size, random_state=random_state)

    rf_2 = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", random_state=random_state)
    rf_2.fit(X_train_2, y_train_2)

    score = rf_2.score(X_test_2, y_test_2)

    return best_features_info, score
