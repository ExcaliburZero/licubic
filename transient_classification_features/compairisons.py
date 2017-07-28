from dask import delayed

import itertools
import numpy as np
import operator
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing

def feature_matrix(category_col, features_cols, data):
    categories = data[category_col].unique()

    combinations = list(itertools.combinations_with_replacement(categories, 2))

    calculations = []
    for (a, b) in combinations:
        calc = delayed(compute_cell)(category_col, features_cols, data, a, b)
        calculations.append(calc)

    cells = delayed(calculations).compute()

    matrix = {}
    for i in range(len(cells)):
        matrix[combinations[i]] = cells[i]

    return matrix

def rank_features(matrix, features):
    best_features = []
    for key in matrix:
        entry = matrix[key]

        for f in entry[0]:
            best_features.append(f[0])

    features_ranking = {x: best_features.count(x) for x in best_features}

    for f in features:
        if f not in features_ranking:
            features_ranking[f] = 0

    return sorted(features_ranking.items(), key=operator.itemgetter(1))[::-1]

def compute_cell(category_col, features_cols, data, a, b):
    if a == b:
        return a_against_all(category_col, features_cols, data, a)
    else:
        return a_against_b(category_col, features_cols, data, a, b)

def a_against_all(category_col, features_cols, data, a):
    new_data = data.copy()

    is_a_col = "is_a"
    new_data[is_a_col] = new_data[category_col].map(lambda x: x == a)

    labels = [True, False]
    features_and_score = calculate_features(is_a_col, features_cols, new_data, labels)

    print(a + " vs ~" + a)

    return features_and_score

def a_against_b(category_col, features_cols, data, a, b):
    new_data = data[data[category_col].isin([a, b])]

    labels = [a, b]
    features_and_score = calculate_features(category_col, features_cols, new_data, labels)

    print(a + " vs " + b)

    return features_and_score

def calculate_features(category_col, features_cols, data, labels):
    test_size = 0.2
    n_best = 3
    random_state = 42

    X = data.as_matrix(features_cols)
    y = np.array(data[category_col])

    score_1, importances = get_feature_importances(X, y, labels)

    feature_indexes = np.argsort(importances)[::-1][:n_best]

    best_features = np.array(features_cols)[feature_indexes]

    best_features_info = list(zip(best_features, importances[feature_indexes]))

    X_2 = data.as_matrix(best_features)
    y_2 = y

    score_2, _ = get_feature_importances(X_2, y_2, labels)

    return best_features_info, score_1, score_2

def get_feature_importances(X, y, labels):
    n_estimators = 10
    random_state = 42
    cv = 2
    shuffle = True

    kf = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    cv_scores = []
    importances = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

        rf.fit(X_train, y_train)

        importances.append(rf.feature_importances_)
        cv_scores.append(score(rf, X_test, y_test, labels))

    mean_score = np.mean(cv_scores, axis=0)

    importances = np.array(importances)
    importances = np.mean(importances, axis=0)

    return mean_score, importances

def score(model, X, y, labels):
    y = np.array(y)

    y_pred = model.predict(X)

    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm
