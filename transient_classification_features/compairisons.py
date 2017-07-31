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
    random_state = 42

    X = data.as_matrix(features_cols)
    y = np.array(data[category_col])

    score_1, importances = get_feature_importances(X, y, labels)

    feature_indexes = np.argsort(importances)[::-1]
    sorted_features_cols = np.array(features_cols)[feature_indexes]

    n_best = choose_num_features(data, category_col, sorted_features_cols)

    best_feature_indexes = feature_indexes[:n_best]

    best_features = np.array(features_cols)[best_feature_indexes]

    best_features_info = list(zip(best_features, importances[best_feature_indexes]))

    X_2 = data.as_matrix(best_features)
    y_2 = y

    score_2, _ = get_feature_importances(X_2, y_2, labels)

    a_examples = len(data[data[category_col] == labels[0]])
    b_examples = len(data[data[category_col] == labels[1]])

    return best_features_info, score_1, score_2, a_examples, b_examples

def get_feature_importances(X, y, labels):
    n_estimators = 10
    random_state = 42
    cv = 5
    shuffle = True

    kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    cv_scores = []
    importances = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

        rf.fit(X_train, y_train)

        importances.append(rf.feature_importances_)
        cv_scores.append(confusion_matrix(rf, X_test, y_test, labels))

    mean_score = np.mean(cv_scores, axis=0)

    importances = np.array(importances)
    #importances = np.mean(importances, axis=0)
    importances = importances[0]

    return mean_score, importances

def choose_num_features(data, category_col, sorted_features_cols):
    n_estimators = 10
    random_state = 42
    cv = 5
    shuffle = True

    increase_threshold = 0.03#0.05

    y = np.array(data[category_col])

    num_features = 1
    prev_score = 0.0
    for i in range(len(sorted_features_cols)):
        features = sorted_features_cols[0:i + 1]

        X = data.as_matrix(features)

        kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        cv_scores = []
        importances = []
        for train, test in kf.split(X, y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

            rf.fit(X_train, y_train)

            importances.append(rf.feature_importances_)
            cv_scores.append(f1_score(rf, X_test, y_test))

        mean_score = np.mean(cv_scores)

        if mean_score - prev_score > increase_threshold:
            num_features = i + 1
        else:
            break

        prev_score = mean_score

    return num_features

def confusion_matrix(model, X, y, labels):
    y = np.array(y)

    y_pred = model.predict(X)

    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm

def f1_score(model, X, y):
    y = np.array(y)

    y_pred = model.predict(X)

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(np.concatenate([y, y_pred]))

    y = encoder.transform(y)
    y_pred = encoder.transform(y_pred)

    return sklearn.metrics.f1_score(y, y_pred)
