from dask import delayed

import itertools
import numpy as np
import operator
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing

def feature_matrix(X, y, feature_names, balanced=False):
    categories = np.unique(y)

    combinations = list(itertools.combinations_with_replacement(categories, 2))

    parallel = True
    if parallel:
        calculations = []
        for (a, b) in combinations:
            calc = delayed(compute_cell)(X, y, feature_names, a, b, balanced)
            calculations.append(calc)

        info = delayed(calculations).compute()
    else:
        info = [compute_cell(X, y, feature_names, a, b, balanced) for (a, b) in combinations]

    matrix = {}
    classifiers = {}
    for i in range(len(info)):
        matrix[combinations[i]] = info[i][0]
        classifiers[combinations[i]] = info[i][1]

    return matrix, classifiers

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

def compute_cell(X, y, feature_names, a, b, balanced):
    if a == b:
        return a_against_all(X, y, feature_names, a, balanced)
    else:
        return a_against_b(X, y, feature_names, a, b, balanced)

def a_against_all(X, y, feature_names, a, balanced):
    is_a = np.vectorize(lambda x: x == a)

    y_2 = is_a(y) 

    labels = np.array([True, False])
    features_score_classifiers = calculate_features(X, y_2, feature_names, labels, balanced)

    print(a + " vs ~" + a)

    return features_score_classifiers

def a_against_b(X, y, feature_names, a, b, balanced):
    indexes = np.where(np.isin(y, [a, b]))
    
    X_2 = X[indexes]
    y_2 = y[indexes]

    labels = np.array([a, b])
    features_score_classifiers = calculate_features(X_2, y_2, feature_names, labels, balanced)

    print(a + " vs " + b)

    return features_score_classifiers

def calculate_features(X, y, feature_names, labels, balanced):
    test_size = 0.2
    random_state = 42

    a = y[y == labels[0]]
    b = y[y == labels[1]]

    a_examples = len(a)
    b_examples = len(b)

    if balanced:
        min_examples = np.min([a_examples, b_examples])

        np.random.seed(random_state)
        a = a.reindex(np.random.permutation(a.index))
        b = b.reindex(np.random.permutation(b.index))

        a_limited = a.iloc[:min_examples]
        b_limited = b.iloc[:min_examples]
        y = pd.concat([a_limited, b_limited])

        a_examples = min_examples
        b_examples = min_examples

    #X = data.as_matrix(features_cols)
    #y = np.array(data[category_col])

    cm_1, importances, _ = get_feature_importances(X, y, labels)

    feature_indexes = np.argsort(importances)[::-1]

    sorted_features_names = np.array(feature_names)[feature_indexes]

    n_best = choose_num_features(X, y, feature_indexes)

    best_feature_indexes = feature_indexes[:n_best]

    best_features = np.array(feature_names)[best_feature_indexes]

    best_features_info = list(zip(best_features, importances[best_feature_indexes]))

    X_2 = X[:,best_feature_indexes]
    y_2 = y

    cm_2, _, cv_score = get_feature_importances(X_2, y_2, labels)

    model = create_final_classifier(X_2, y_2)
    classifier = (best_features, model)

    return (best_features_info, cm_1, cm_2, a_examples, b_examples, cv_score), classifier

def get_feature_importances(X, y, labels):
    n_estimators = 10
    random_state = 42
    cv = 5
    shuffle = True

    #kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    kf = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    cv_cms = []
    cv_scores = []
    importances = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

        rf.fit(X_train, y_train)

        importances.append(rf.feature_importances_)
        cv_cms.append(confusion_matrix(rf, X_test, y_test, labels))
        cv_scores.append(score(rf, X_test, y_test))

    cm = cv_cms[0]
    mean_score = np.mean(cv_scores)

    importances = np.array(importances)
    importances = importances[0]

    return cm, importances, mean_score

def choose_num_features(X, y, sorted_feature_indexes):
    n_estimators = 10
    random_state = 42
    cv = 5
    shuffle = True

    increase_threshold = 0.03#0.05

    num_features = 1
    prev_score = 0.0
    for i in range(len(sorted_feature_indexes)):
        features = sorted_feature_indexes[0:i + 1]

        X_sel = X[:,features]

        kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        cv_scores = []
        importances = []
        for train, test in kf.split(X_sel, y):
            X_train, X_test, y_train, y_test = X_sel[train], X_sel[test], y[train], y[test]

            rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

            rf.fit(X_train, y_train)

            importances.append(rf.feature_importances_)
            cv_scores.append(score(rf, X_test, y_test))

        mean_score = np.mean(cv_scores)

        if mean_score - prev_score > increase_threshold:
            num_features = i + 1
        else:
            break

        prev_score = mean_score

    return num_features

def create_final_classifier(X, y):
    n_estimators = 10
    random_state = 42

    rf = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=random_state)

    rf.fit(X, y)

    return rf

def confusion_matrix(model, X, y, labels):
    y = np.array(y)

    y_pred = model.predict(X)

    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm

def score(model, X, y):
    y = np.array(y)

    y_pred = model.predict(X)

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(np.concatenate([y, y_pred]))

    y = encoder.transform(y)
    y_pred = encoder.transform(y_pred)

    return normalized_matthews_correlation(y, y_pred)

def normalized_matthews_correlation(y, y_pred):
    num_a = y[y == 0].size
    num_b = y[y == 1].size

    if num_a == 0:
        raise ValueError("No actual positive cases given.")
    if num_b == 0:
        raise ValueError("No actual negative cases given.")

    values = get_table_values(y, y_pred)

    t_p = values[values == "t_p"].size
    f_n = values[values == "f_n"].size

    f_p = values[values == "f_p"].size
    t_n = values[values == "t_n"].size

    w_a = 1.0 / num_a
    w_b = 1.0 / num_b

    t_p = t_p * w_a
    f_n = f_n * w_a

    t_n = t_n * w_b
    f_p = f_p * w_b

    numer = t_p * t_n - f_p * f_n

    p_1 = (t_p + f_p)
    p_2 = (t_p + f_n)
    p_3 = (t_n + f_p)
    p_4 = (t_n + f_n)

    denom = p_1 * p_2 * p_3 * p_4

    result = numer / np.sqrt(denom)

    if np.isnan(result):
        result = 0.0

    return result

def get_table_values(y, y_pred):
    results = []
    for i in range(y.size):
        if y_pred[i] == 0:
            if y[i] == y_pred[i]:
                results.append("t_p")
            else:
                results.append("f_p")
        else:
            if y[i] == y_pred[i]:
                results.append("t_n")
            else:
                results.append("f_n")

    return np.array(results)

