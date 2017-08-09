import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotnine as plt9
import scikitplot as sciplt
import seaborn as sns
import sklearn.metrics
import sys

def main():
    classifiers_file = sys.argv[1]
    data_file = sys.argv[2]

    classifier = create_classifier(classifiers_file)

    data = pd.read_csv(data_file)

    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr", "fpr20", "fpr35", "fpr50", "fpr65", "fpr80"
        ,  "roms", "ptpv", "skewness", "kurtosis", "ampl", "stetson_I", "stetson_J", "stetson_K", "pst", "pdfp", "sk"
        ,  "cum_sum"
        ,  "neumann_eta", "residual_br_fa_ratio", "shapiro_wilk", "slopes_10per"
        ,  "slopes_90per", "abv_1std", "abv_1std_slopes", "bel_1std", "bel_1std_slopes"
        ]

    category_col = "category"
    data = data[features_cols + [category_col]]

    data = remove_partial_entries(data)

    category_minimum = 0
    data = remove_small_categories(data, category_col, features_cols, category_minimum)

    predictions = classifier.predict(data)

    y = np.array(data[category_col])
    y_pred = predictions

    date_time = str(datetime.datetime.now())
    filepath = "cm_%s.svg" % date_time
    save_confusion_matrix(y, y_pred, filepath, data, category_col, classifier)

def remove_partial_entries(data):
    size_before = len(data)
    data = data.dropna()
    size_after = len(data)

    if size_before > size_after:
        print("%d points removed due to missing feature values." % (size_before - size_after))

    return data

def remove_small_categories(data, category_col, features_cols, category_minimum):
    categories = data[category_col].unique()

    category_info = data.groupby(category_col).count()[features_cols[0]]
    good_categories = category_info.index[np.array(category_info) > category_minimum]

    good_categories = [x for x in categories if x in good_categories]

    num_removed_categories = len(categories) - len(good_categories)
    if num_removed_categories > 0:
        print("Removed %d categories with less than %d instances." % (num_removed_categories, category_minimum))

    data = data[data[category_col].isin(good_categories)]

    return data

def save_confusion_matrix(y, y_pred, filepath, data, category_col, classifier):
    true_cats = np.array(data[category_col].unique())
    pred_cats = np.array(classifier.categories + [classifier.unknown])

    labels = np.sort(np.unique(np.concatenate([true_cats, pred_cats])))
    true_labels = np.where(np.isin(labels, true_cats))
    pred_labels = np.where(np.isin(labels, pred_cats))

    #sns.set_context("talk", font_scale=2.2)
    #figsize = (54, 38)
    #figsize = (20, 16)
    figsize = (26, 18)
    fontsize = 20
    sns.set_context("poster")
    #figsize = (20, 16)
    #fontsize = 20

    title = "Confusion Matrix ~ CRTS North -> CRTS South"

    #normalize = "percent"
    normalize = True
    ax = sciplt.plotters.plot_confusion_matrix(y, y_pred, hide_zeros=True, normalize=normalize, x_tick_rotation=90, title_fontsize="large", text_fontsize="large", true_label_indexes=true_labels, pred_label_indexes=pred_labels, labels=labels, figsize=figsize, title=title)
    for text in ax.texts:
        #text.set_weight('bold')
        #text.set_fontsize(80)
        text.set_fontsize(fontsize * 1.2)
    for text in ax.xaxis.get_ticklabels():
        text.set_fontsize(fontsize)
    #    text.set_weight('bold')
    for text in ax.yaxis.get_ticklabels():
        text.set_fontsize(fontsize)
    #    text.set_weight('bold')
    #ax.xaxis.label.set_weight('bold')
    #ax.yaxis.label.set_weight('bold')
    #ax.title.set_weight('bold')
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.title.set_fontsize(fontsize)
    #plt.show()
    plt.savefig(filepath)

def create_classifier(classifiers_file):
    binary_classifiers = load_binary_classifiers(classifiers_file)

    a_vs_not_a = get_a_versus_not_a(binary_classifiers)

    classifier = BlackBoxClassifier(a_vs_not_a)

    return classifier

def load_binary_classifiers(classifiers_file):
    with open(classifiers_file, "rb") as f:
        return pickle.load(f)

def get_a_versus_not_a(binary_classifiers):
    binaries = binary_classifiers.keys()
    wanted_binaries = [(a, b) for (a, b) in binaries if a == b]

    a_vs_not_a = collections.OrderedDict({})
    for binary in wanted_binaries:
        classifier = binary_classifiers[binary]
        a_vs_not_a[binary] = classifier

    key_order = sorted(a_vs_not_a.keys())
    a_vs_not_a = collections.OrderedDict((k, a_vs_not_a[k]) for k in key_order)

    return a_vs_not_a

class BlackBoxClassifier(object):

    def __init__(self, a_vs_not_a):
        self.a_vs_not_a = a_vs_not_a

        self.categories = [a for (a, _) in a_vs_not_a.keys()]

        self.unknown = "unknown"
        if self.unknown in self.categories:
            raise Exception("There cannot be a category named %s as this name is reserved." % self.unknown)

        self.unknown_threshold = 0.50

    def predict(self, data):
        results = self.get_class_probabilities(data)

        predictions = []
        for r in results:
            best_match = np.argmax(r)
            best_score = r[best_match]

            if best_score > self.unknown_threshold:
                pred = self.categories[best_match]
            else:
                pred = self.unknown

            predictions.append(pred)

        return predictions

    def get_class_probabilities(self, data):
        results = []
        for (features, classifier) in self.a_vs_not_a.values():
            X = data.as_matrix(features)
            res = classifier.predict_proba(X)
            results.append(res)

        results = np.array(results)

        results = results[:,:,1]
        results = np.swapaxes(results, 0, 1)

        return results

if __name__ == "__main__":
    main()
