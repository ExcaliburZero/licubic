import collections
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
    #data = data[features_cols]
    data = data[features_cols + [category_col]]

    size_before = len(data)
    data = data.dropna()
    size_after = len(data)

    #n_samples = 100
    #data = data.iloc[:n_samples]

    if size_before > size_after:
        print("%d points removed due to missing feature values." % (size_before - size_after))

    predictions = classifier.predict(data)

    prediction_col = "predicted"
    data[prediction_col] = predictions

    y = np.array(data[category_col])
    y_pred = np.array(data[prediction_col])
    #print("")
    #print(sklearn.metrics.accuracy_score(y, y_pred))

    print(data[[category_col, prediction_col]])

    #print("")
    #print(sklearn.metrics.classification_report(y, y_pred))

    #print(data[[category_col, prediction_col]])

    print("")
    print("Unknown: ", len(data[data[prediction_col] == "Unknown"]))

    sciplt.plotters.plot_confusion_matrix(y, y_pred, hide_zeros=True, normalize=True, x_tick_rotation=90, title_fontsize="large", text_fontsize="large")
    plt.show()

    categories = classifier.categories
    probabilities = classifier.get_class_probabilities(data)
    
    for i in range(probabilities.shape[1]):
        X = probabilities[:,i]

        category = categories[i]

        #sns.kdeplot(X, bw=0.005)
        plt.hist(X, bins=25)
        plt.xlim([0.0, 1.0])
        plt.title(category)
        #plt.show()
        plt.savefig("data/faraway_" + category + ".png")
        plt.close()

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

    return a_vs_not_a

class BlackBoxClassifier(object):

    def __init__(self, a_vs_not_a):
        self.a_vs_not_a = a_vs_not_a

        self.categories = [a for (a, _) in a_vs_not_a.keys()]

        self.unknown = "unknown"
        if self.unknown in self.categories:
            raise Exception("There cannot be a category named %s as this name is reserved." % self.unknown)

        self.unknown_threshold = 0.51

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
