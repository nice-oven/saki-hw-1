import re

from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2, mutual_info_classif
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from datetime import datetime
import csv

filename = 'C:/Users/formlos moin/source/repos/saki/01/Exercise 1 - Classification (Prof OSS)/SAKI Exercise 1 - Transaction Classification - Data Set.csv'

# header ;Auftragskonto;Buchungstag;Valutadatum;Buchungstext;Verwendungszweck;Beguenstigter/Zahlungspflichtiger;Kontonummer;BLZ;Betrag;Waehrung;label


class MixedNaive(BaseEstimator):
    """
    here to combine different naive bayes methods so that they can be used with cross_validate
    """
    def __init__(self, scales, min_cats):
        self._scales = scales
        self._min_cats = min_cats
        self._clfs = []
        self._log_class_prior = None
        self._class_prior = None

        self._vectorizer = None
        self._top_features = None

    @property
    def scales(self):
        return self._scales

    @property
    def class_prior(self):
        return self._class_prior

    @property
    def class_log_prior(self):
        return self._log_class_prior

    @property
    def min_cats(self):
        return self._min_cats

    def fit(self, X, y, **kwargs):
        # check if text must be preprocessed
        for i, feature in enumerate(self._scales.iloc[0]):
            if feature == "multinomial_textclass":
                # tokenize
                tokenized = [" ".join(re.findall("[a-zA-Z]+", text)) for text in
                             X.iloc[:, i]]  # todo fragile...
                # vectorize
                self._vectorizer = CountVectorizer()
                vectorized = self._vectorizer.fit_transform(tokenized)
                # pick top k relevant features of each evaluation, merge into one list
                chi_2_vals = chi2(vectorized, y)[0]
                mi_vals = mutual_info_classif(vectorized, y)
                top_k = 20  # todo put somewhere where it makes more sense
                top_chi_2 = np.argpartition(chi_2_vals, -1 * top_k)
                top_mi = np.argpartition(mi_vals, -1 * top_k)
                self._top_features = set(np.concatenate((top_chi_2[-1*top_k:], top_mi[-1*top_k:])))
                # put them into table with scale multinomial_feature
                for j, top_feature in enumerate(self._top_features):
                    col_name = "multi_" + str(j) + "_" + self._vectorizer.get_feature_names()[top_feature]
                    X.insert(0, col_name, vectorized.toarray()[:, top_feature])
                    self._scales.insert(0, col_name, "multinomial")

        # initialize classifiers
        used_scales = set(self._scales.iloc[0])

        for scale in used_scales:
            if scale == 'nominal':
                nom_mask = np.where(self._scales == 'nominal', True, False)
                nom_mask = nom_mask.reshape((nom_mask.size, ))
                # todo min_cats = ... find a non-info leaking way to deal with new categories in test
                # get number of cats
                cnb = CategoricalNB(min_categories=self._min_cats)
                self._clfs.append([cnb, nom_mask])
            if scale == 'ratio':
                ratio_mask = np.where(self._scales == 'ratio', True, False)
                ratio_mask = ratio_mask.reshape((ratio_mask.size, ))
                gnb = GaussianNB()
                self._clfs.append([gnb, ratio_mask])
            if scale == 'multinomial':
                pass  # todo delete this case?!
            if scale == 'multinomial_textclass':
                multinomial_mask = np.where(self._scales == 'multinomial', True, False)
                multinomial_mask = multinomial_mask.reshape((multinomial_mask.size, ))
                mnb = MultinomialNB()
                self._clfs.append([mnb, multinomial_mask])

        # fit classifiers
        for clf in self._clfs:
            clf[0].fit(X.iloc[:, clf[1]], y)

        # set prior
        if type(self._clfs[0][0]) == CategoricalNB or type(self._clfs[0][0]) == MultinomialNB:
            self._log_class_prior = self._clfs[0][0].class_log_prior_
            self._class_prior = np.exp(self._log_class_prior)
        elif type(self._clfs[0][0]) == GaussianNB:
            self._class_prior = self._clfs[0][0].class_prior_
            self._log_class_prior = np.log(self._class_prior)


    def predict_proba(self, X):  # todo check order of scales / features
        out_proba = 1
        for clf in self._clfs:
            out_proba *= clf[0].predict_proba(X.iloc[:, clf[1]])
        out_proba /= self._class_prior**(len(self._clfs)-1)
        return out_proba

    def predict_log_proba(self, X):  # todo add feature preprocessing similar to fit
        # preprocess multinomial data, if existent
        for i, scale in enumerate(self._scales.iloc[0]):
            if scale == "multinomial_textclass":
                idx = list(X.columns).index(self._scales.columns[i])
                # tokenize
                tokenized = [" ".join(re.findall("[a-zA-Z]+", text)) for text in
                             X.iloc[:, idx]]  # todo fragile...
                # vectorize
                vectorized = self._vectorizer.transform(tokenized)

                for j, top_feature in enumerate(self._top_features):
                    col_name = "multi_" + str(j) + "_" + self._vectorizer.get_feature_names()[top_feature]
                    X.insert(0, col_name, vectorized.toarray()[:, top_feature])

        out_log_proba = 0
        for clf in self._clfs:
            try:
                out_log_proba += clf[0].predict_log_proba(X.iloc[:, clf[1]])
            except IndexError as ie:
                print("error predicting: sample category unknown to CatNB")
            except UserWarning as uw:
                print("nok")
        out_log_proba -= self.class_log_prior * (len(self._clfs) - 1)
        expanded_sum = np.expand_dims(np.log(np.sum(np.exp(out_log_proba), axis=1)), axis=1)
        out_log_proba_normalized = out_log_proba - expanded_sum
        return out_log_proba_normalized

    def predict(self, X, **kwargs):
        pred = self.predict_log_proba(X)
        return self._clfs[0][0].classes_[np.argmax(pred, axis=1)]

    def score(self, X, y, **kwargs):
        """
        accuracy
        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

class Data:
    def __init__(self):
        self._dataset = pd.read_csv(filename, delimiter=";")
        self._scales = pd.DataFrame.from_dict({"Auftragskonto": ['nominal'],
             "Buchungstag": ['nominal'],  # todo make ordinal through either reading it as a date or derive features (DoW)
             "Valutadatum": ['nominal'],
             "Buchungstext": ['nominal'],
             "Verwendungszweck": ['multinomial_textclass'],
             "Beguenstigter/Zahlungspflichtiger": ['nominal'],
             "Kontonummer": ['nominal'],
             "BLZ": ['nominal'],
             "Betrag": ['ratio']})

    @property
    def x(self):
        return self._dataset[self._scales.columns]

    @property
    def y(self):
        return self._dataset["label"]

    @property
    def scales(self):
        return self._scales

    def cleanse(self):
        for i in range(self._dataset.shape[1]):
            filler = "0" if type(self._dataset.iloc[0, i]) == str else 0  # todo make sure filler value is new to the column
            if self._dataset.iloc[:, i].isna().any():
                print("filling missing values in " + self._dataset.columns[i] + " with " +
                      (("#" + str(filler)) if type(filler) == int else ("'" + filler + "'")))
                self._dataset.iloc[:, i] = self._dataset.iloc[:, i].fillna(filler)

    def encode(self):
        # encode ordinal data
        ord_mask = np.where(self._scales == 'nominal', True, False).squeeze()
        ord_mask = np.concatenate(([False], ord_mask, [False, False]))  # set false for index, currency, label
        ord_enc = OrdinalEncoder()  # later use with unknown_value
        ord_enc.fit(self._dataset.iloc[:, ord_mask])
        self._dataset.iloc[:, ord_mask] = ord_enc.transform(self._dataset.iloc[:, ord_mask])

        # take labels & encode

        lbl_enc = LabelEncoder()
        lbl_enc.fit(self._dataset['label'])
        self._dataset['label'] = lbl_enc.transform(self._dataset['label'])

    def prep_date_features(self):
        # todo
        # bd = Buchungsdatum
        # vd = Valutadatum <- können wir uns sparen, da nur für #125 nicht gleich
        # Suche nach:
        #   - wochentag
        #   - wochenende ja / nein
        #   - Anfang / Ende des Monats

        # wochentag
        weekdays = [datetime.strptime(date, "%d.%m.%Y").weekday() for date in self._dataset["Buchungstag"]]
        self._dataset.insert(1, "Wochentag", weekdays)
        self._scales.insert(0, "Wochentag", "nominal")

        # tim tag im monat
        dom = [datetime.strptime(date, "%d.%m.%Y").strftime("%d") for date in self._dataset["Buchungstag"]]
        self._dataset.insert(1, "TIM", dom)
        self._scales.insert(0, "TIM", "nominal")

        print("preprocessed date features")

    def prep_bow(self):
        raise NotImplementedError


def load_data_test():
    data = Data()
    data.cleanse()
    data.prep_date_features()
    data.encode()

def combine_two_categorical():
    data = Data()
    data.cleanse()
    data.encode()

    knr = np.array(data.x["Kontonummer"]).reshape((-1, 1))
    text = np.array(data.x["BLZ"]).reshape((-1, 1))

    k_cat_nb = CategoricalNB()
    t_cat_nb = CategoricalNB()

    k_cat_nb.fit(knr, data.y)
    t_cat_nb.fit(text, data.y)

    k_proba = k_cat_nb.predict_log_proba(knr)
    t_proba = t_cat_nb.predict_log_proba(text)

    combined_proba = k_proba + t_proba - k_cat_nb.class_log_prior_
    combined_proba -= np.expand_dims(logsumexp(combined_proba, axis=1), axis=1)

    # now the same thing but in one cat_nb
    combi = data.x.loc[:, ["BLZ", "Kontonummer"]]

    c_cat_nb = CategoricalNB()

    c_cat_nb.fit(combi, data.y)

    proba = c_cat_nb.predict_log_proba(combi)

    diff = np.exp(proba) - np.exp(combined_proba)

    print("total difference in probabilities: %d" % np.sum(np.abs(diff)))


def run_combined_naive_bayes(x, y, scales, features):
    # set up for cross val
    # run each fold
    # show result
    # to categorical
    x = x.loc[:, features]
    scales = scales.loc[:, features]

    nom_mask = np.where(scales == 'nominal', True, False)
    nom_mask = nom_mask.reshape((nom_mask.size, ))
    min_cats = [np.unique(x.iloc[:, i]).size for i in range(x.shape[1]) if nom_mask[i]]

    iterations = 10
    score = 0
    f1 = 0
    for j in range(iterations):

        clf = MixedNaive(scales.copy(), min_cats)

        random_order = np.arange(0, x.shape[0])
        np.random.shuffle(random_order)

        x_1 = x.iloc[random_order, :].copy()
        y_1 = y[random_order].copy()

        scoring = ['balanced_accuracy', 'f1_micro']
        scores = cross_validate(clf, x_1, y_1, scoring=scoring)
        score += np.sum(scores['test_balanced_accuracy'])
        f1 += np.sum(scores['test_f1_micro'])
    print("Acc: %.2f" % (score/(5 * iterations)))
    print("F1:  %.2f" % (f1/(5*iterations)))
    return [score/(5 * iterations), f1/(5*iterations)]


def simple_textclass():
    data = Data()
    data.cleanse()
    data.encode()

    mixed_nb = MixedNaive(data.scales.copy())

    mixed_nb.fit(data.x[:180].copy(), data.y[:180].copy())
    mixed_nb.predict_log_proba(data.x[180:].copy())

def bin_str_to_bool_arr(bin_str):
    bool_ = [character == '1' for character in bin_str]
    return np.array(bool_)

def try_all_combi():
    feature_list = np.array(["TIM", "Wochentag", "Auftragskonto", "Buchungstag", "Valutadatum", "Buchungstext", "Verwendungszweck",
                              "Beguenstigter/Zahlungspflichtiger", "Kontonummer", "BLZ", "Betrag"])

    data = Data()
    data.cleanse()
    data.prep_date_features()
    data.encode()

    combis = 2 ** (feature_list.size) - 1  # this is the number that in binary should be size long 1's

    bin_str = [np.binary_repr(col_nr, width=feature_list.size) for col_nr in range(1, combis + 1)]
    bools = [bin_str_to_bool_arr(row) for row in bin_str]

    masks = np.array(bools, dtype=bool)

    file = open("result.csv", "a", newline="")
    writer = csv.writer(file)
    header = ["Acc", "F1"]
    header.extend(feature_list)
    try:
        writer.writerow(header)
        for i, mask in enumerate(masks):
            features = feature_list[mask]

            print("(%d%%)" % int(i / masks.shape[0]), " running on:", features)

            result = run_combined_naive_bayes(data.x, data.y, data.scales, features)
            result.extend(mask)

            writer.writerow(result)
            if i % 10 == 0:
                file.flush()
    finally:
        file.close()


if __name__ == "__main__":
    try_all_combi()


    data = Data()
    data.cleanse()
    data.prep_date_features()
    data.encode()
    run_combined_naive_bayes(data.x, data.y, data.scales, ["Kontonummer", "Wochentag", "Betrag", "Verwendungszweck"])
    print("ok")


    # final adjustments to the features


    # pick features
    feature_list1 = np.array(["Auftragskonto", "Buchungstag", "Valutadatum", "Buchungstext", "Verwendungszweck",
                     "Beguenstigter/Zahlungspflichtiger", "Kontonummer", "BLZ", "Betrag"])

    



    exit()
