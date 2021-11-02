import pandas
from CombineDatasets import get_combined_dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
import numpy as np


class CorrectLabels:

    def __init__(self,
                 dataset,
                 label_column_name: str,
                 index_column_name: str,
                 epochs: int,
                 threshold,
                 repeats: int = 500,
                 split_rate: int = 4,
                 ):
        self.dataset = dataset
        self.threshold = threshold
        self.label_column_name = label_column_name
        self.index_column_name = index_column_name
        self.repeats = repeats
        self.epochs = epochs
        self.split_rate = split_rate
        self.models = self.form_models()

    @staticmethod
    def shuffle_dataset(dataset):
        shuffled_dataset = shuffle(dataset)
        return shuffled_dataset

    def get_train_test(self, dataset):
        X = dataset.drop(self.label_column_name)
        y = dataset[self.index_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/self.split_rate), random_state=7)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def form_models(self):
        models = {}
        models["LR"] = LogisticRegression()
        # models["PasAgr"] = PassiveAggressiveClassifier() No predict_proba
        models["KNN"] = KNeighborsClassifier()
        models["SVM"] = SVC()
        models["DCT"] = DecisionTreeClassifier()
        models["RFC"] = RandomForestClassifier()
        models["ABC"] = AdaBoostClassifier()
        models["GBC"] = GradientBoostingClassifier()
        models["GNB"] = GaussianNB()
        models["MNB"] = MultinomialNB()

        return models

    def get_trained_models(self, X_train, y_train):
        fitted_models = {}
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
        return fitted_models

    def get_predict(self, X_test, fitted_models: dict):
        preds = {}
        for model_name, model in fitted_models.items():
            predict = model.predict(X_test)
            predict_proba = model.predict_proba(X_test)
            mask = np.max(predict_proba, axis=1) > self.threshold
            preds[model_name] = (predict, mask)
        return preds

    def repeat(self, dataset):
        for i in range(self.repeats):
            shuffled_dataset = self.shuffle_dataset(dataset)
            X_train, X_test, y_train, y_test = self.get_train_test(shuffled_dataset)
            fitted_models = self.get_trained_models(X_train, y_train)
            predictions = self.get_predict(X_test, fitted_models)



