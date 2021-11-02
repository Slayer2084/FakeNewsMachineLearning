import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
import numpy as np
import time


class CorrectLabels:

    def __init__(self,
                 dataset,
                 label_column_name: str,
                 index_column_name: str,
                 epochs: int,
                 threshold,
                 preprocessing_pipe,
                 repeats: int = 500,
                 split_rate: int = 4,
                 ):
        self.dataset = dataset
        self.threshold = threshold
        self.preprocessing_pipe = preprocessing_pipe
        self.label_column_name = label_column_name
        self.index_column_name = index_column_name
        self.repeats = repeats
        self.epochs = epochs
        self.split_rate = split_rate
        self.models = self.form_models()

    @staticmethod
    def shuffle_dataset(dataset):
        print("Shuffling Data...")
        shuffled_dataset = shuffle(dataset)
        return shuffled_dataset

    def get_train_test(self, dataset):
        X = dataset.drop([self.label_column_name, self.index_column_name], axis="columns")
        y = dataset[self.label_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/self.split_rate), random_state=7)
        # print(X_train.shape, y_train.shape)
        self.preprocessing_pipe.fit(X)
        X_train, X_test = self.preprocessing_pipe.transform(X_train), self.preprocessing_pipe.transform(X_test)
        print("Transforming Data...")
        # print(X_train.shape, y_train.shape)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def form_models():
        models = {}
        print("Forming Models...")
        models["LR"] = LogisticRegression(n_jobs=-1)
        # models["PasAgr"] = PassiveAggressiveClassifier(n_jobs=-1) No predict_proba
        models["KNN"] = KNeighborsClassifier(n_jobs=-1)
        models["SVM"] = SVC()
        models["DCT"] = DecisionTreeClassifier()
        models["RFC"] = RandomForestClassifier(n_jobs=-1)
        models["ABC"] = AdaBoostClassifier()
        models["GBC"] = GradientBoostingClassifier()
        models["GNB"] = GaussianNB()
        models["MNB"] = MultinomialNB()

        return models

    def get_trained_models(self, X_train, y_train):
        fitted_models = {}
        print("Starting to train models...")
        for idx, (model_name, model) in enumerate(self.models.items()):
            time1 = time.time()
            model.fit(X_train.toarray(), y_train)
            fitted_models[model_name] = model
            time2 = time.time()
            print("Successfully trained ", model_name, "in ", ((time2-time1)*1000.0), "ms only ",
                  (len(self.models) - idx), " more to go!")
        return fitted_models

    def get_predict(self, X_test, fitted_models: dict):
        preds = {}
        for model_name, model in fitted_models.items():
            predict = model.predict(X_test)
            predict_proba = model.predict_proba(X_test)
            mask = np.max(predict_proba, axis=1) > self.threshold
            preds[model_name] = (predict, mask)
        return preds

    def repeat(self):
        dataset = self.dataset
        results = []
        for i in range(self.repeats):
            shuffled_dataset = self.shuffle_dataset(dataset)
            X_train, X_test, y_train, y_test = self.get_train_test(shuffled_dataset)
            fitted_models = self.get_trained_models(X_train, y_train)
            predictions = self.get_predict(X_test, fitted_models)
            results.append(predictions)
        for prediction in results:
            print(prediction)


if __name__ == "__main__":
    from CombineDatasets import get_combined_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer

    df = get_combined_dataset()
    df["index"] = df.index
    df = df[["index", "content", "label"]]


    pipe = ColumnTransformer([
        ("vec", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                      lowercase=False, smooth_idf=False, analyzer="char", use_idf=True,
                                      sublinear_tf=True, norm="l2", binary=True), "content")
    ], remainder="passthrough")
    label_corrector = CorrectLabels(df, "label", "index", 10, 0.90, pipe, repeats=1)
    label_corrector.repeat()


