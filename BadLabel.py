from CombineDatasets import get_combined_dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC


class CorrectLabels:

    def __init__(self,
                 dataset,
                 label_column_name: str,
                 repeats,
                 split_rate: int = 4,
                 ):
        self.dataset = dataset
        self.X = dataset.drop(label_column_name, axis="columns")
        self.y = dataset[label_column_name]
        self.repeats = repeats
        self.split_rate = split_rate
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(self.X, self.y)
        self.models = self.form_models()

    @staticmethod
    def shuffled_dataset(dataset):
        shuffled_dataset = shuffle(dataset)
        return shuffled_dataset

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/self.split_rate), random_state=7)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def form_models(self):
        models = {}
        models["LR"] = LogisticRegression()
        models["PasAgr"] = PassiveAggressiveClassifier()
        models["KNN"] = KNeighborsClassifier()
        models["SVM"] = SVC()
        models["DCT"] = DecisionTreeClassifier()
        models["RFC"] = RandomForestClassifier()
        models["ABC"] = AdaBoostClassifier()
        models["GBC"] = GradientBoostingClassifier()
        models["GNB"] = GaussianNB()
        models["MNB"] = MultinomialNB()

        return models

    def train_models(self):
        fitted_models = []
        for model in self.models:
            model.fit(self.X_train)
            fitted_models.append(model)


