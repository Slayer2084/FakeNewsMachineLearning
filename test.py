import sys
from hulearn.classification import FunctionClassifier
from hulearn.experimental.interactive import InteractiveCharts
import pandas
import pandas as pd
import numpy as np
import optuna
from functools import partial
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from hulearn.classification import FunctionClassifier
from hulearn.experimental import InteractiveCharts
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from TweetTextProcessor import TweetTextProcessor, DataFrameColumnExtracter
from TweetPreprocessing import Preprocessing
pandas.set_option("display.max_colwidth", None)
pandas.set_option("display.max_columns", None)
df = pd.read_csv("2Cleaned_Fake_News_Dataset.csv", index_col='index', sep=";").reset_index(drop=True)
preprocessor = Preprocessing(df)
# df = preprocessor.remove_names().remove_chars().lemmatize().convert_emoji().remove_spaces(chain=False)
df = preprocessor.remove_chars().convert_emoji().remove_rare_words().remove_spaces(chain=False)
preprocessor = Preprocessing(df)
preprocessor.get_report()
X, y = df.drop('label', axis="columns"), df["label"]



tweet_data = ColumnTransformer([
    ("transform", TfidfVectorizer(analyzer='char', binary=True, sublinear_tf=True, norm='l2', use_idf=True, smooth_idf=False, lowercase=False, strip_accents=None, ngram_range=[1, 3]), "tweet")
])

sentiment_data = ColumnTransformer([
    ("scaler", RobustScaler(), ["polarity", "subjectivity"])
])

feature_union = FeatureUnion([
    ("tweet_data", tweet_data),
    ("sentiment_data", sentiment_data)
])


model = Pipeline([
    ("feature_uníon", feature_union),
    ("clf", SGDClassifier(alpha=3.245124648050599e-05, power_t=0.5915927990148733, penalty='elasticnet', l1_ratio=0.1598540038297913, max_iter=4327, learning_rate='optimal', eta0=2.5863786772891575))
])

model.fit(X, y)

print(np.mean(cross_val_score(model, X, y, cv=10)))
print(cross_val_score(model, X, y, cv=10))

