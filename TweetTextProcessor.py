from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class TweetTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tweet_text_transformer = Pipeline(steps=[
            ('count_vectoriser', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer())])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.tweet_text_transformer.fit_transform(X.squeeze()).toarray()


class DataFrameColumnExtracter(TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]
