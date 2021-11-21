from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from gensim.sklearn_api import W2VTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class TweetTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tweet_text_transformer = Pipeline(steps=[
            ('count_vectoriser', CountVectorizer()),
            ('tfidf', TfidfTransformer())])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.tweet_text_transformer.fit_transform(X.squeeze()).toarray()


def get_feature_union(df):
    text_vect_pipe = ColumnTransformer([
        ("Word2Vec", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                lowercase=False, smooth_idf=False, analyzer="char", use_idf=True,
                                sublinear_tf=True, norm="l2", binary=True), ["content", "lemmatized", "removed_names",
                                        "converted_emojis", "removed_rare_words"]),
    ])

    num_pipe = ColumnTransformer([
        ("Scaler", RobustScaler(), ["likeCount", "quoteCount", "retweetCount", "replyCount", "nHashtags",
                                    "nMentionedUsers", "opFollowerCount", "opFollowingCount", "opPostedTweetsCount",
                                    "opFavouritesCount", "opListedCount", "polarity", "subjectivity", "char_count",
                                    "verb_count", "noun_count", "adj_count", "word_count", "sent_count", "avg_word_len",
                                    "avg_sent_len", "num_rare_words", "num_lemmatized_words", "num_rare_words"])
    ])

    tag_pipe = ColumnTransformer([
        ("Scaler", RobustScaler(), [k for k in df.columns if 'tags_' in k])
    ])

    hot_encoded_pipe = ColumnTransformer([
        ("OneHotEncoder", OneHotEncoder(), ["source", "hashtags", "shortened_outlinks"])
    ])

    pipe = FeatureUnion([
        ("text_features", text_vect_pipe),
        ("numerical_features", num_pipe),
        ("tag_features", tag_pipe),
        ("hot_encoded_features", hot_encoded_pipe)
    ])
    return pipe
