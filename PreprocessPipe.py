from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from gensim.sklearn_api import W2VTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


def get_feature_union(df):
    text_pipe = ColumnTransformer([
        ("Word2Vec", TfidfVectorizer(), ["content", "lemmatized", "removed_names",
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
        ("text", text_pipe),
        ("num", num_pipe),
        ("tag", tag_pipe),
        ("hot_encoded", hot_encoded_pipe)
    ])

    return pipe
