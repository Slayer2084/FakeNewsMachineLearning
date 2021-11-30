from sklearn.compose import ColumnTransformer
from gensim.sklearn_api import W2VTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_pandas import DataFrameMapper

text_column = ["content", "lemmatized", "removed_names", "converted_emojis", "removed_rare_words"]

num_column = ["likeCount", "quoteCount", "retweetCount", "replyCount", "nHashtags",
              "nMentionedUsers", "opFollowerCount", "opFollowingCount", "opPostedTweetsCount",
              "opFavouritesCount", "opListedCount", "polarity", "subjectivity", "char_count",
              "verb_count", "noun_count", "adj_count", "word_count", "sent_count", "avg_word_len",
              "avg_sent_len", "num_rare_words", "num_lemmatized_words", "num_rare_words"]

hot_encoded_column = ["source"]  # ["source", "hashtags", "shortened_outlinks"]


def get_feature_union(df):
    tag_column = [k for k in df.columns if 'tags_' in k]

    get_text_features = FunctionTransformer(lambda x: x[text_column], validate=False)
    get_num_features = FunctionTransformer(lambda x: x[num_column], validate=False)
    get_tag_features = FunctionTransformer(lambda x: x[tag_column], validate=False)
    get_hot_encoded_features = FunctionTransformer(lambda x: x[hot_encoded_column], validate=False)

    feature_union = FeatureUnion([
        ("text_features", Pipeline([
            ("selector", get_text_features),
            ("Word2Vec", TfidfVectorizer())
        ])),
        ("num_features", Pipeline([
            ("selector", get_num_features),
            ("Scaler", RobustScaler())
        ])),
        ("tag_features", Pipeline([
            ("selector", get_tag_features),
            ("Scaler", RobustScaler())
        ])),
        ("hot_encoded_features", Pipeline([
            ("selector", get_hot_encoded_features),
            ("encoder", OneHotEncoder())
        ]))
    ])
    mapper = DataFrameMapper([
        ("content", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                    lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                    sublinear_tf=True, norm='l2', binary=True)),
        ("lemmatized", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                       lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                       sublinear_tf=True, norm='l2', binary=True)),
        ("removed_names", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                          lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                          sublinear_tf=True, norm='l2', binary=True)),
        ("converted_emojis", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                             lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                             sublinear_tf=True, norm='l2', binary=True)),
        ("removed_rare_words", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                               lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                               sublinear_tf=True, norm='l2', binary=True)),
        (num_column, RobustScaler()),
        (tag_column, RobustScaler()),
        (hot_encoded_column, OneHotEncoder())
    ])

    return mapper


if __name__ == "__main__":
    from CombineDatasets import get_combined_dataset
    from FeatureEngineering import get_features

    df_features = get_features(get_combined_dataset().sample(10))
    pipe = get_feature_union(df_features)
    X = df_features.drop("label", axis="columns")
    y = df_features["label"]
    print(X)
    print(y)
    transformed_X = pipe.fit_transform(X)
    print(transformed_X)
