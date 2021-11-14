from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from gensim.sklearn_api import W2VTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def get_feature_union(df):
    text_vect_pipe = ColumnTransformer([
        ("Word2Vec", W2VTransformer(), ["content", "lemmatized", "removed_names",
                                        "converted_emojis", "removed_rare_words"]),
    ])

    num_pipe = ColumnTransformer([
        ("Scaler", RobustScaler(), ["likeCount", "quoteCount", "retweetCount", "replyCount", "nHashtags",
                                    "nMentionedUsers", "opFollowerCount", "opFollowingCount", "opPostedTweetsCount",
                                    "opFavouritesCount", "opListedCount", "polarity", "subjectivity", "char_count",
                                    "verb_count", "noun_count", "adj_count", "word_count", "sent_count", "avg_word_len",
                                    "avg_sent_len", "num_rare_words", "num_lemmatized_words"])
    ])

    tag_pipe = ColumnTransformer([
        ("Scaler", RobustScaler(), [k for k in df.columns if 'tags_' in k])
    ])

    hot_encoded_pipe = ColumnTransformer([
        ("OneHotEncoder", OneHotEncoder, ["source", "hashtags", "shortened_outlinks"])
    ])

    pipe = FeatureUnion([
        ("text_features", text_vect_pipe),
        ("numerical_features", num_pipe),
        ("tag_features", tag_pipe),
        ("hot_encoded_features", hot_encoded_pipe)
    ])
    return pipe
