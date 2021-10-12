import pandas as pd
import ScrapeTweets

def get_features(df):
    df["tweet_object"] = df["content"].apply(lambda text: ScrapeTweets.get_tweet_object(text))
    df[""]
    return df