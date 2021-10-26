# YouTube Video: https://www.youtube.com/watch?v=rhBZqEWsZU4
import numpy as np
import pandas as pd
import tweepy.error
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import requests


import twitter_credentials


# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        super().__init__()
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    def tweets_to_data_frame(self, tweets):
        df_from_tweets = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

        df_from_tweets['id'] = np.array([tweet.id for tweet in tweets])
        df_from_tweets['language'] = np.array([tweet.lang for tweet in tweets])
        df_from_tweets['length'] = np.array([len(tweet.text) for tweet in tweets])
        df_from_tweets['geo'] = np.array([tweet.geo for tweet in tweets])

        return df_from_tweets

    def part_of_tsv_to_data_frame(self, tsv_filename):
        part_of_df = pd.read_csv(tsv_filename, sep='\t', nrows=1000000)
        return part_of_df

    def tsv_to_data_frame(self, tsv_filename):
        chunk_list = []
        df_chunk = pd.read_csv(tsv_filename, sep='\t', chunksize=1000000)
        for chunk in df_chunk:
            chunk = self.clean_data_frame(chunk)
            chunk_list.append(chunk)
        df_non_clean = pd.concat(chunk_list)
        df_non_clean = df_non_clean.to_frame()
        return df_non_clean

    def clean_data_frame(self, df_to_clean):
        cleaned_df = df_to_clean.drop(df_to_clean[df_to_clean.lang != "de"].index)
        cleaned_df = cleaned_df['tweet_id']
        return cleaned_df

    def df_to_csv(self, df_to_csv, csv_filename):
        df_to_csv.to_csv(path_or_buf=csv_filename, sep=';', index=False)

    def tweet_id_to_full_info_df(self, df_to_convert):
        full_df = pd.DataFrame(
            columns=['created_at', 'id', 'full_text', 'entities', 'data', 'retweet_count', 'favorite_count'])
        banned_user_count = 0
        unavailable_tweets_count = 0
        created_at = []
        id = []
        full_text = []
        entities = []
        source = []
        retweet_count = []
        favorite_count = []
        for tweet_id in df_to_convert['tweet_id']:
            try:
                data = api.get_status(tweet_id, tweet_mode='extended')._json
                # data.pop('created_at')
                # data.pop('id')
                # data.pop('id_str')
                # data.pop('text')
                # data.pop('truncated')
                # data.pop('entities')
                # data.pop('data')
                # data.pop('extended_entities')
                # data.pop('in_reply_to_status_id')
                # data.pop('in_reply_to_status_id_str')
                # data.pop('in_reply_to_user_id')
                # data.pop('in_reply_to_user_id_str')
                # data.pop('in_reply_to_screen_name')
                # data.pop('user')
                # data.pop('geo')
                # data.pop('coordinates')
                # data.pop('place')
                # data.pop('contributors')
                # data.pop('is_quote_status')
                # data.pop('retweet_count')
                # data.pop('favorite_count')
                # data.pop('favorited')
                # data.pop('retweeted')
                # data.pop('possibly_sensitive')
                # data.pop('possibly_sensitive_appealable')
                # data.pop('lang')
                created_at.append(data.get('created_at'))
                id.append(data.get('id'))
                full_text.append(data.get('full_text'))
                entities.append(data.get('entities'))
                source.append(data.get('data'))
                retweet_count.append(data.get('retweet_count'))
                favorite_count.append(data.get('retweet_count'))
            except tweepy.TweepError as e:
                if e.api_code == 63:
                    print("Fehlercode:", e.api_code, " User wurde gebannt.")
                    banned_user_count += banned_user_count
                if e.api_code == 144:
                    print("Fehlercode:", e.api_code, " Tweet nicht mehr verfügbar.")
                    unavailable_tweets_count += unavailable_tweets_count

        created_at = np.array(created_at)
        id = np.array(id)
        full_text = np.array(full_text)
        entities = np.array(entities)
        source = np.array(source)
        retweet_count = np.array(retweet_count)
        favorite_count = np.array(favorite_count)
        full_df['created_at'] = created_at
        full_df['id'] = id
        full_df['full_text'] = full_text
        full_df['entities'] = entities
        full_df['data'] = source
        full_df['retweet_count'] = retweet_count
        full_df['favorite_count'] = favorite_count
        print(unavailable_tweets_count)
        print(banned_user_count)
        return full_df

    def join_csvs(self, csv_filenames, result_file_name):
        df_for_csvs = pd.concat([pd.read_csv(f, sep=',', index_col="id") for f in csv_filenames], ignore_index=True)
        print(df_for_csvs)
        df_for_csvs.to_csv(path_or_buf=result_file_name, sep=";")

    def unshorten_url(self, url):
        session = requests.Session()
        resp = session.head(url, allow_redirects=True)
        return resp.url


class Claim_Buster():
    api_key = "34f38df230ff4affb5ebc0b989dd5e9b"
    def get_checkability_score(self, statement):
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{statement}"
        request_headers = {"x-api-key": self.api_key}

        # Send the GET request to the API and store the api response
        api_response = requests.post(url=api_endpoint, headers=request_headers)

        # Print out the JSON payload the API sent back
        return api_response.json()



if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    claim_buster = Claim_Buster()
    api = twitter_client.get_twitter_client_api()
    tweet = api.get_status(1330896290029441026)
    # tweets = api.user_timeline(screen_name="JoeBiden", count=120)
    # print(dir(tweets[0]))
    # df = tweet_analyzer.tweets_to_data_frame(tweets)
    # df = tweet_analyzer.tsv_to_data_frame("full_dataset_clean.tsv")
    # print(df)
    # print(type(df))
    # tweet_analyzer.df_to_csv(df, "clean_dataset.csv")
    # print(df.size / 5)
    # print(tweet._json)
    # df = pd.read_csv("clean_dataset.csv")
    # print(df)
    # print(type(df))
    # df = tweet_analyzer.tweet_id_to_full_info_df(df.head(1000))
    # print(df)
    # print(type(df))
    # df.to_csv("dataset_expended.csv", sep=";")
    # tweet_analyzer.join_csvs(["Constraint_Train.csv", "Constraint_Val.csv", "Constraint_Test.csv"], "Fake_News_Dataset.csv")
    # print(claim_buster.get_checkability_score("Your mom stinks"))


