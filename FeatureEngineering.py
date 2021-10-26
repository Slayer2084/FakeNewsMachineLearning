import pandas as pd
import ScrapeTweets


def get_features(df):
    df["tweet_object"] = df["content"].apply(lambda text: ScrapeTweets.get_tweet_object(text))
    df["time"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.date)
    df["likeCount"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.likeCount)
    df["quoteCount"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.quoteCount)
    df["source"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.sourceLabel)
    df["retweetCount"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.retweetCount)
    df["replyCount"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.replyCount)
    df["username"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.username)
    df["op_object"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.user)
    df["hashtags"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.hashtags)
    df["nHashtags"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else len(tweet.hashtags))
    df["mentionedUsers"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.mentionedUsers)
    df["nMentionedUsers"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else len(tweet.mentionedUsers))
    df["outlinks"] = df["tweet_object"].apply(lambda tweet: None if tweet == None else tweet.outlinks)
    df["opVerified"] = df["op_object"].apply(lambda user: None if tweet == None else user.verified)
    df["opCreated"] = df["op_object"].apply(lambda user: None if tweet == None else user.created)
    df["opFollowerCount"] = df["op_object"].apply(lambda user: None if tweet == None else user.followersCount)
    df["opFriendCount"] = df["op_object"].apply(lambda user: None if tweet == None else user.friendsCount)
    df["opStatusesCount"] = df["op_object"].apply(lambda user: None if tweet == None else user.statusesCount)
    df["opFavouritesCount"] = df["op_object"].apply(lambda user: None if tweet == None else user.favouritesCount)
    df["opListedCount"] = df["op_object"].apply(lambda user: None if tweet == None else user.listedCount)
    df["opProtected"] = df["op_object"].apply(lambda user: None if tweet == None else user.protected)



    return df


if __name__ == "__main__":
    print("test")
    from CombineDatasets import get_combined_dataset
    df = get_combined_dataset().head(10)
    print(get_features(df))
