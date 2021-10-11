from CombineDatasets import get_combined_dataset
from sklearn.model_selection import train_test_split
import ScrapeTweets


df = get_combined_dataset()
df["time"] = df["time"].apply(lambda text: ScrapeTweets.get_timestamp(text))
print(df["time"])