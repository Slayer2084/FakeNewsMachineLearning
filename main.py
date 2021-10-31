from CombineDatasets import get_combined_dataset
from sklearn.model_selection import train_test_split
from FeatureEngineering import get_features





df = get_combined_dataset()
df_with_features = get_features(df.head(50))
print(df_with_features["shortened_outlinks"])
