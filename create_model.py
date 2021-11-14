from CombineDatasets import get_combined_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from FeatureEngineering import get_features
from PreprocessPipe import get_feature_union


df = get_combined_dataset()
df_with_features = get_features(df)
print(df_with_features)

feature_union = get_feature_union(df_with_features)

model_list = {}

for model in model_list:
    model = Pipeline([
        ("features", feature_union),
        ("classifier", model)
    ])

