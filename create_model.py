from CombineDatasets import get_combined_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from FeatureEngineering import get_features
from PreprocessPipe import get_feature_union
from BadLabel import CorrectLabels


df = get_combined_dataset()
df["index"] = df.index

df_with_features = get_features(df.head(25))
feature_union = get_feature_union(df_with_features)
labelCorrector = CorrectLabels(df_with_features, "label", 5, 0.9, feature_union)
df = labelCorrector.clean_up_bad_labels()


model_list = {}

for model in model_list:
    model = Pipeline([
        ("features", feature_union),
        ("classifier", model)
    ])

