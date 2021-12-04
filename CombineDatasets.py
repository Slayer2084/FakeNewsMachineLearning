import pandas
import pandas as pd


def get_combined_features_dataset():
    df = pandas.read_csv(filepath_or_buffer="data/CombinedWithFeatures.csv", sep=";")
    df = df.drop("index", axis="columns")
    return df


def get_combined_dataset():
    mickes = pd.read_csv("data/MickesClubhouseCOVID19RumourDataset/MickesClubhouseCOVID19RumourDataset.csv", sep=",")
    constraint_test = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Test.csv", sep=",")
    constraint_train = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Train.csv", sep=",")
    constraint_val = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Val.csv", sep=",")
    constraint = pd.concat([constraint_test, constraint_train, constraint_val], ignore_index=True, sort=False)
    constraint = constraint.drop(columns="id")
    constraint["label"] = constraint["label"].map({"real": 1, "fake": 0})
    constraint = constraint.rename({"tweet": "content"}, axis=1)
    mickes = mickes[mickes.label != "U"]
    mickes["label"] = mickes["label"].map({"T": 1, "F": 0})
    combined = pd.concat([mickes, constraint], ignore_index=True, sort=False)
    return combined


if __name__ == "__main__":
    print(get_combined_features_dataset())
