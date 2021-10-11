import pandas as pd


def get_combined_dataset():
    Mickes = pd.read_csv("data/MickesClubhouseCOVID19RumourDataset/MickesClubhouseCOVID19RumourDataset.csv", sep=",")
    Constraint_Test = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Test.csv", sep=",")
    Constraint_Train = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Train.csv", sep=",")
    Constraint_Val = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Val.csv", sep=",")
    Constraint = pd.concat([Constraint_Test, Constraint_Train, Constraint_Val], ignore_index=True, sort=False)
    Constraint = Constraint.drop(columns="id")
    Constraint["label"] = Constraint["label"].map({"real": 1, "fake": 0})
    Constraint = Constraint.rename({"tweet": "content"}, axis=1)
    Mickes = Mickes[Mickes.label != "U"]
    Mickes["label"] = Mickes["label"].map({"T": 1, "F": 0})
    Combined = pd.concat([Mickes, Constraint], ignore_index=True, sort=False)
    return Combined


if __name__ == "__main__":
    print(get_combined_dataset())
