from functools import partial
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from xgboost import XGBClassifier
from CombineDatasets import get_combined_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from FeatureEngineering import get_features
from PreprocessPipe import get_feature_union
from BadLabel import CorrectLabels
from sklearn.model_selection import cross_val_score

df = get_combined_dataset()
df_with_features = get_features(df.head(100))
feature_union = get_feature_union(df_with_features)
labelCorrector = CorrectLabels(df_with_features, "label", 5, 0.9, feature_union)
df = labelCorrector.clean_up_bad_labels()

# Model List
# PassiveAggressiveClassifier
'''PassiveAggressiveClassifier(loss=trial.suggest_categorical("pas_agr_loss", ["hinge", "squared_hinge"]),
                            validation_fraction=trial.suggest_float("pas_agr_validation_fraction", 0.1, 1),
                            C=trial.suggest_float("pas_agr_C", 0.1, 1),
                            max_iter=trial.suggest_int("pas_agr_max_iter", 500, 10000),
                            tol=pas_agr_tol,
                            n_jobs=-1)'''
# pas_agr_params
'''pas_agr_tol = trial.suggest_categorical("pas_agr_tol", ["float", None])
    if pas_agr_tol == "float":
        pas_agr_tol = trial.suggest_float("pas_agr_tol_float", 0.00001, 0.01)'''

# SGDClassifier
'''SGDClassifier(loss=trial.suggest_categorical("sgd_loss", ["hinge", "log", "squared_hinge",
                                                                            "perceptron", "squared_error",
                                                                            "epsilon_insensitive"]),
                                alpha=trial.suggest_float("sgd_alpha", 0.00001, 0.001),
                                power_t=trial.suggest_float("sgd_power_t", 0.1, 1),
                                penalty=sgd_penalty,
                                l1_ratio=sgd_l1_ratio,
                                max_iter=trial.suggest_int("sgd_max_iter", 500, 10000),
                                learning_rate=trial.suggest_categorical("sgd_learning_rate",
                                                                        ["constant", "optimal",
                                                                         "invscaling", "adaptive"]),
                                eta0=trial.suggest_float("sgd_eta0", 0, 10),
                                n_jobs=-1)'''
# SGD_params
'''sgd_l1_ratio = 0.15
    sgd_penalty = trial.suggest_categorical("sgd_penalty", ["l2", "l1", "elasticnet"])
    if sgd_penalty == "elasticnet":
        sgd_l1_ratio = trial.suggest_float("sgd_l1_ratio", 0, 1)'''

# RandomForestClassifier
'''RandomForestClassifier(n_estimators=trial.suggest_int("RFC_n_estimators", 10, 500),
                                         criterion=trial.suggest_categorical("RFC_criterion", ["gini", "entropy"]),
                                         min_samples_split=RFC_min_samples_split,
                                         min_samples_leaf=RFC_min_samples_leaf,
                                         max_features=trial.suggest_categorical("RFC_max_features", ["auto", "log2"]),
                                         n_jobs=-1)'''
# RFC_params
'''pas_agr_tol = trial.suggest_categorical("pas_agr_tol", ["float", None])
    if pas_agr_tol == "float":
        pas_agr_tol = trial.suggest_float("pas_agr_tol_float", 0.00001, 0.01)

    # RFC_params
    RFC_min_samples_split_int_or_float = trial.suggest_categorical("RFC_min_samples_split_int_or_float",
                                                                   ["int", "float"])
    if RFC_min_samples_split_int_or_float == "int":
        RFC_min_samples_split = trial.suggest_int("RFC_min_samples_split", 1, 5)
    else:
        RFC_min_samples_split = trial.suggest_float("RFC_min_samples_split", 2, 5)

    RFC_min_samples_leaf_int_or_float = trial.suggest_categorical("RFC_min_samples_leaf_int_or_float", ["int", "float"])
    if RFC_min_samples_leaf_int_or_float == "int":
        RFC_min_samples_leaf = trial.suggest_int("RFC_min_samples_leaf", 1, 5)
    else:
        RFC_min_samples_leaf = trial.suggest_float("RFC_min_samples_leaf", 2, 5)'''

# XGBClassifier
'''XGBClassifier(eta=trial.suggest_float("XGB_eta", 0, 1),
                                gamma=trial.suggest_int("XGB_gamma", 0, 100),
                                max_depth=trial.suggest_int("XGB_max_depth", 1, 100),
                                min_child_weight=trial.suggest_int("XGB_min_child_weight", 0, 10000),
                                max_delta_step=trial.suggest_int("XGB_max_delta_step", 0, 100),
                                subsample=trial.suggest_float("XGB_subsample", 0, 1),
                                sampling_method=trial.suggest_categorical("XGB_sampling_method",
                                                                          ["uniform", "gradient_based"]),
                                reg_lambda=trial.suggest_int("XGB_reg_lambda", 1, 10),
                                reg_alpha=trial.suggest_float("XGB_reg_alpha", 0.000001, 1),
                                n_jobs=-1,
                                use_label_encoder=False)'''


def optimize(trial, X, y):
    # pas_agr_params
    pas_agr_tol = trial.suggest_categorical("pas_agr_tol", ["float", None])
    if pas_agr_tol == "float":
        pas_agr_tol = trial.suggest_float("pas_agr_tol_float", 0.00001, 0.01)

    # Model
    model = PassiveAggressiveClassifier(loss=trial.suggest_categorical("pas_agr_loss", ["hinge", "squared_hinge"]),
                                        validation_fraction=trial.suggest_float("pas_agr_validation_fraction", 0.1, 1),
                                        C=trial.suggest_float("pas_agr_C", 0.1, 1),
                                        max_iter=trial.suggest_int("pas_agr_max_iter", 500, 10000),
                                        tol=pas_agr_tol,
                                        n_jobs=-1)

    model = Pipeline([
        ("features", feature_union),
        ("classifier", model)
    ])
    train_x, valid_x, train_y, valid_y = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    model.fit(train_x, train_y)

    return cross_val_score(model, valid_x, valid_y)


print(df)
X, y = df.drop("label", axis="columns"), df["label"]
optimization_function = partial(optimize, X=X, y=y)
study = optuna.create_study(direction="maximize")
study.optimize(optimization_function, n_trials=1000)
print(study.best_trial)
