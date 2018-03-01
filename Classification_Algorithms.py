import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, pairwise_distances
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pandas.tools.plotting import scatter_matrix
from sklearn.datasets import load_iris


def decision_tree_clssfr(df_train, df_test, *args, **kwargs):
    estm = DecisionTreeClassifier()
    X = {
        "train": df_train.drop("Type", axis =1),
        "test" : df_test.drop("Type", axis =1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    estm.fit(X["train"], y["train"])

    classes = estm.classes_
    feat_importance = dict(zip(list(df_train.drop("Type", axis =1)),estm.feature_importances_))

    pred = dict()
    pred["train"] = estm.predict(X["train"])
    pred["test"] = estm.predict(X["test"])

    # print(estm.score(X["train"], y["train"]))

    pred_proba = dict()
    pred_proba["train"] = estm.predict_proba(X["train"])
    pred_proba["test"] = estm.predict_proba(X["test"])

    # print(pd.DataFrame(pred_proba["train"], columns=classes ))

    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "y_pred_proba": pred_proba,
        "classes": classes,
        "feature_importance": feat_importance
    }
    return ret_val
def log_regrsn_clssfr(df_train, df_test, *args, **kwargs):
    estm = LogisticRegression()
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    estm.fit(X["train"], y["train"])

    classes = estm.classes_

    pred = dict()
    pred["train"] = estm.predict(X["train"])
    pred["test"] = estm.predict(X["test"])

    pred_proba = dict()
    pred_proba["train"] = estm.predict_proba(X["train"])
    pred_proba["test"] = estm.predict_proba(X["test"])

    # print(pd.DataFrame(pred_proba["train"], columns=classes ))

    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "y_pred_proba": pred_proba,
        "classes": classes,
    }
    return ret_val
def SVM_clssfr(df_train, df_test, *args, **kwargs):
    estm = LinearSVC()
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    estm.fit(X["train"], y["train"])

    classes = estm.classes_

    pred = dict()
    pred["train"] = estm.predict(X["train"])
    pred["test"] = estm.predict(X["test"])


    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "classes": classes,
    }
    return ret_val
def KNN_clssfr(df_train, df_test,k_neigbors_arg=5, *args, **kwargs):
    estm = KNeighborsClassifier(n_neighbors=k_neigbors_arg)

    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    estm.fit(X["train"], y["train"])

    classes = estm.classes_

    pred = dict()
    pred["train"] = estm.predict(X["train"])
    pred["test"] = estm.predict(X["test"])

    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "classes": classes,
    }
    return ret_val
def rendom_forest_clssfr(df_train, df_test,k_neigbors_arg=5, *args, **kwargs):
    estm = RandomForestClassifier()
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    estm.fit(X["train"], y["train"])

    classes = estm.classes_
    feat_importance = dict(zip(list(df_train.drop("Type", axis=1)), estm.feature_importances_))

    pred = dict()
    pred["train"] = estm.predict(X["train"])
    pred["test"] = estm.predict(X["test"])

    # print(estm.score(X["train"], y["train"]))

    pred_proba = dict()
    pred_proba["train"] = estm.predict_proba(X["train"])
    pred_proba["test"] = estm.predict_proba(X["test"])

    # print(pd.DataFrame(pred_proba["train"], columns=classes ))

    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "y_pred_proba": pred_proba,
        "classes": classes,
        "feature_importance": feat_importance
    }
    return ret_val
