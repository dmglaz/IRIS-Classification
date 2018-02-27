import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from pandas.tools.plotting import scatter_matrix

from sklearn.datasets import load_iris

from warnings import filterwarnings

import csv, json
from os import getcwd

def get_iris_df():
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=[x[:-5] for x in iris.feature_names])
    iris_df = iris_df.join(pd.Series(iris.target, name='Type'))
    categories = dict(zip([0, 1, 2], iris.target_names))
    iris_df['Type'] = iris_df['Type'].apply(lambda x: categories[x])
    return train_test_split(iris_df, test_size=0.3)
def scatter_plot_iris_df(df):
    iris_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = df['Type'].apply(lambda x: iris_colors[x])
    scatter = scatter_matrix(df, figsize=(15, 10), c=colors, s=150, alpha=1)
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

    pred_proba = dict()
    pred_proba["train"] = estm.predict_proba(X["train"])
    pred_proba["test"] = estm.predict_proba(X["test"])

    ret_val = {
        "y_true": y,
        "y_pred": pred,
        "y_pred_proba": pred_proba,
        "classes": classes,
        "feature_importance": feat_importance
    }
    return ret_val

def print_confussion_mtrx(y_true, y_pred, classes):
        confsn_mtrx=pd.DataFrame(data=confusion_matrix(y_true=y_true, y_pred=y_pred),
                                 index=classes,
                                 columns=classes
                                 )
        print(confsn_mtrx)




iris_df_TRAIN, iris_df_TEST = get_iris_df()
col_names = list(iris_df_TRAIN.drop("Type", axis=1).columns)


# scatter_plot_iris_df(iris_df)
ans = decision_tree_clssfr(iris_df_TRAIN, iris_df_TEST)
print_confussion_mtrx(ans["y_true"]["train"],
                      ans["y_pred"]["train"],
                      ans["classes"])
print(classification_report(y_true=ans["y_true"]["test"], y_pred=ans["y_pred"]["test"]))
