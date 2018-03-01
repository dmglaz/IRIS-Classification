import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, pairwise_distances
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
def voting_clsfr(df_train, df_test, voting_method="hard"):
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    classifiers = [('Log_Reg', LogisticRegression()),
                   ('Dec_Tree', DecisionTreeClassifier()),
                   ('SVM', SVC()),
                   ("KNN", KNeighborsClassifier())
                   ]
    clsf_voting = VotingClassifier(estimators=classifiers, voting=voting_method)
    clsf_voting.fit(X["train"], y["train"])
    results = pd.DataFrame(index=["train","test"])


    for clf_name, clf in classifiers:
        clf.fit(X["train"], y["train"])
        results[clf_name] = [clf.score(X["train"], y["train"]), clf.score(X["test"], y["test"])]
    results["Voting"] = [clsf_voting.score(X["train"], y["train"]), clsf_voting.score(X["test"], y["test"])]
    print(results)
def baggin_clsfr(df_train, df_test):
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }

    clf_base_dec_tree = DecisionTreeClassifier()
    clf_bagging_dec_tree = BaggingClassifier(base_estimator=clf_base_dec_tree, n_estimators=100)
    clf_bagging_dec_tree.fit(X["train"], y["train"])

    clf_base_log_reg = LogisticRegression()
    clf_bagging_log_reg = BaggingClassifier(base_estimator=clf_base_log_reg, n_estimators=100)
    clf_bagging_log_reg.fit(X["train"], y["train"])

    tree_clsf = DecisionTreeClassifier()
    tree_clsf.fit(X["train"], y["train"])

    log_reg_clsf = LogisticRegression()
    log_reg_clsf.fit(X["train"], y["train"])

    results = pd.DataFrame(index=["train","test"])
    results["Decision Tree"] = [tree_clsf.score(X["train"], y["train"]), tree_clsf.score(X["test"], y["test"])]
    results["Bagging Decision Tree"] = [clf_bagging_dec_tree.score(X["train"], y["train"]), clf_bagging_dec_tree.score(X["test"], y["test"])]
    results["Log Reg"] = [log_reg_clsf.score(X["train"], y["train"]), log_reg_clsf.score(X["test"], y["test"])]
    results["Bagging Log Reg"] = [clf_bagging_log_reg.score(X["train"], y["train"]), clf_bagging_log_reg.score(X["test"], y["test"])]

    print(results)
def adaBoost_clsf(df_train, df_test):
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }

    clf_base = DecisionTreeClassifier()
    clf_adaboost_tree = AdaBoostClassifier(base_estimator=clf_base,
                                      n_estimators=200,
                                      learning_rate=0.01)
    clf_adaboost_tree.fit(X["train"], y["train"])

    clf_base_log_reg = LogisticRegression()
    clf_adaboost_log_reg = AdaBoostClassifier(base_estimator=clf_base_log_reg,
                                               n_estimators=200,
                                               learning_rate=0.01)
    clf_adaboost_log_reg.fit(X["train"], y["train"])

    tree_clsf = DecisionTreeClassifier()
    tree_clsf.fit(X["train"], y["train"])

    log_reg_clsf = LogisticRegression()
    log_reg_clsf.fit(X["train"], y["train"])

    results = pd.DataFrame(index=["train", "test"])
    results["Decision Tree"] = [tree_clsf.score(X["train"], y["train"]), tree_clsf.score(X["test"], y["test"])]
    results["AdaBoost Decision Tree"] = [clf_adaboost_tree.score(X["train"], y["train"]),clf_adaboost_tree.score(X["test"], y["test"])]
    results["Log Reg"] = [log_reg_clsf.score(X["train"], y["train"]), log_reg_clsf.score(X["test"], y["test"])]
    results["AdaBoost Log Reg"] = [clf_adaboost_log_reg.score(X["train"], y["train"]),clf_adaboost_log_reg.score(X["test"], y["test"])]

    print(results)
def gadient_boosting(df_train, df_test):
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    clf_GB = GradientBoostingClassifier(max_depth=3,
                                        n_estimators=200,
                                        learning_rate=0.01)
    clf_GB.fit(X["train"], y["train"])
    print ("{:3} classifier:\n \
    \ttrain accuracy: {:.2f}\n \
    \ttest accuracy: {:.2f}" \
        .format('DT gradient boosting',
                clf_GB.score(X["train"], y["train"]),
                clf_GB.score(X["test"], y["test"])))
    k = 5
    scores = cross_val_score(clf_GB,X["train"], y["train"], cv=k )
    print ("Scores : " + (k * "{:.3f} ").format(*scores))
    print ("Average:", scores.mean())
def grid_search(df_train, df_test):
    clf = DecisionTreeClassifier()
    clf.get_params()
    X = {
        "train": df_train.drop("Type", axis=1),
        "test": df_test.drop("Type", axis=1)
    }
    y = {
        "train": df_train["Type"],
        "test": df_test["Type"]
    }
    clf_gs = GridSearchCV(clf, param_grid={'max_depth': [5,8,12,15]},cv=2)
    clf_gs.fit(X["train"], y["train"])
    print ("Best parameters:", clf_gs.best_params_)
    print ("Best score:", clf_gs.best_score_)