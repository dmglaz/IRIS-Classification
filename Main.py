import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, pairwise_distances
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pandas.tools.plotting import scatter_matrix
from sklearn.datasets import load_iris
filterwarnings('ignore')

from Classification_Algorithms import *
from Clustering_Algorithms import *

import nltk;nltk.download()
def get_iris_df():
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=[x[:-5] for x in iris.feature_names])
    iris_df = iris_df.join(pd.Series(iris.target, name='Type'))
    categories = dict(zip([0, 1, 2], iris.target_names))
    iris_df['Type'] = iris_df['Type'].apply(lambda x: categories[x])
    train, test = train_test_split(iris_df, test_size=0.3)
    return {"all":iris_df, "train": train, "test": test}
def scatter_plot_iris_df(df):
    iris_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = df['Type'].apply(lambda x: iris_colors[x])
    scatter = scatter_matrix(df, figsize=(15, 10), c=colors, s=150, alpha=1)

def report(y_true, y_pred, classes):
    print(classification_report(y_true=y_true,
                                y_pred=y_pred))

    confsn_mtrx=pd.DataFrame(data=confusion_matrix(y_true=y_true, y_pred=y_pred),
                                 index=classes,
                                 columns=classes
                                 )
    print(confsn_mtrx)

iris_df = get_iris_df()
col_names = list(iris_df["train"].drop("Type", axis=1).columns)


# scatter_plot_iris_df(iris_df)
"""
  ---  Classifications ---
"""
dcn_tree_ans = decision_tree_clssfr(iris_df["train"], iris_df["test"])
log_rgrsn_ans = log_regrsn_clssfr(iris_df["train"], iris_df["test"])
SVM_ans = SVM_clssfr(iris_df["train"], iris_df["test"])
KNN_ans = KNN_clssfr(iris_df["train"], iris_df["test"],3)
rnd_frst_ans = rendom_forest_clssfr(iris_df["train"], iris_df["test"],3)
voting_clsfr(iris_df["train"], iris_df["test"])
baggin_clsfr(iris_df["train"], iris_df["test"])
adaBoost_clsf(iris_df["train"], iris_df["test"])
gadient_boosting(iris_df["train"], iris_df["test"])
grid_search(iris_df["train"], iris_df["test"])
clas_ans = rnd_frst_ans

report(clas_ans["y_true"]["test"],
       clas_ans["y_pred"]["test"],
       clas_ans["classes"])

"""
  ---  Clustering ---
"""
KMeans_ans = Kmeans_clssfr(iris_df["all"],3)
aglo_ans = Aglo_cluster_clssfr(iris_df["all"],3)
clust_ans = KMeans_ans

flwr_to_clst_dict ={
                "setosa":0,
                "versicolor":1,
                "virginica":2
                    }
clst_to_flwr__dict ={0:"virginica",
                     1: "setosa",
                     2: "versicolor"
                     }

confusion_matrix(y_true = clust_ans["y_true"],
                 y_pred = [clst_to_flwr__dict[x] for x in clust_ans["y_pred"]]
                 )

print(classification_report(y_true = KMeans_ans["y_true"]["train"].apply(lambda x: flower_to_cluster_dict[str(x)]),
                            y_pred = KMeans_ans["y_pred"]["train"]
                            ))


