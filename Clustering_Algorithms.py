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

def Kmeans_clssfr(df,  n_clust = 3):
    estm = KMeans(n_clusters= n_clust)

    X =  df.drop("Type", axis=1)
    y = df["Type"]
    pred = estm.fit_predict(X)

    ret_val = {
        "y_true": y,
        "y_pred": pred,
    }
    return ret_val
def calc_inertia(k, scaler, data):
    norm_data = scaler().fit_transform(data)
    kmeans = KMeans(n_clusters=k, max_iter=10, random_state=1, init='random', n_init=10)
    kmeans.fit(data)
    return kmeans.inertia_
def calc_inertia_NO_SCALER(k, data):
    kmeans = KMeans(n_clusters=k, max_iter=10, random_state=1, init='random', n_init=10)
    kmeans.fit(data)
    return kmeans.inertia_
def plot_inertia(scaler, data):
    inertias = [(k, calc_inertia(k, scaler, data)) for k in range(1, 31)]
    plt.plot(*zip(*inertias))
    plt.title('Inertia vs. k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 31))
def plot_inertia_NO_SCALER(data):
    inertias = [(k,calc_inertia_NO_SCALER(k,  data.drop("Type", axis=1))) for k in range(1, 10)]
    plt.plot(*zip(*inertias))
    plt.title('Inertia vs. k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 10))
def Aglo_cluster_clssfr(df,n_clust = 3):
    X = df.drop("Type", axis=1)
    y = df["Type"]
    link=linkage(X, method="complete", metric = "euclidean")
    # dn = dendrogram(link)
    cluster = fcluster(link, t=n_clust, criterion="maxclust")
    # print(cluster)
    ret_val = {
        "y_true": y,
        "y_pred": cluster,
    }
    return ret_val
