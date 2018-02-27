import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC
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
    return iris_df
def scatter_plot_iris_df(df):
    iris_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = df['Type'].apply(lambda x: iris_colors[x])
    scatter = scatter_matrix(df, figsize=(15, 10), c=colors, s=150, alpha=1)


iris_df = get_iris_df()
scatter_plot_iris_df(iris_df)

