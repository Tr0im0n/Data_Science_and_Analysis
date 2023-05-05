
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tools #######################################################################


def sort_val_vec(values: np.ndarray, matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    order = np.argsort(values)[::-1]
    return [values[i] for i in order], [matrix[i] for i in order]


def pca(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    cov = df.cov()
    e_vals, e_vecs = np.linalg.eig(cov)
    e_vecs = np.transpose(e_vecs)
    return sort_val_vec(e_vals, e_vecs)


# Code ################################################################

os.chdir(r"../data")
wine_df = pd.read_csv("winequality-red.csv", delimiter=';')
# print(wine_df.describe())


def plot_all():
    for attribute in wine_df.columns:
        plt.plot(wine_df[attribute])
        plt.title(attribute)
        plt.xlabel("index")
        plt.ylabel(attribute)
        plt.show()
        plt.close()


def test1():
    vals, vecs = pca(wine_df)
    print(vecs[0]*1000)
    plt.plot(vecs[0]*1000)
    plt.show()
    for i, j in zip(vecs[0]*1000, wine_df.columns):
        print(j, i, sep=" & ")

test1()
