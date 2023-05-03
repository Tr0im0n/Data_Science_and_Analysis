
import os
import pandas as pd
from matplotlib import pyplot as plt

os.chdir(r"../data")
file_name = "housing_example.csv"
df = pd.read_csv(file_name, delimiter=',')
column_names = tuple(df.columns)


def plot1():
    for column in column_names:
        plt.plot(df[column], label=column)

    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()


def get_nan_count():
    return df.isna().sum().sum()


def summary():
    means = tuple(df[column].mean() for column in column_names)
    medians = tuple(df[column].median() for column in column_names)
    stds = tuple(df[column].std() for column in column_names)
    for i, j, k in zip(means, medians, stds):
        print(i, j, k)

