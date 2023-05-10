import csv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

os.chdir(r"..\data")
file_name = "StarTypeDataset.csv"
# star_df = pd.read_csv(file_name)

my_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
# my_transpose = np.transpose(my_data)
# temp_array = my_transpose[0]
# mag_array = my_transpose[0]


def simple_plot():
    plt.scatter(my_data[0, :], my_data[1, :])
    plt.show()


def initial_centers(k, bounds):
    means = tuple(b[1]+b[0]/2 for b in bounds)
    radii = tuple(0.3*abs(mean-b[0]) for mean, b in zip(means, bounds))
    return np.array([ [radii[0]*np.cos(i*np.pi), radii[1]*np.sin(i*np.pi)] for i in np.linspace(0, 2, k+1)[:k] ])


def distance2(p1, p2):
    return pow(p1, 2) + pow(p2, 2)


def new_centers(centers):
    clusters = [[] for _ in range(len(centers))]
    print(len(clusters))

    for i, point in enumerate(my_data):
        distances = tuple(distance2(point, center) for center in centers)
        print(len(distances))
        print(np.argmin(distances))
        clusters[np.argmin(distances)].append(i)

    ans = np.array([])
    for cluster in clusters:
        sumi = 0
        for i in cluster:
            sumi += my_data[i]
        np.append(ans, sumi/(len(cluster)-1))

    return ans


def k_means(k):
    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
    print(bounds)
    centers = initial_centers(k, bounds)

    plt.scatter(my_data[:, 0], my_data[:, 1])
    plt.scatter(*np.transpose(centers))

    for i in range(10):
        centers = new_centers(centers)
        plt.scatter(*np.transpose(centers))

    plt.show()

    return centers


print( k_means(5) )
