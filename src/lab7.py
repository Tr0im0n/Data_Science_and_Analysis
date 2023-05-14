import csv
import os
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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
    return sum(tuple(pow(i - j, 2) for i, j in zip(p1, p2)))


def make_new_centers(centers, return_clusters=False):
    # build clusters
    n_centers = len(centers)
    clusters = [[] for _ in range(n_centers)]

    for i, point in enumerate(my_data):
        distances = tuple(distance2(point, center) for center in centers)
        group = np.argmin(distances)
        if group > n_centers:
            print(group)
            group = group // n_centers
        clusters[group].append(i)

    # get new centers from clusters
    ans = np.zeros_like(centers)
    for n, cluster in enumerate(clusters):
        sumi = np.array([0.0, 0.0])
        for i in cluster:
            sumi += my_data[i]
        ans[n] = sumi/(len(cluster)-1)

    if return_clusters:
        return ans, clusters

    return ans


def k_means(k=5):
    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
    # print(bounds)
    centers = initial_centers(k, bounds)

    plt.scatter(my_data[:, 0], my_data[:, 1])
    plt.scatter(*np.transpose(centers))

    for i in range(10):
        centers = make_new_centers(centers)
        plt.scatter(*centers.transpose())

    plt.show()

    centers, clusters = make_new_centers(centers, True)
    for cluster in clusters:
        plt.scatter(my_data[cluster, 0], my_data[cluster, 1])

    plt.scatter(*centers.transpose())
    plt.show()

    return centers


def anim():
    fig = plt.figure()
    # ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    ax = plt.axes()

    k = 5
    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
    centers = initial_centers(k, bounds)

    def init():
        plt.scatter(my_data[:, 0], my_data[:, 1], s=36, marker='*')
        plt.scatter(*centers.transpose(), s=48, c='k')
        return

    def animate(frame):
        ax.clear()
        ax.set_title(f"Frame: {frame}")

        nonlocal centers

        if not frame:
            centers = initial_centers(k, bounds)
            init()
            return

        centers, clusters = make_new_centers(centers, True)
        for cluster in clusters:
            plt.scatter(my_data[cluster, 0], my_data[cluster, 1], s=36, marker='*')

        plt.scatter(*centers.transpose(), s=48, c='k')
        return

    animation = FuncAnimation(fig,
                              func=animate,
                              frames=8,
                              init_func=init,
                              interval=500,
                              repeat=True)
    plt.show()
    plt.close()
    return


anim()
# print( k_means(5) )
