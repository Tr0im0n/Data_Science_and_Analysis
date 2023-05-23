import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Readme ##############################################################################
# You might need to change the filestructure in "import data"
# def k_means() calculates the k_means of the given dataset
# def anim() shows an animation of this proces
# Steps:
# 1. Find some initial centers: can be random, the way I implemented it
#    is by picking evenly spaced points on a circle in the centre of the dataset
# 2. Divide the data into clusters: calculate the distance between a point and
#    all the centers and add the point to the cluster belonging to the nearest center
# 3. Calculate the new center: average of all the points in a cluster

# Function calls are on line 121 and 122

# import data ##########################################################################
os.chdir(r"..\data")
file_name = "StarTypeDataset.csv"
my_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)


# Tool functions ##########################################################################

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


# Final functions ##########################################################################

def k_means(data: np.ndarray, k=5, clusters: bool = False):
    bounds = tuple((min(data[:, i]), max(data[:, i])) for i in range(data.shape[1]))
    centers = initial_centers(k, bounds)

    for _ in range(7):
        centers = make_new_centers(centers)

    return make_new_centers(centers, clusters)


def anim(data: np.ndarray, k=5):
    fig = plt.figure()
    # ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    ax = plt.axes()

    bounds = tuple((min(data[:, i]), max(data[:, i])) for i in range(data.shape[1]))
    centers = initial_centers(k, bounds)

    def init():
        plt.scatter(data[:, 0], data[:, 1], s=36, marker='*')
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
            plt.scatter(data[cluster, 0], data[cluster, 1], s=36, marker='*')

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


print(*tuple(f"Center: {i} \t Cluster indices: {j}" for i, j in zip(*k_means(my_data, 5, True))), sep="\n")
anim(my_data)


# Old #################################################################################################

def k_means_old(k=5):
    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
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
