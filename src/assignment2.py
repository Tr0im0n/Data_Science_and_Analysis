
import os
import random

import numpy as np
from matplotlib import pyplot as plt

"""
README: 
You might have to change the directory stuff in the import data section 

Final function:
line 166

Steps: 
1. Split data in training data and test data
2. Get centers and clusters from function made in previous assignment
3. Convert data format
4. Construct spatial map ( size = 0.3 )
for each point
5. look through the surrounding chunks for points 
    get the distance to the point and its label and add that to the list
5b. If not k points in the list repeat but with a bigger search radius
6. Count the votes of k closest points(labels), add the point and label to the answer list

Evaluation:
Visual proof
Each color is a cluster
Stars are the training set
Plus-signs are the testing set

Returns:
Points for which search radius had to be expanded
The results
A scatter plot

"""

# import data ##########################################################################
os.chdir(r"..\data")
file_name = "StarTypeDataset.csv"
my_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)


# Tool functions ##########################################################################

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
    # build clusters
    for i, point in enumerate(my_data):
        distances = tuple(distance2(point, center) for center in centers)
        group = np.argmin(distances)
        """
        if group > n_centers:
            print(group)
            group = group // n_centers
        """
        clusters[group].append(i)
    # get new centers from clusters
    ans = np.zeros_like(centers)
    for n, cluster in enumerate(clusters):
        sumi = np.array([0.0, 0.0])
        for i in cluster:
            sumi += my_data[i]
        ans[n] = sumi/(len(cluster)-1)
    # return
    if return_clusters:
        return ans, clusters
    else:
        return ans


def clusters_to_labeled_data(clusters, data=my_data):
    ans = []
    for i, cluster in enumerate(clusters):
        for index in cluster:
            ans.append([*data[index], i])
    return ans


def min_and_max(data):
    x_min = min(data[:, 0])
    x_max = max(data[:, 0])
    y_min = min(data[:, 1])
    y_max = max(data[:, 1])
    return x_min, x_max, y_min, y_max


def get_chunk_coord(x, y, chunk_size: float = 0.3):
    return -int((x - 1) // chunk_size), -int((y - 1) // chunk_size)


def construct_spatial_map(data, chunk_size: float = 0.3):
    ans = {}
    for x, y, label in data:
        chunk_coord = get_chunk_coord(x, y, chunk_size)
        if chunk_coord not in ans:
            ans[chunk_coord] = [(x, y, label)]
        else:
            ans[chunk_coord].append((x, y, label))
    return ans


def split_data(data, fraction: float = 0.8):
    train = []
    test = []
    for i in data:
        if random.random() < fraction:
            train.append(i)
        else:
            test.append(i)
    return np.array(train), np.array(test)


def get_surrounding_chunks(cx, cy, spatial_map, search_radius: int = 1):
    return [(x, y) for x in range(cx - search_radius, cx + search_radius + 1)
                   for y in range(cy - search_radius, cy + search_radius + 1)
                   if (x, y) in spatial_map]


def eval_point(test_point, spatial_map, k: int, search_radius: int = 1):
    # look through the surrounding chunks for points
    # get the distance to the point and its label and add that to the list
    chunk_coord = get_chunk_coord(test_point[0], test_point[1])
    surrounding_chunks = get_surrounding_chunks(chunk_coord[0], chunk_coord[1], spatial_map, search_radius)
    distance_list = []
    for chunk_key in surrounding_chunks:
        for train_x, train_y, label in spatial_map[chunk_key]:
            distance_list.append([distance2(test_point, (train_x, train_y)), label])
    # check if there are k points in this area
    counter = 0
    for distance, label in distance_list:
        if distance >= 0.09:
            counter += 1
    if counter < k:
        print(counter, test_point)
        # recursively try again but with a bigger search radius
        return eval_point(test_point, spatial_map, k, search_radius+1)

    # determine which label is correct
    label_count_dict = {}  # could have been any other type
    max_count = 0
    max_label = -1

    def my_key(ele):
        return ele[0]

    distance_list.sort(key=my_key)  # could lambda be used here?

    label_list = [label for distance, label in distance_list[:k+1]]

    for label in label_list[:k+1]:
        if label not in label_count_dict:
            count = label_list.count(label)
            label_count_dict[label] = count
            if count > max_count:
                max_count = count
                max_label = label

    return max_label


# Final functions ##########################################################################

def k_means(data: np.ndarray, k=5, clusters: bool = False):
    bounds = tuple((min(data[:, i]), max(data[:, i])) for i in range(data.shape[1]))
    centers = initial_centers(k, bounds)

    for _ in range(7):
        centers = make_new_centers(centers)

    return make_new_centers(centers, clusters)


def knn(data, k: int = 6):
    # split data
    train_data, test_data = split_data(data)
    # get the cluster from previously implemented k-means
    my_centers, my_clusters = k_means(train_data, 5, True)
    # change format of data
    labeled_data = clusters_to_labeled_data(my_clusters)
    # construct spatial map
    spatial_map = construct_spatial_map(labeled_data)
    # plot training data
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for values in spatial_map.values():
        for x, y, color_i in values:
            plt.scatter(x, y, s=60, c=colors[color_i], marker='*')
    # test
    ans = []
    for test_point in test_data:
        ans.append([*test_point, eval_point(test_point, spatial_map, k, 1)])
    # plot training data
    for x, y, color_i in ans:
        plt.scatter(x, y, s=240, c=colors[color_i], marker='+')
    plt.show()
    # return not really necessary
    return ans


print("Answers: ")
print(*knn(my_data), sep="\n")
