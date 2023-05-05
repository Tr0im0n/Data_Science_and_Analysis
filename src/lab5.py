
import numpy as np
from matplotlib import pyplot as plt

# Tools ###############################################################################


def make_cov_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.dot(matrix, np.transpose(matrix)) / (len(matrix[0])-1)


def center_matrix(matrix: np.ndarray) -> np.ndarray:
    for row in matrix:
        row -= np.mean(row)
    return matrix


def sort_val_vec(values: np.ndarray, matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    order = np.argsort(values)[::-1]
    return [values[i] for i in order], [matrix[i] for i in order]


def pca(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    centered_matrix = center_matrix(matrix)
    cov = make_cov_matrix(centered_matrix)
    e_vals, e_vecs = np.linalg.eig(cov)
    e_vecs = np.transpose(e_vecs)
    return sort_val_vec(e_vals, e_vecs)

# Exercises ################################################################################


def eigen():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    w, v = np.linalg.eig(a)
    d = np.diag(w)
    print(c := np.dot(a, v))
    print(e := np.dot(v, d))
    print(np.array_equal(c, e))


def pca1():
    xs = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
                   [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]])

    e_vals, e_vecs = pca(xs)

    for val, vec in zip(e_vals, e_vecs):
        plt.plot([-val*vec[0], val*vec[0]], [-val*vec[1], val*vec[1]], label=val)
    plt.gca().set_aspect(1)
    plt.scatter(xs[0], xs[1], marker='x')
    plt.legend()
    plt.grid(True)
    plt.show()


# eigen()
pca1()

