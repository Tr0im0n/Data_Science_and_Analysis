
import numpy as np
from matplotlib import pyplot as plt


def eigen():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    w, v = np.linalg.eig(a)
    d = np.diag(w)
    print(c := np.dot(a, v))
    print(e := np.dot(v, d))
    print(np.array_equal(c, e))


def pca1(display=True):
    xs = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
                   [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]])
    for row in xs:
        row -= np.mean(row)

    cov = np.dot(xs, np.transpose(xs)) / (len(xs[0])-1)
    w, v = np.linalg.eig(cov)
    print(w, v, sep='\n')
    w = w[::-1]
    v = np.transpose(v)
    v = np.array([v[1], v[0]])
    print(w, v, sep='\n')

    if display:
        for val, vec in zip(w, v):
            plt.plot([-val*vec[0], val*vec[0]], [-val*vec[1], val*vec[1]], label=val)
        plt.gca().set_aspect(1)
        plt.scatter(xs[0], xs[1], marker='x')
        plt.legend()
        plt.grid(True)
        plt.show()


# eigen()
pca1(True)

