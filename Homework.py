import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt


# %matplotlib inline

def plot_decision_boundary(X, Z, W=None, b=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1], c=Z, cmap=plt.cm.cool)
    ax.set_autoscale_on(False)

    a = - W[0, 0] / W[0, 1]
    xx = np.linspace(-30, 30)
    yy = a * xx - (b[0]) / W[0, 1]

    ax.plot(xx, yy, 'k-', c=plt.cm.cool(1.0 / 3.0))


def loadDataset(split, X=[], XT=[], Z=[], ZT=[]):
    dataset = datasets.load_iris()
    c = list(zip(dataset['data'], dataset['target']))
    np.random.seed(224)
    np.random.shuffle(c)
    x, t = zip(*c)
    sp = int(split * len(c))
    X = x[:sp]
    XT = x[sp:]
    Z = t[:sp]
    ZT = t[sp:]
    names = ['Sepal. length', 'Sepal. width', 'Petal. length', 'Petal. width']
    return np.array(X), np.array(XT), np.array(Z), np.array(ZT), names


def pred(X, W, b):
    value = X * W + b
    return 1 / (1 + math.exp(-value))


def loglikelihood(X, Z, W, b):
    sum = 0.0
    error = 0.0
    for i in range(0, X.shape[0]):
        # ToDo: Find how to add the bias
        sum += Z[i] * math.log1p(pred(W[i], X[i], b)) + (1 - Z[i]) * math.log1p(1 - pred(W[i], X[i], b))
    sum *= (-1)
    return np.empty_like(Z)


if __name__ == "__main__":
    # prepare data
    split = 0.67
    X, XT, Z, ZT, names = loadDataset(split)

    # combine two of the 3 classes for a 2 class problem
    Z[Z == 2] = 1
    ZT[ZT == 2] = 1

    # only look at 2 dimensions of the input data for easy visualisation
    X = X[:, :2]
    XT = XT[:, :2]
