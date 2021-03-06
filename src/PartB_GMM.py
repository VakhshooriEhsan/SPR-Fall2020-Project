import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# ------------------------------------------------------------------------------------------------------

data = pd.read_csv('data/UKM.csv', index_col=None, header=None).values
# data = pd.read_csv('data/iris.data', sep=",", header=None).values
# data = pd.read_csv('data/vehicle.dat', sep=" ", header=None).values[:, :19]
# data = pd.read_csv('data/Health.dat', sep=" ", header=None).values
np.random.shuffle(data)
X = data[:, 0:(len(data[0])-1)]
Y = data[:, len(data[0])-1]


trainX = X[:int(0.8*len(X)), :]
trainY = Y[:int(0.8*len(X))]
testX = X[int(0.8*len(X)):, :]
testY = Y[int(0.8*len(X)):]
classes = np.unique(Y)
_K = [1, 5, 10]

for K in range(len(_K)):
    xx=0
    yy=1
    pgm = []
    for i in range(len(classes)):
        pgm += [GaussianMixture(n_components=_K[K], random_state=0).fit(trainX[trainY==classes[i]][:, [xx,yy]])]
    for i in range(len(classes)):
        plt.figure(1)
        plot_gmm(pgm[i], trainX[trainY==classes[i]][:, [xx,yy]], label=False)
    for i in range(len(classes)):
        plt.figure(2)
        plot_gmm(pgm[i], testX[testY==classes[i]][:, [xx,yy]], label=False)

    plt.show()

train_res = [0.0, 0.0, 0.0]
res = [0.0, 0.0, 0.0]
for K in range(len(_K)):
    for time in range(5):
        np.random.shuffle(data)
        X = data[:, 0:len(data[0])-1]
        Y = data[:, len(data[0])-1]
        for fold in range(5):
            trainX = np.delete(X, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
            trainY = np.delete(Y, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
            testX = X[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X)), :]
            testY = Y[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))]

            gm = []
            for i in range(len(classes)):
                gm += [GaussianMixture(n_components=_K[K], random_state=0).fit(trainX[trainY==classes[i]])]

            y = []
            for i in range(len(classes)):
                y += [multivariate_normal.pdf(np.array(trainX), mean=gm[i].means_[0], cov=gm[i].covariances_[0])]

            for j in range(1, _K[K]):
                for i in range(len(classes)):
                    y[i] = np.maximum(y[i], multivariate_normal.pdf(np.array(trainX), mean=gm[i].means_[j], cov=gm[i].covariances_[j]))

            ry = trainY.copy()
            for i in range(len(classes)):
                tmp = (y[i]>=y[0])
                for j in range(1, len(classes)):
                    tmp = np.logical_and(tmp, y[i]>=y[j])
                ry[tmp] = classes[i]
            train_res[K] += 1-np.sum(ry!=trainY)/len(trainY)

            y = []
            for i in range(len(classes)):
                y += [multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[0], cov=gm[i].covariances_[0])]

            for j in range(1, _K[K]):
                for i in range(len(classes)):
                    y[i] = np.maximum(y[i], multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[j], cov=gm[i].covariances_[j]))

            ry = testY.copy()
            for i in range(len(classes)):
                tmp = (y[i]>=y[0])
                for j in range(1, len(classes)):
                    tmp = np.logical_and(tmp, y[i]>=y[j])
                ry[tmp] = classes[i]
            res[K] += 1-np.sum(ry!=testY)/len(testY)
    res[K] /= 25
    train_res[K] /= 25

print("Average train accuracy for each K = 1, 5, 10:")
print(train_res)

print("Average test accuracy for each K = 1, 5, 10:")
print(res)

BK = _K[np.argmax(np.array(res))]
print("Best K:")
print(BK)
