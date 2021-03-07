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
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# ------------------------------------------------------------------------------------------------------

data = pd.read_csv('data/UKM.csv', index_col=None, header=None).values
# np.random.shuffle(data)
X = data[:, 0:5]
Y = data[:, 5]

trainX = X[:int(0.8*len(X)), :]
trainY = Y[:int(0.8*len(X))]
testX = X[int(0.8*len(X)):, :]
testY = Y[int(0.8*len(X)):]
classes = np.unique(Y)
K = 1

xx=0
yy=4
pgm = []
for i in range(len(classes)):
    pgm += [GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[i]][:, [xx,yy]])]
for i in range(len(classes)):
    plt.figure(1)
    plot_gmm(pgm[i], trainX[trainY==classes[i]][:, [xx,yy]], label=False)
for i in range(len(classes)):
    plt.figure(2)
    plot_gmm(pgm[i], testX[testY==classes[i]][:, [xx,yy]], label=False)


gm = []
for i in range(len(classes)):
    gm += [GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[i]])]

y = []
for i in range(len(classes)):
    y += [multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[0], cov=gm[i].covariances_[0])]

for j in range(1, K):
    for i in range(len(classes)):
        y[i] = np.maximum(y[i], multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[j], cov=gm[i].covariances_[j]))

ry = testY.copy()
for i in range(len(classes)):
    tmp = (y[i]>=y[0])
    for j in range(1, len(classes)):
        tmp = np.logical_and(tmp, y[i]>=y[j])
    ry[tmp] = classes[i]
print(1-np.sum(ry!=testY)/len(testY))

plt.show()
