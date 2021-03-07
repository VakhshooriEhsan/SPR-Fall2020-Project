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
    # labels = gmm.fit(X).predict(X)
    # if label:
    #     ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # else:
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
trainX = X[:int(0.7*len(X)), :]
trainY = Y[:int(0.7*len(X))]
testX = X[int(0.7*len(X)):, :]
testY = Y[int(0.7*len(X)):]
classes = np.unique(Y)
K = 1

gm = []

for i in range(len(classes)):
    gm += [GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[i]])]
# gm0 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[0]])
# gm1 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[1]])
# gm2 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[2]])
# gm3 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[3]])

# xx=0
# yy=4
# pgm0 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[0]][:, [xx, yy]])
# pgm1 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[1]][:, [xx, yy]])
# pgm2 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[2]][:, [xx, yy]])
# pgm3 = GaussianMixture(n_components=K, random_state=0).fit(trainX[trainY==classes[3]][:, [xx, yy]])
# plot_gmm(gm0, trainX[trainY==classes[0]], label=False)
# plot_gmm(gm1, trainX[trainY==classes[1]], label=False)
# plot_gmm(gm2, trainX[trainY==classes[2]], label=False)
# plot_gmm(gm3, trainX[trainY==classes[3]], label=False)

# plt.figure(2)
# plot_gmm(gm0, testX[testY==classes[0]], label=False)
# plot_gmm(gm1, testX[testY==classes[1]], label=False)
# plot_gmm(gm2, testX[testY==classes[2]], label=False)
# plot_gmm(gm3, testX[testY==classes[3]], label=False)

y = []
for i in range(len(classes)):
    y += [multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[0], cov=gm[i].covariances_[0])]
# y0 = multivariate_normal.pdf(np.array(testX), mean=gm0.means_[0], cov=gm0.covariances_[0])
# y1 = multivariate_normal.pdf(np.array(testX), mean=gm1.means_[0], cov=gm1.covariances_[0])
# y2 = multivariate_normal.pdf(np.array(testX), mean=gm2.means_[0], cov=gm2.covariances_[0])
# y3 = multivariate_normal.pdf(np.array(testX), mean=gm3.means_[0], cov=gm3.covariances_[0])

for j in range(1, K):
    for i in range(len(classes)):
        y[i] = np.maximum(y[i], multivariate_normal.pdf(np.array(testX), mean=gm[i].means_[j], cov=gm[i].covariances_[j]))
        # y0 = np.maximum(y0, multivariate_normal.pdf(np.array(testX), mean=gm0.means_[i], cov=gm0.covariances_[i]))
        # y1 = np.maximum(y1, multivariate_normal.pdf(np.array(testX), mean=gm1.means_[i], cov=gm1.covariances_[i]))
        # y2 = np.maximum(y2, multivariate_normal.pdf(np.array(testX), mean=gm2.means_[i], cov=gm2.covariances_[i]))
        # y3 = np.maximum(y3, multivariate_normal.pdf(np.array(testX), mean=gm3.means_[i], cov=gm3.covariances_[i]))

ry = testY.copy()
for i in range(len(classes)):
    tmp = (y[i]>=y[0])
    for j in range(1, len(classes)):
        tmp = np.logical_and(tmp, y[i]>=y[j])
    ry[tmp] = classes[i]
# y[np.logical_and(y0>=y1, y0>=y2, y0>=y3)] = classes[0]
# y[np.logical_and(y1>=y0, y1>=y2, y1>=y3)] = classes[1]
# y[np.logical_and(y2>=y0, y2>=y1, y2>=y3)] = classes[2]
# y[np.logical_and(y3>=y0, y3>=y1, y3>=y2)] = classes[3]
print(1-np.sum(ry!=testY)/len(testY))

# plt.figure(1)
# plt.plot(X[Y==classes[0]][:, 0], X[Y==classes[0]][:, 1], '.', label='Train_datasets_1 class_0 (T)')
# plt.plot(M[:, 0], M[:, 1], 'o', label='closed form solution')
# plt.title('Bayesian classification Train_datasets_1')
# plt.ylabel('x1')
# plt.xlabel('x0')
# plt.legend()

plt.show()
