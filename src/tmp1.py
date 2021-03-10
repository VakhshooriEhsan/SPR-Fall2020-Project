import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat

datafile = loadmat('data/Dataset2.mat')
X = datafile['X']
y = datafile['y']
data = np.column_stack((X,y))
np.random.shuffle(data)
X = data[:, [0, 1]]
y = data[:, 2]
print(y)
X_2d = X
y_2d = y

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

C_range = np.array([0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40])
gamma_range = np.array([0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40])

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

C_2d_range = np.array([4])
gamma_2d_range = np.array([10])
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
# for C in C_range:
#     for gamma in gamma_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    # plt.subplot(len(C_range), len(gamma_range), k + 1)

    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

plt.show()
