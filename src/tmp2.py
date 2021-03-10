import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import pandas as pd

data = pd.read_csv('data/vehicle.dat', sep=" ", header=None).values
X = data[:, 0:18]
y = data[:, 18]
data = np.column_stack((X,y))
np.random.shuffle(data)
X = data[:, [0, 1]]
y = data[:, 18]
iy = y.copy()
classes = np.unique(y)
for i in range(len(y)):
    for j in range(len(classes)):
        if(y[i]==classes[j]):
            iy[i] = float(j)
X_2d = X
y_2d = y

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

C_range = np.array([100, 400, 1000, 4000, 10000, 40000, 100000, 400000])
gamma_range = np.array([0.00000001, 0.00000004, 0.0000001, 0.0000004, 0.000001, 0.000004, 0.00001, 0.00004])

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

C_2d_range = np.array([10000])
gamma_2d_range = np.array([0.0000004])
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
# for C in C_range:
#     for gamma in gamma_ra/nge:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 0]
    Z = Z.reshape(xx.shape)
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    # plt.subplot(len(C_range), len(gamma_range), k + 1)
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iy, cmap=plt.cm.RdBu_r, edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

plt.show()
