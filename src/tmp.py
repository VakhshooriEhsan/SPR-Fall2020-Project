import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

from scipy.io import loadmat
datafile = loadmat('data/Dataset2.mat')
X = datafile['X']
y = datafile['y']
data = np.column_stack((X,y))
np.random.shuffle(data)
X = data[:, [0, 1]]
y = data[:, 2]

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

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

C_2d_range = np.array([4])
gamma_2d_range = np.array([10])
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)

    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

# scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
scores = np.array([[0.53032618, 0.5304496,  0.53702028, 0.53037908, 0.53146635, 0.71120188,
  0.54843961, 0.53040846],
 [0.53043197, 0.53044373, 0.53043785, 0.81409345, 0.83598002, 0.88367323,
  0.90264473, 0.86420217],
 [0.53039083, 0.53043197, 0.70516603, 0.84816339, 0.88124008, 0.90461358,
  0.91336468, 0.93482809],
 [0.53040259, 0.8228563,  0.8554687,  0.89879518, 0.92115192, 0.92701146,
  0.93867764, 0.93723185],
 [0.71583309, 0.86325595, 0.8929474,  0.92160447, 0.92943873, 0.93672054,
  0.93769615, 0.93382897],
 [0.86569498, 0.90411989, 0.9221099,  0.92654129, 0.92699383, 0.93816632,
  0.93720835, 0.92940934],
 [0.89682045, 0.92411989, 0.92259183, 0.92940347, 0.9353159,  0.94792242,
  0.93483397, 0.9314017 ],
 [0.91727299, 0.9269762,  0.92702909, 0.92993241, 0.93238907, 0.94256244,
  0.9343109,  0.92989127]])


plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
