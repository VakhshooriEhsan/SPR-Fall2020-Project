import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification

# -------------------------------------- Linear SVM --------------------------------------
datafile = loadmat('data/Dataset1.mat')

X = datafile['X']
y = datafile['y']
data = np.column_stack((X,y))
np.random.shuffle(data)
X = data[:, [0, 1]]
y = data[:, [2]]
n = len(X)

train_X = X[:int(0.8*n), :]
test_X = X[int(0.8*n):, :]
train_y = y[:int(0.8*n), :]
test_y = y[int(0.8*n):, :]

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(train_X, train_y[:, 0])
w = clf.coef_[0]
a = -w[0] / w[1]
xx_1 = np.linspace(np.min(X[:, 0]),np.max(X[:, 0]))
yy_1 = a * xx_1 - clf.intercept_[0] / w[1]
ry_1 = clf.predict(test_X)

clf = svm.SVC(kernel='linear', C = 100.0)
clf.fit(train_X, train_y[:, 0])
w = clf.coef_[0]
a = -w[0] / w[1]
xx_100 = np.linspace(np.min(X[:, 0]),np.max(X[:, 0]))
yy_100 = a * xx_100 - clf.intercept_[0] / w[1]
ry_100 = clf.predict(test_X)

print("accuracy for C=1:")
print(np.sum(ry_1==test_y[:, 0])/len(test_y))
print("accuracy for C=100:")
print(np.sum(ry_100==test_y[:, 0])/len(test_y))

plt.figure(1)
plt.plot(train_X[:, [0]][train_y==0], train_X[:, [1]][train_y==0], '.', label='train datasets class_0')
plt.plot(train_X[:, [0]][train_y==1], train_X[:, [1]][train_y==1], '.', label='train datasets class_1')
plt.plot(xx_1, yy_1, '-', label='C = 1')
plt.plot(xx_100, yy_100, '-', label='C = 100')
plt.title('train datasets')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

plt.figure(2)
plt.plot(test_X[:, [0]][test_y==0], test_X[:, [1]][test_y==0], '.', label='test datasets class_0')
plt.plot(test_X[:, [0]][test_y==1], test_X[:, [1]][test_y==1], '.', label='test datasets class_1')
plt.plot(xx_1, yy_1, '-', label='C = 1')
plt.plot(xx_100, yy_100, '-', label='C = 100')
plt.title('test datasets')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

# plt.show()

print("-----------------------------------------------------------------")
# -------------------------------------- Kernel SVM for two-class --------------------------------------
datafile = loadmat('data/Dataset2.mat')

X = datafile['X']
y = datafile['y']
data = np.column_stack((X,y))
np.random.shuffle(data)
X = data[:, [0, 1]]
y = data[:, [2]]
n = len(X)

# _C = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
# res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# for C in range(len(_C)):
#     for time in range(10):
#         np.random.shuffle(data)
#         X = data[:, [0, 1]]
#         y = data[:, [2]]
#         for fold in range(10):
#             train_X = np.delete(X, np.s_[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))], axis=0)
#             train_y = np.delete(y, np.s_[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))], axis=0)
#             test_X = X[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X)), :]
#             test_y = y[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))]

#             clf = svm.SVC(kernel='rbf', C = _C[C])
#             clf.fit(train_X, train_y[:, 0])
#             ry = clf.predict(test_X)
#             res[C] += np.sum(ry==test_y[:, 0])/len(test_y)
#     res[C] /= 100

# BC = _C[np.argmax(np.array(res))]
# print("Best C:")
# print(BC)

# _C = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
# _gamma = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
# res = np.zeros((len(_C), len(_gamma)))

# for C in range(len(_C)):
#     for G in range(len(_gamma)):
#         for time in range(5):
#             np.random.shuffle(data)
#             X = data[:, [0, 1]]
#             y = data[:, [2]]
#             for fold in range(5):
#                 train_X = np.delete(X, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
#                 train_y = np.delete(y, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
#                 test_X = X[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X)), :]
#                 test_y = y[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))]

#                 clf = svm.SVC(kernel='rbf', C = _C[C], gamma=_gamma[G])
#                 clf.fit(train_X, train_y[:, 0])
#                 ry = clf.predict(test_X)
#                 res[C][G] += np.sum(ry==test_y[:, 0])/len(test_y)
#         res[C][G] /= 25

# print(res)

# nn = np.argmax(np.array(res))
# print("Best C:")
# print(_C[int(nn/len(_gamma))])
# print("Best gamma:")
# print(_gamma[nn%len(_gamma)])

# plt.show()

print("-----------------------------------------------------------------")
# -------------------------------------- Kernel SVM for multi-class --------------------------------------
data = pd.read_csv('data/vehicle.dat', sep=" ", header=None).values
X = data[:, 0:18]
y = data[:, [18]]
classes = np.unique(y[:, 0])
n = len(X)

_C = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for C in range(len(_C)):
    for time in range(10):
        np.random.shuffle(data)
        X = data[:, 0:18]
        y = data[:, [18]]
        for fold in range(10):
            train_X = np.delete(X, np.s_[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))], axis=0)
            train_y = np.delete(y, np.s_[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))], axis=0)
            test_X = X[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X)), :]
            test_y = y[int(fold*0.1*len(X)):int((fold+1)*0.1*len(X))]

            clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C = _C[C]))
            clf.fit(train_X, train_y[:, 0])
            ry = clf.predict(test_X)
            res[C] += np.sum(ry==test_y[:, 0])/len(test_y)
    res[C] /= 100

BC = _C[np.argmax(np.array(res))]
print("Best C:")
print(BC)

_C = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
_gamma = [0.001, 0.004, 0.01, 0.04, 0.1, 0.4, 1, 4]
res = np.zeros((len(_C), len(_gamma)))

for C in range(len(_C)):
    for G in range(len(_gamma)):
        for time in range(5):
            np.random.shuffle(data)
            X = data[:, 0:18]
            y = data[:, [18]]
            for fold in range(5):
                train_X = np.delete(X, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
                train_y = np.delete(y, np.s_[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))], axis=0)
                test_X = X[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X)), :]
                test_y = y[int(fold*0.2*len(X)):int((fold+1)*0.2*len(X))]

                clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C = _C[C], gamma=_gamma[G]))
                clf.fit(train_X, train_y[:, 0])
                ry = clf.predict(test_X)
                res[C][G] += np.sum(ry==test_y[:, 0])/len(test_y)
        res[C][G] /= 25

print(res)

nn = np.argmax(np.array(res))
print("Best C:")
print(_C[int(nn/len(_gamma))])
print("Best gamma:")
print(_gamma[nn%len(_gamma)])
