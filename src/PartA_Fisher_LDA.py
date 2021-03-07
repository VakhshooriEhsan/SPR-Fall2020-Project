import os 
from PIL import Image
import numpy as np

C = 7
N = 0
classNames = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
# imData = [[], [], [], [], [], [], []]
x = [np.empty((0,32*32), int), np.empty((0,32*32), int), np.empty((0,32*32), int), np.empty((0,32*32), int), np.empty((0,32*32), int), np.empty((0,32*32), int), np.empty((0,32*32), int)]

dirname = 'data/jaffe'
for fname in os.listdir(dirname):
    N += 1
    # imData[classNames.index(fname[3:5])] += [Image.open(os.path.join(dirname, fname))]
    tmp = np.asarray(Image.open(os.path.join(dirname, fname)).resize((32, 32))).flatten()
    x[classNames.index(fname[3:5])] = np.append(x[classNames.index(fname[3:5])], np.array([tmp]), axis=0)

mu = []
for i in range(C):
    mu += [np.mean(x[i], axis=0)]

S = []
for i in range(C):
    S += [len(x[i]) * np.cov(np.transpose(x[i]))]

S_W = S[0].copy()
for i in range(1, C):
    S_W = S_W + S[i]

mu_ = mu[0] * len(x[0])
for i in range(1, C):
    mu_ = mu_ + (mu[i] * len(x[i]))
mu_ /= N

S_B = len(x[0]) * np.matmul(np.transpose([mu[0]-mu_]), [mu[0]-mu_])
for i in range(1, C):
    S_B += len(x[i]) * np.matmul(np.transpose([mu[i]-mu_]), [mu[i]-mu_])

# S_W = np.transpose(S_W)
# S_B = np.transpose(S_B)

eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(S_W),S_B))
# print(len(eigval))
eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
print(len(eigen_pairs))
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
print(len(eigen_pairs[0]))
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))

print(w)
