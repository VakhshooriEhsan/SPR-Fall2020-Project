import os 
from PIL import Image
import numpy as np

C = 7
N = 0
classNames = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
# imData = [[], [], [], [], [], [], []]
x = [np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int)]

dirname = 'data/jaffe'
for fname in os.listdir(dirname):
    N += 1
    # imData[classNames.index(fname[3:5])] += [Image.open(os.path.join(dirname, fname))]
    tmp = np.asarray(Image.open(os.path.join(dirname, fname))).flatten()
    x[classNames.index(fname[3:5])] = np.append(x[classNames.index(fname[3:5])], np.array([tmp]), axis=0)

mu = []
for i in range(C):
    mu += [np.mean(x[i], axis=0)]

S = []
for i in range(C):
    S += [np.sum((x[i]-mu[i]) * (x[i]-mu[i]), axis=0)]

S_W = S[0].copy()
for i in range(1, C):
    S_W = S_W + S[i]

mu_ = mu[0] * len(x[0])
for i in range(1, C):
    mu_ = mu_ + (mu[i] * len(x[i]))
mu_ /= N

S_B = len(x[0]) * (mu[0]-mu_) * (mu[0]-mu_)
for i in range(1, C):
    S_B += len(x[i]) * (mu[i]-mu_) * (mu[i]-mu_)
