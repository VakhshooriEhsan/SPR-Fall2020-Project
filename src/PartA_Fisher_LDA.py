import os 
from PIL import Image
import numpy as np

n = 7
classNames = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
# imData = [[], [], [], [], [], [], []]
x = [np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int), np.empty((0,256*256), int)]

dirname = 'data/jaffe'
for fname in os.listdir(dirname):
    # imData[classNames.index(fname[3:5])] += [Image.open(os.path.join(dirname, fname))]
    tmp = np.asarray(Image.open(os.path.join(dirname, fname))).flatten()
    x[classNames.index(fname[3:5])] = np.append(x[classNames.index(fname[3:5])], np.array([tmp]), axis=0)

mu = []
for i in range(n):
    mu += [np.mean(x[i], axis=0)]

# print(mu)

S = []
for i in range(n):
    S += [np.sum((x[i]-mu[i]) * (x[i]-mu[i]), axis=0)]

S_W = S[0].copy()
for i in range(1, n):
    S_W = S_W + S[i]

