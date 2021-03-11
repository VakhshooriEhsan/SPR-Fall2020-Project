import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt

class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        target_classes = np.unique(y)

        mean_vectors = []
 
        for _cls in target_classes:
            mean_vectors.append(np.mean(X[y == _cls], axis=0))

        data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
        B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == target_classes[i]].shape[0]
            mean_vec = mean_vec.reshape(1, X.shape[1])
            mu1_mu2 = mean_vec - data_mean
        
            B += n * np.dot(mu1_mu2.T, mu1_mu2)
        
        s_matrix = []
 
        for _cls, mean in enumerate(mean_vectors):
            Si = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == target_classes[_cls]]:
                t = (row - mean).reshape(1, X.shape[1])
                Si += np.dot(t.T, t)
            s_matrix.append(Si)
        
        S = np.zeros((X.shape[1], X.shape[1]))
        for s_i in s_matrix:
            S += s_i
        
        S_inv = np.linalg.inv(S)
 
        S_inv_B = S_inv.dot(B)
        
        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)
    
        idx = eig_vals.argsort()[::-1]
 
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        return eig_vecs

y = []
X = []
dirname = 'data/jaffe'
for fname in os.listdir(dirname):
    y += [fname[3:5]]
    X += [np.asarray(Image.open(os.path.join(dirname, fname)).resize((32, 32))).flatten()]

y = np.array(y)
X = np.array(X)

lda = LDA()
eig_vecs = lda.fit(X, y)
W = eig_vecs[:, :1024]

transformed = X.dot(W)

print(transformed)

img = Image.fromarray(np.reshape(transformed[0], (32, 32)), 'L')
img.show()

# img = Image.fromarray(np.reshape(p, (32, 32)), 'L')
# img.show()
