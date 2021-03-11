import pandas as pd
import numpy as np
 
 
def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]] 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0])) 
    beta[V.shape[0] - 1] = np.ones((a.shape[0])) 
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
    return beta
 
 
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return (a, b)
 
 
def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
            prev[t - 1, j] = np.argmax(probability) 
            omega[t, j] = np.max(probability)
 
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S = np.flip(S, axis=0)
    return S
 

data = pd.read_csv('data/Corporation1.csv').values
data = data[200:900, 0]
data[data<50] = 0
data[np.logical_and(data>=50, data<100)] = 1
data[data>=100] = 2
V = data.astype(int)

a = np.ones((3, 3))
a = a / np.sum(a, axis=1)

b = np.array(((1, 3, 5), (2, 4, 6), (3, 3, 3)))
b = b / np.sum(b, axis=1).reshape((-1, 1))

initial_distribution = np.array((0.4, 0.3, 0.3))

a, b = baum_welch(V, a, b, initial_distribution, n_iter=100)

res = viterbi(V, a, b, initial_distribution)
res = np.array(res)
print(res)
