import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pickle

import snips as snp  # my snippets
snp.prettyplot(matplotlib)  # my aesthetic preferences for plotting

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)
#
# df = df.iloc[:5, :5]
# df["user_id"] = [1, 1, 2, 2, 3]
# df["item_id"] = [1, 2, 1, 2, 1]
# df["rating"] = [5, 3, 1, 5, 5]



n_u = len(df["user_id"].unique())
n_m = len(df["item_id"].unique())
sparsity = len(df)/(n_u*n_m)
print("sparsity of ratings is %.2f%%" %(sparsity*100))

# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(df,test_size=0.25)
#
# train_data = pd.DataFrame(train_data)
# test_data = pd.DataFrame(test_data)


train_data = df
# Create training and test matrix
R = np.zeros((n_u, n_m))
for line in train_data.itertuples():
    R[line[1] - 1, line[2] - 1] = line[3]

# T = np.zeros((n_u, n_m))
# for line in test_data.itertuples():
#     T[line[1] - 1, line[2] - 1] = line[3]


n_u = n_m = 100
R = R[:n_u, :n_m]


users, movies = R.nonzero()
users = np.unique(users)
movies = np.unique(movies)

f = 10
from numpy import random

# P = np.array([[1.0, 1], [1, 1], [1, 1]])
# Q = np.array([[1, 1 ], [1, 1], [1, 1.0]])
# R = np.array([[2, 0, 5], [3, 2, 5], [0, 0, 1]])


# P = np.array([[1.0, 1], [1, 1], [1, 1]])
# Q = np.array([[1, 1 ], [1, 1], [1, 1.0]])

P = 3 * np.random.rand(n_u, f)  # Latent user feature matrix
Q = 3 * np.random.rand(n_m, f)  # Latent movie feature matrix

# Compute sum of the element-wise MSE
def rmse_score(R, Rp):
    E = np.multiply((R - Rp), R > 0)
    SE = np.multiply(E, E)
    rmse = np.sqrt(np.sum(SE) / np.count_nonzero(SE))
    return rmse


runs = 40
gamma = 5e-5  # learning rate
lmbda = 0.1  # regularization strength

train_errors = []
test_errors = []
for run in range(0, runs):
    train_errors.append(rmse_score(R, np.dot(P, Q.T)))
    # test_errors.append(rmse_score(T, np.dot(P, Q.T)))

    ERR = R - np.dot(P, Q.T)  # compute error with present values of Q, P
    Pcopy = P.copy()  # Make a copy of Q so that we aren't using overwritten values in the P update

    for u in users:
        rated = np.nonzero(R[u, :])[0]  # column indices for movies rated by this user
        err = ERR[u, :].take(rated)  # row of errors for ratings by this user
        Qu = Q.take(rated, axis=0)  # sub-matrix of movie features for movies rated by this user
        P[u, :] += gamma * (np.dot(Qu.T, err))  # update rule

    for m in movies:
        rated = np.nonzero(R[:, m])[0]  # row indices for users who rated this movie
        err = ERR[:, m].take(rated)  # row of errors for ratings of movies rated by this user
        Pm = Pcopy.take(rated, axis=0)  # sub-matrix of movie features of movies rated by this user
        Q[m, :] = Q[m, :] + gamma * (np.dot(Pm.T, err))  # update rule

train_errors