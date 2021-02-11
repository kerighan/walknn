import knn
import random
import numpy as np
from rwnn import RWNN
import networkx as nx
from sklearn.datasets import make_blobs
import tensorflow as tf


k = 10
N = 10000
n_test = 9500
X, y_true = make_blobs(N, 800, centers=15, cluster_std=40)


test_index = random.sample(range(N), n_test)
train_index = [i for i in range(N) if i not in test_index]

y_train = y_true.copy()
y_train[test_index] = -1
nn = knn.graph(X, k)
xs = np.repeat(np.arange(N), k)
ys = nn.flatten()
G = nx.DiGraph()
G.add_edges_from(zip(xs, ys))

inp = tf.keras.layers.Input(shape=(800,))
dense = tf.keras.layers.Dense(128, activation="sigmoid")(inp)
dense = tf.keras.layers.Dense(128, activation="sigmoid")(dense)
out = tf.keras.layers.Dense(15, activation="softmax")(dense)
model = tf.keras.models.Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X[train_index], y_true[train_index],
          epochs=40, batch_size=200,
          validation_data=(X[test_index], y_true[test_index]))


rwnn = RWNN(walk_len=6)
y_pred = rwnn.fit_transform(G, X, y_train, epochs=6,  batch_size=400)

print((y_pred == y_true).mean())
print((y_pred[train_index] == y_true[train_index]).mean())
print((y_pred[test_index] == y_true[test_index]).mean())

# rwnn = RWNN(walk_len=1, n_walks=1)
# y_pred = rwnn.fit_transform(G, X, y_train)

# print((y_pred == y_true).mean())
# print((y_pred[test_index] == y_true[test_index]).mean())
