import numpy as np
import networkx as nx
import walker
from rwnn import RWNN
import time

G = nx.read_gexf("datasets/cora/G.gexf")
feats = np.load("datasets/cora/features.npy")
y_true = np.load("datasets/cora/labels.npy").argmax(axis=1)

# G = nx.read_gexf("datasets/citeseer/G.gexf")
# feats = np.load("datasets/citeseer/features.npy")
# y_true = np.load("datasets/citeseer/labels.npy")

# create test set
test_index = np.arange(len(G.nodes))
np.random.shuffle(test_index)
test_index = test_index[:1000]
labels = y_true.copy()
labels[test_index] = -1

start = time.time()
rwnn = RWNN(walk_len=6, n_walks=30)
y_pred = rwnn.fit_transform(G, feats, labels, epochs=2, batch_size=300)
print((y_pred == y_true).mean())
print((y_pred[test_index] == y_true[test_index]).mean())
print(time.time() - start)
