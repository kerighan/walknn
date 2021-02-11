import walker
import numpy as np
import networkx as nx
from rwnn.model import create_model_2nd

G = nx.read_gexf("datasets/cora/G.gexf")
feats = np.load("datasets/cora/features.npy")
y_true = np.load("datasets/cora/labels.npy").argmax(axis=1)

# create test set
test_index = np.arange(len(G.nodes))
np.random.shuffle(test_index)
test_index = test_index[:1000]
labels = y_true.copy()
labels[test_index] = -1

n_nodes = len(G.nodes)
n_walks = 25
walk_len = 6
walks = walker.random_walks(G, n_walks=n_walks, walk_len=walk_len, p=.25, q=.75)
walks = walker.random_walks(
    G, n_walks=1, walk_len=walk_len, start_nodes=walks.flatten(), p=.25, q=.75)
walks = walks.reshape((n_nodes * n_walks, walk_len, walk_len))
labels = np.tile(labels, n_walks)

# seperate train and predict set
X_train, y_train = [], []
for i in range(walks.shape[0]):
    if labels[i] != -1:
        X_train.append(walks[i])
        y_train.append(labels[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# create and fit model
model = create_model_2nd(feats, y_true, 32, 4, walk_len)
model.fit(X_train, y_train, epochs=5, batch_size=200)

# predict for all walks
y_pred = model.predict(walks)
results = {}
for i in range(walks.shape[0]):
    sample = walks[i, 0, 0]
    if sample in results:
        results[sample] *= y_pred[i]
    else:
        results[sample] = y_pred[i]

y_final = [0] * len(G.nodes)
for i in results:
    y_final[i] = results[i].argmax()
y_final = np.array(y_final)

print((y_final == y_true).mean())
print(test_index)
print(y_final[test_index])
print(y_true[test_index])
print((y_final[test_index] == y_true[test_index]).mean())
