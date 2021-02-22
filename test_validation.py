import numpy as np
import networkx as nx
from walknn import WalkNN

G = nx.read_gexf("datasets/cora/G.gexf")
feats = np.load("datasets/cora/features.npy")
y_true = np.load("datasets/cora/labels.npy")

# G = nx.read_gexf("datasets/citeseer/G.gexf")
# feats = np.load("datasets/citeseer/features.npy")
# y_true = np.load("datasets/citeseer/labels.npy")

# create random test set
labels = y_true.copy()
test_index = np.random.choice(
    range(len(G.nodes)),
    replace=False,
    size=1000)
labels[test_index] = -1

nn = WalkNN(n_walks=25, walk_len=20, epochs=2, batch_size=500)
y_pred = nn.fit_predict(G, feats, labels)

accuracy = (y_pred == y_true).mean()
val_accuracy = (y_pred[test_index] == y_true[test_index]).mean()
print(f"acc={accuracy:.03f}")
print(f"val_acc={val_accuracy:.03f}")
nn.save("model.p")

nn = WalkNN.load("model.p")
y_pred = nn.predict(G, feats)
accuracy = (y_pred == y_true).mean()
print(f"acc={accuracy:.03f}")
