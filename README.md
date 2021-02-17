WalkNN - Random walk-based graph neural network 
================================================

Based on networkx and tensorflow.
This algorithm achieves the 5th best result on Cora and Citeseer datasets (see test_validation.py)

```python
import numpy as np
import networkx as nx
from walknn import WalkNN

G = nx.read_gexf("datasets/cora/G.gexf")
feats = np.load("datasets/cora/features.npy")
labels = np.load("datasets/cora/labels.npy")

nn = WalkNN(walk_len=5, n_walks=25,
            batch_size=300, epochs=2)
y_pred = nn.fit_predict(G, feats, labels)
nn.save("model.p")

nn = WalkNN.load("model.p")
y_pred = nn.predict(G, feats)

accuracy = (y_pred == labels).mean()
print(f"acc={accuracy:.03f}")
```