import networkx as nx
import pandas as pd
import numpy as np


content = open("datasets/citeseer/citeseer.content").read().splitlines()
content = [item.split("\t") for item in content]
content = [[item[0], list(map(int, item[1:-1])), item[-1]] for item in content]


df = pd.DataFrame(content)
print(df)
df.columns = ["node", "features", "label"]
df.to_pickle("datasets/citeseer/dataframe.p")

nodes = df["node"].tolist()
node2id = {node: i for i, node in enumerate(nodes)}

edges = open("datasets/citeseer/citeseer.cites").read().splitlines()
edges = [item.split("\t") for item in edges]
edges = [[src, tgt] for src, tgt in edges if src in nodes and tgt in nodes]

features = np.array(df["features"].tolist())
np.save("datasets/citeseer/features.npy", features)
print(features)

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
nx.write_gexf(G, "datasets/citeseer/G.gexf")

labels = set(df["label"])
label2id = {label: i for i, label in enumerate(labels)}
y_true = np.array([label2id[label] for label in df["label"]])
np.save("datasets/citeseer/labels.npy", y_true)
