from .model import create_model, create_classifier
import numpy as np
import walker
import time


class WalkNN:
    def __init__(
        self,
        n_walks=20,
        walk_len=5,
        latent_dim=[64, 16, 32],
        n_heads=4,
        p=.2, q=2,
        subsampling=.1,
        batch_size=400,
        epochs=1
    ):
        self.n_walks = n_walks
        self.walk_len = walk_len
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.p = p
        self.q = q
        self.subsampling = subsampling
        self.batch_size = batch_size
        self.epochs = epochs

    def fit_predict(self, G, feats, y):
        # get constants
        self.n_feats = feats.shape[1]
        self.n_classes = y.max() + 1
        n_nodes = len(G.nodes)

        # create random walks
        walks = walker.random_walks(
            G, p=self.p, q=self.q,
            n_walks=self.n_walks, walk_len=self.walk_len,
            sub_sampling=self.subsampling)
        labels = np.tile(y, self.n_walks)

        # seperate train and predict set
        train_set = np.where(y != -1)[0]
        has_validation_data = train_set.shape[0] != n_nodes
        if not has_validation_data:
            X_train = walks
            y_train = labels
        else:
            walk_train_set = np.where(labels != -1)[0]
            X_train = walks[walk_train_set]
            y_train = labels[walk_train_set]

        # create model
        model, classifier = create_model(
            feats, y,
            self.latent_dim,
            self.n_heads,
            self.walk_len)

        # fit model
        model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size)
        self.classifier = classifier

        # predict for all walks
        y_pred = model.predict(walks)            

        # combine 
        res = np.zeros((n_nodes,), dtype=np.uint16)
        for i in range(n_nodes):
            index = n_nodes * np.arange(self.n_walks) + i
            pred = np.argmax(np.sum(y_pred[index], axis=0))
            res[i] = pred
        return res

    def predict(self, G, X):
        from tensorflow.keras.layers import Input, Embedding
        from tensorflow.keras.models import Sequential

        n_nodes = len(G.nodes)
        n_feats = X.shape[1]

        # create random walks
        walks = walker.random_walks(
            G, p=self.p, q=self.q,
            n_walks=self.n_walks, walk_len=self.walk_len,
            sub_sampling=self.subsampling, verbose=False)

        inp = Input((self.walk_len,))
        features = Embedding(n_nodes, n_feats, weights=[X])

        model = Sequential()
        model.add(inp)
        model.add(features)
        model.add(self.classifier)
        model.compile("nadam", "categorical_crossentropy", 
                      metrics=["accuracy"])
        y_pred = model.predict(walks, batch_size=1000)

        res = np.zeros((n_nodes,), dtype=np.uint16)
        for i in range(n_nodes):
            index = n_nodes * np.arange(self.n_walks) + i
            pred = np.argmax(np.sum(y_pred[index], axis=0))
            res[i] = pred
        return res

    def save(self, filename):
        import pickle
        with open(filename, "wb") as f:
            self.classifier_weights = self.classifier.get_weights()
            del self.classifier
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        import pickle
        with open(filename, "rb") as f:
            nn = pickle.load(f)

        if hasattr(nn, "classifier_weights"):
            nn.classifier = create_classifier(
                nn.n_feats,
                nn.n_classes,
                nn.latent_dim,
                nn.n_heads,
                nn.walk_len)
            nn.classifier.set_weights(nn.classifier_weights)
            del nn.classifier_weights
        return nn
