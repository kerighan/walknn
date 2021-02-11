from .model import create_model
import numpy as np
import walker


class RWNN:
    def __init__(
        self,
        n_walks=25,
        walk_len=10,
        latent_dim=256,
        n_heads=8,
        p=.25, q=.85,
        subsampling=.1
    ):
        self.n_walks = n_walks
        self.walk_len = walk_len
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.p = p
        self.q = q
        self.subsampling = subsampling
    
    def fit_transform(self, G, X, y, batch_size=400, epochs=1):
        # create random walks
        walks = walker.random_walks(
            G, p=self.p, q=self.q,
            n_walks=self.n_walks, walk_len=self.walk_len,
            sub_sampling=self.subsampling)
        labels = np.tile(y, self.n_walks)

        # seperate train and predict set
        X_train, y_train = [], []
        for i in range(walks.shape[0]):
            if labels[i] != -1:
                X_train.append(walks[i])
                y_train.append(labels[i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # create and fit model
        model = create_model(X, y, self.latent_dim, self.n_heads, self.walk_len)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # predict for all walks
        y_pred = model.predict(walks)
        results = {}
        for i in range(walks.shape[0]):
            sample = walks[i, 0]
            if sample in results:
                results[sample] *= y_pred[i]
            else:
                results[sample] = y_pred[i]
        
        y_final = [0] * len(G.nodes)
        for i in results:
            y_final[i] = results[i].argmax()
        return np.array(y_final)
