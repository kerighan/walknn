from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from .layer import WeightedAttention


def create_model(feats, labels, latent_dim, n_heads, walk_len, dropout=.1):
    n_nodes = feats.shape[0]
    n_feats = feats.shape[1]
    n_classes = labels.max() + 1

    # create model
    inp = Input(shape=(walk_len,))
    embedd = Embedding(n_nodes, n_feats, weights=[feats], trainable=False)(inp)
    embedd = Dense(latent_dim, activation="relu")(embedd)

    # layer
    layer = WeightedAttention(n_heads, latent_dim)(embedd)
    layer = Dropout(dropout)(layer)

    latent = Dense(128, activation="sigmoid")(layer)
    latent = Dropout(dropout)(latent)
    latent = Dense(128, activation="sigmoid")(latent)
    latent = Dropout(dropout)(latent)
    out = Dense(n_classes, activation="softmax")(latent)

    model = Model(inp, out)
    model.compile(
        "nadam",
        "sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    return model


def create_model_2nd(feats, labels, latent_dim, n_heads, walk_len, dropout=.1):
    n_nodes = feats.shape[0]
    n_feats = feats.shape[1]
    n_classes = labels.max() + 1

    # create model
    inp = Input(shape=(walk_len, walk_len))
    embedd = Embedding(n_nodes, n_feats, weights=[feats], trainable=False)
    dense = Dense(latent_dim, activation="sigmoid")

    # embedding and projection
    embedding = Sequential()
    embedding.add(TimeDistributed(embedd))
    embedding.add(TimeDistributed(dense))

    # layer
    layer = embedding(inp)
    layer = WeightedAttention(n_heads)(layer)
    layer = Dropout(dropout)(layer)

    # 2nd layer
    layer_2 = WeightedAttention(n_heads)(layer)
    layer_2 = Dropout(dropout)(layer_2)

    latent = Dense(latent_dim, activation="sigmoid")(layer_2)
    latent = Dropout(dropout)(latent)
    latent = Dense(latent_dim, activation="sigmoid")(latent)
    latent = Dropout(dropout)(latent)
    out = Dense(n_classes, activation="softmax")(latent)

    model = Model(inp, out)
    model.compile(
        "nadam",
        "sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    return model