from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from .layer import WeightedAttention


def create_model(
    feats,
    labels,
    latent_dim,
    n_heads,
    walk_len,
):
    n_nodes = feats.shape[0]
    n_feats = feats.shape[1]
    n_classes = labels.max() + 1

    # create model
    inp = Input(shape=(walk_len,))
    features = Embedding(n_nodes, n_feats,
                         weights=[feats],
                         trainable=False)(inp)
    embedd = Dense(latent_dim[0], activation="sigmoid")(features)

    # layer
    layer = WeightedAttention(n_heads, latent_dim[1])(embedd)

    latent = Dense(latent_dim[2], activation="tanh")(layer)
    latent = Dense(latent_dim[2], activation="tanh")(latent)
    out = Dense(n_classes, activation="softmax")(latent)

    model = Model(inp, out)
    model.compile(
        "nadam", "sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    return model
