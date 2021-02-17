from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from .layer import WeightedAttention


def create_classifier(
    n_feats,
    n_classes,
    latent_dim,
    n_heads,
    walk_len,
):
    inp_2 = Input(shape=(walk_len, n_feats))

    # layers
    embedd = Dense(latent_dim[0], activation="sigmoid")
    attention = WeightedAttention(n_heads, latent_dim[1])
    hidden_1 = Dense(latent_dim[2], activation="tanh")
    hidden_2 = Dense(latent_dim[2], activation="tanh")
    out = Dense(n_classes, activation="softmax")

    classifier = Sequential()
    classifier.add(inp_2)
    classifier.add(embedd)
    classifier.add(attention)
    classifier.add(hidden_1)
    classifier.add(hidden_2)
    classifier.add(out)
    classifier.compile(
        "nadam", "sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return classifier


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
    inp_2 = Input(shape=(walk_len, n_feats))

    # layers
    features = Embedding(n_nodes, n_feats, weights=[feats], trainable=False)
    embedd = Dense(latent_dim[0], activation="sigmoid")
    attention = WeightedAttention(n_heads, latent_dim[1])
    hidden_1 = Dense(latent_dim[2], activation="tanh")
    hidden_2 = Dense(latent_dim[2], activation="tanh")
    out = Dense(n_classes, activation="softmax")

    model = Sequential()
    model.add(inp)
    model.add(features)
    model.add(embedd)
    model.add(attention)
    model.add(hidden_1)
    model.add(hidden_2)
    model.add(out)
    model.compile(
        "nadam", "sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()

    classifier = Sequential()
    classifier.add(inp_2)
    classifier.add(embedd)
    classifier.add(attention)
    classifier.add(hidden_1)
    classifier.add(hidden_2)
    classifier.add(out)
    classifier.compile(
        "nadam", "sparse_categorical_crossentropy",
        metrics=["accuracy"])

    return model, classifier
