import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class WeightedAttention(Layer):
    def __init__(self, n_heads, latent_dim=256, **kwargs):
        super(WeightedAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.latent_dim = latent_dim

    def build(self, input_shape):
        in_dim = input_shape[-1]
        latent_dim = self.latent_dim

        # projection weights
        self.query = tf.Variable(
            tf.random.normal((self.n_heads, in_dim, latent_dim)),
            trainable=True)
        self.key = tf.Variable(
            tf.random.normal((self.n_heads, in_dim, latent_dim)),
            trainable=True)
        self.value = tf.Variable(
            tf.random.normal((self.n_heads, in_dim, latent_dim)),
            trainable=True)

        # attention weights
        self.W = tf.Variable(
            tf.random.normal((self.n_heads, latent_dim, 1)),
            trainable=True)
        self.O = tf.Variable(
            tf.random.normal((self.n_heads * latent_dim, in_dim)),
            trainable=True)
        self.alpha = tf.Variable(tf.ones((self.n_heads,)), trainable=True)

    def call(self, input):
        origin = input[:, 0, :]
        walk = input[:, :, :]

        # repeat origin to match walk length
        origin_seq = tf.repeat(origin[:, None, :],
                               input.shape[1],
                               axis=1)

        # multiheaded attention
        results = []
        for i in range(self.n_heads):
            key = tf.nn.sigmoid(tf.matmul(origin_seq, self.key[i]))
            query = tf.nn.sigmoid(tf.matmul(walk, self.query[i]))
            value = tf.nn.sigmoid(tf.matmul(walk, self.value[i]))
            edge = key * query

            score = tf.matmul(edge, self.W[i])
            score = tf.nn.softmax(score, axis=-2)
            res = tf.math.reduce_sum(score * value, axis=1)
            results.append(res)

        # concat and project
        results = tf.concat(results, axis=-1)
        results = tf.nn.sigmoid(tf.matmul(results, self.O))

        # alpha = 1. / (1. + tf.exp(self.alpha))
        # return alpha * results + (1 - alpha) * origin
        return results + origin
        # return tf.concat([results, origin], axis=-1)


# class WeightedAttention(Layer):
#     def __init__(self, n_heads, latent_dim=256, **kwargs):
#         super(WeightedAttention, self).__init__(**kwargs)
#         self.n_heads = n_heads
#         self.latent_dim = latent_dim

#     def build(self, input_shape):
#         in_dim = input_shape[-1]
#         # latent_dim = self.latent_dim
#         latent_dim = in_dim // 2

#         # projection weights
#         self.A = tf.Variable(
#             tf.random.normal((in_dim, latent_dim)),
#             trainable=True)
#         self.B = tf.Variable(
#             tf.random.normal((self.n_heads, in_dim, latent_dim)),
#             trainable=True)

#         # attention weights
#         self.W = tf.Variable(
#             tf.random.normal((self.n_heads, in_dim, 1)),
#             trainable=True)
#         self.O = tf.Variable(
#             tf.random.normal((self.n_heads * latent_dim, latent_dim)),
#             trainable=True)
#         self.bias = tf.Variable(
#             np.zeros((1,), dtype=np.float32),
#             trainable=True)

#     def call(self, input):
#         origin = input[:, 0, :]
#         walk = input[:, 1:, :]
#         edge = walk * origin[:, None, :]

#         # multiheaded attention
#         results = []
#         for i in range(self.n_heads):
#             # compute score
#             score = 10 * tf.tanh(tf.matmul(edge, self.W[i]))
#             score = tf.nn.softmax(score, axis=1)

#             # compute values
#             values = tf.matmul(walk, self.B[i])
            
#             # linear combination of values
#             res = tf.math.reduce_sum(score * values, axis=-2, keepdims=False)
#             results.append(res)
        
#         # concat and project
#         results = tf.concat(results, axis=-1)
#         results = tf.matmul(results, self.O)

#         origin = tf.matmul(origin, self.A)
#         return tf.concat([results, origin], axis=-1)

