import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.initializers import RandomNormal


class WeightedAttention(Layer):
    def __init__(self, n_heads, latent_dim=256, **kwargs):
        super(WeightedAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.latent_dim = latent_dim

    def build(self, input_shape):
        in_dim = input_shape[-1]
        latent_dim = self.latent_dim

        initializer = RandomNormal(mean=0., stddev=1.)

        transform_shape = (self.n_heads, in_dim, latent_dim)
        # projection weights
        self.query = self.add_weight(
            "query", transform_shape,
            initializer=initializer,
            regularizer=l1(1e-6),
            dtype=np.float32)
        self.key = self.add_weight(
            "key", transform_shape,
            initializer=initializer,
            regularizer=l1(1e-6),
            dtype=np.float32)
        self.value = self.add_weight(
            "value", transform_shape,
            initializer=initializer,
            regularizer=l1(1e-6),
            dtype=np.float32)

        # attention weights
        self.W = self.add_weight(
            "W", (self.n_heads, latent_dim, 1),
            initializer=initializer,
            dtype=np.float32)
        self.O = self.add_weight(
            "O", (self.n_heads * latent_dim, in_dim),
            initializer=initializer,
            dtype=np.float32)
        self.P = self.add_weight(
            "P", (input_shape[-2] - 1, in_dim),
            regularizer=l1(1e-6),
            initializer=initializer,
            dtype=np.float32)
        self.bias = self.add_weight(
            "bias", (self.n_heads,),
            initializer=initializer,
            regularizer=l2(1e-6),
            dtype=np.float32)

    def call(self, input):
        origin = input[:, 0, :]
        walk = input[:, 1:, :]

        # repeat origin to match walk length
        origin_seq = tf.repeat(origin[:, None, :],
                               input.shape[1] - 1,
                               axis=1)
        origin_seq += self.P

        # multiheaded attention
        results = []
        for i in range(self.n_heads):
            key = tf.nn.sigmoid(tf.matmul(origin_seq, self.key[i]))
            query = tf.nn.sigmoid(tf.matmul(walk, self.query[i]))
            value = tf.nn.sigmoid(tf.matmul(walk, self.value[i]))
            edge = tf.cos(key * query) + self.bias[i]

            score = tf.matmul(edge, self.W[i])
            score = tf.nn.softmax(score, axis=-2)
            res = tf.math.reduce_sum(score * value, axis=1)
            results.append(res)

        # concat and project
        results = tf.concat(results, axis=-1)
        results = tf.nn.sigmoid(tf.matmul(results, self.O))
        # return results + origin

        # origin_transform = tf.matmul(origin, self.P)
        return tf.concat([results, origin], axis=-1)
        # return results


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

