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
        # latent_dim = self.latent_dim
        latent_dim = in_dim // 2

        # projection weights
        self.A = tf.Variable(
            tf.random.normal((in_dim, latent_dim)),
            trainable=True)
        self.B = tf.Variable(
            tf.random.normal((self.n_heads, in_dim, latent_dim)),
            trainable=True)

        # attention weights
        self.W = tf.Variable(
            tf.random.normal((self.n_heads, in_dim, 1)),
            trainable=True)
        self.O = tf.Variable(
            tf.random.normal((self.n_heads * latent_dim, latent_dim)),
            trainable=True)
        self.bias = tf.Variable(
            np.zeros((1,), dtype=np.float32),
            trainable=True)

    def call(self, input):
        walk_len = input.shape[-2]

        origin = input[:, 0, :]
        walk = input[:, 1:, :]

        edge = tf.tanh(walk - origin[:, None, :])
        
        # attention selector
        # edge_a = tf.tanh(walk - origin[:, None, :])
        # edge_b = tf.tanh(walk * origin[:, None, :])
        # bias = tf.nn.sigmoid(self.bias)
        # edge = bias * edge_a + (1 - bias) * edge_b

        # multiheaded attention
        results = []
        for i in range(self.n_heads):
            # compute score
            score = 10 * tf.tanh(tf.matmul(edge, self.W[i]))
            score = tf.nn.softmax(score, axis=1)

            # compute values
            values = tf.matmul(walk, self.B[i])
            
            # linear combination of values
            res = tf.math.reduce_sum(score * values, axis=-2, keepdims=False)
            results.append(res)
        
        # concat and project
        results = tf.concat(results, axis=-1)
        results = tf.matmul(results, self.O)

        origin = tf.matmul(origin, self.A)
        return tf.concat([results, origin], axis=-1)


# class WeightedAttention(Layer):
#     def __init__(self, n_heads, **kwargs):
#         super(WeightedAttention, self).__init__(**kwargs)
#         self.n_heads = n_heads

#     def build(self, input_shape):
#         in_dim = input_shape[-1]
#         self.W = tf.Variable(
#             tf.random.normal((self.n_heads, in_dim, 1)),
#             trainable=True)
#         self.W_a = tf.Variable(
#             tf.random.normal((in_dim, in_dim // 2)),
#             trainable=True)
#         self.W_b = tf.Variable(
#             tf.random.normal((in_dim, in_dim // 2)),
#             trainable=True)
#         self.O = tf.Variable(
#             tf.random.normal((self.n_heads * in_dim, in_dim)),
#             trainable=True)
#         self.bias = tf.Variable(
#             np.zeros((1,), dtype=np.float32),
#             trainable=True)

#     def call(self, input):
#         walk_len = input.shape[-2]

#         print(input.shape, "input")
#         if len(input.shape) == 4:
#             origin = input[:, 0, 0, :]
#             # edge = tf.math.tanh(input * origin[:, None, None, :])

#             a = tf.matmul(input, self.W_a)
#             b = tf.repeat(
#                 tf.repeat(tf.matmul(origin, self.W_b)[:, None, None, :], walk_len, 2),
#                 walk_len, 1)

#             print(a.shape)
#             print(b.shape)

#             edge = tf.tanh(tf.concat([a, b], axis=-1))
#         else:
#             origin = input[:, 0, :]
#             # edge = tf.math.tanh(input * origin[:, None, :])
#             edge = tf.tanh(tf.concat([
#                 tf.matmul(input, self.W_b),
#                 tf.repeat(tf.matmul(origin, self.W_a)[:, None, :], walk_len, 1)
#             ], axis=-1))
#         print(edge.shape, "edge")

#         # edge = tf.math.tanh(tf.matmul(input * origin[:, None, :], self.W_a))
#         # edge = tf.math.cos(input - origin[:, None, :])
#         # edge = (input - origin[:, None, :]) + self.bias
#         # edge = tf.math.tanh((input * origin[:, None, :]) * self.bias[0] + self.bias[1])

#         # print(edge_b)
#         # edge = edge_a * self.bias + edge_b * (1 - self.bias)

#         results = []
#         for i in range(self.n_heads):
#             score = tf.matmul(edge, self.W[i])
#             score = tf.nn.softmax(score, axis=1)
#             res = tf.math.reduce_sum(score * input, axis=-2, keepdims=False)
#             results.append(res)
#         results = tf.concat(results, axis=-1)
#         results = tf.matmul(results, self.O)
#         print(results.shape, "res")
#         if len(results.shape) == 3:
#             return results * (1 - self.bias) + origin[:, None, :]
#         else:
#             return results * (1 - self.bias) + origin
#         # return tf.concat([results, origin], axis=-1)
#         # return results * (1 - self.bias) + origin * self.bias
