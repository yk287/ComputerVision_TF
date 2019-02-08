
import tensorflow as tf
from tf_util import *

from tensorflow.contrib import layers

def encoder(x):
    """
    Encoder part of a vanilla autoencoder

    :param x: Input Image
    :return:
    """

    with tf.name_scope('encoder'):
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 32, activation_fn=tf.nn.relu)

        mu = layers.fully_connected(x, num_outputs=16, activation_fn=None)
        logsigma = layers.fully_connected(x, num_outputs=16, activation_fn=None)
        sigma = tf.exp(logsigma)

        z = mu + tf.random_normal(tf.shape(sigma)) * sigma

        return z, mu, sigma

def decoder(z):
    """
    Decoder part of a vanilla autoencoder

    :param x:
    :return:
    """

    with tf.name_scope('decoder'):
        x = tf.contrib.layers.fully_connected(z, 32, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 784, activation_fn=tf.tanh)

        logits = x
        reconstruction = tf.nn.sigmoid(logits)

        return logits, reconstruction
