
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

        x = tf.contrib.layers.conv2d(x, 64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.conv2d(x, 128, kernel_size=4, stride=1, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.conv2d(x, 256, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.conv2d(x, 512, kernel_size=4, stride=1, activation_fn=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)

        mu = layers.fully_connected(x, num_outputs=128, activation_fn=None)
        logsigma = layers.fully_connected(x, num_outputs=128, activation_fn=None)
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

        x = layers.fully_connected(z, num_outputs= 512 * 4 * 4)
        x = tf.reshape(x, [-1, 4, 4, 512])

        x = tf.contrib.layers.conv2d_transpose(x, 256, kernel_size=4, stride=2, padding='valid', activation_fn=tf.nn.relu)
        x = tf.contrib.layers.conv2d_transpose(x, 64, kernel_size=4, stride=2, padding='valid', activation_fn=tf.nn.relu)
        x = tf.contrib.layers.conv2d_transpose(x, 1, kernel_size=7, padding='valid', activation_fn=None)

        logits = x
        reconstruction = tf.nn.sigmoid(logits)

        return logits, reconstruction
