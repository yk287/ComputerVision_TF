
import tensorflow as tf
from tf_util import *

def encoder(x, reuse=False):
    """
    Encoder part of a vanilla autoencoder

    :param x: Input Image
    :return:
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('encoder'):
        x = tf.contrib.layers.fully_connected(x, 512, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 64, activation_fn=tf.nn.relu)

        mu = tf.contrib.layers.fully_connected(x, num_outputs=96, activation_fn=None)
        logsigma = tf.contrib.layers.fully_connected(x, num_outputs=96, activation_fn=None)
        sigma = tf.exp(logsigma)

        z = mu + tf.random_normal(tf.shape(sigma)) * sigma

        return z, mu, sigma

def discriminator(x, reuse=False):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("discriminator"):
        fc1 = tf.contrib.layers.fully_connected(x, 256, activation_fn=None)
        relu1 = leaky_relu(fc1, alpha=0.01)
        fc2 = tf.contrib.layers.fully_connected(relu1, 256, activation_fn=None)
        relu2 = leaky_relu(fc2, alpha=0.01)
        logits = tf.contrib.layers.fully_connected(relu2, 1, activation_fn=None)
        return logits

def generator(z, reuse=False):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("generator"):

        fc1 = tf.contrib.layers.fully_connected(z, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu)
        img = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.nn.tanh)
        return img
