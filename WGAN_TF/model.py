
import tensorflow as tf
from tf_util import *

import keras

def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        x = tf.contrib.layers.fully_connected(x, 512, activation_fn=None)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=None)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=None)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        return x

def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):

        x = tf.contrib.layers.fully_connected(z, 128, activation_fn=None)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=None)
        x = tf.layers.batch_normalization(x, axis=0)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 512, activation_fn=None)
        x = tf.layers.batch_normalization(x, axis=0)
        x = leaky_relu(x, alpha=0.2)
        x = tf.contrib.layers.fully_connected(x, 1024, activation_fn=None)
        x = tf.layers.batch_normalization(x, axis=0)
        x = leaky_relu(x, alpha=0.2)

        img = tf.contrib.layers.fully_connected(x, 784, activation_fn=tf.nn.tanh)

        return img
