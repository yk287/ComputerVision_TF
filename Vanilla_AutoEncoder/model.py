
import tensorflow as tf
from tf_util import *

def encoder(x):
    """
    Encoder part of a vanilla autoencoder

    :param x: Input Image
    :return:
    """

    with tf.variable_scope("encoder"):
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 16, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 4, activation_fn=tf.nn.relu)

        return x

def decoder(x):
    """
    Decoder part of a vanilla autoencoder

    :param x:
    :return:
    """

    with tf.variable_scope("decoder"):
        x = tf.contrib.layers.fully_connected(x, 4, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 16, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = tf.contrib.layers.fully_connected(x, 784, activation_fn=tf.tanh)

        return x



def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        fc1 = tf.contrib.layers.fully_connected(x, 256, activation_fn=None)
        relu1 = leaky_relu(fc1, alpha=0.01)
        fc2 = tf.contrib.layers.fully_connected(relu1, 256, activation_fn=None)
        relu2 = leaky_relu(fc2, alpha=0.01)
        logits = tf.contrib.layers.fully_connected(relu2, 1, activation_fn=None)
        return logits

def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):

        fc1 = tf.contrib.layers.fully_connected(z, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu)
        img = tf.contrib.layers.fully_connected(fc2, 784, activation_fn=tf.nn.tanh)
        return img
