
import tensorflow as tf
from tf_util import *

def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):

        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.contrib.layers.conv2d(x, 32, kernel_size=5, activation_fn=None)
        x = leaky_relu(x, alpha=0.01)
        x = tf.contrib.layers.max_pool2d(x, kernel_size=2, stride=2)
        x = tf.contrib.layers.conv2d(x, 64, kernel_size=5, activation_fn=None)
        x = leaky_relu(x, alpha=0.01)
        x = tf.contrib.layers.max_pool2d(x, kernel_size=2, stride=2)

        flatten1 = tf.contrib.layers.flatten(x)
        fc = tf.contrib.layers.fully_connected(flatten1, 4 * 4 * 64)
        fc = leaky_relu(fc, alpha=0.01)
        logits = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)
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
        fc1 = tf.layers.batch_normalization(fc1)
        fc2 = tf.contrib.layers.fully_connected(fc1, 7 * 7 * 128, activation_fn=tf.nn.relu)
        fc2 = tf.layers.batch_normalization(fc2)

        img = tf.reshape(fc2, [-1, 7, 7, 128])

        conv1 = tf.layers.conv2d_transpose(img, 64, 4, 2, padding='same', activation=tf.nn.relu)
        conv1 = tf.layers.batch_normalization(conv1)
        conv2 = tf.layers.conv2d_transpose(conv1, 1, 4, 2, padding='same', activation=tf.tanh)

        img=tf.reshape(conv2, [-1, 784])
        return img
