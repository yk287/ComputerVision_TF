
import tensorflow as tf
from tf_util import *

def discriminator(x, n_classes):
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

        #logits for source (Discriminate between Fake/Real)
        logits_src = tf.contrib.layers.fully_connected(relu2, 1, activation_fn=None)

        # logits for source (Discriminate between Fake/Real)
        logits_cls = tf.contrib.layers.fully_connected(relu2, n_classes, activation_fn=None)

        return logits_src, logits_cls

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
