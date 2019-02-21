
import tensorflow as tf
from tf_util import *

from tensorflow.contrib import layers


class conv_down():
    def __init__(self, channel_input, channel_out=0, kernel=3, stride=1, padding='valid', Norm=True, Dropout=0.0):

        if channel_out == 0:
            self.channel_output = channel_input * 2
        else:
            self.channel_output = channel_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.Norm = Norm
        self.Dropout = Dropout

    def output(self, x):

        with tf.variable_scope('conv_down_%d' % self.channel_output):

            x = tf.contrib.layers.conv2d(x, self.channel_output, kernel_size=self.kernel, stride=self.stride,
                                         padding=self.padding, activation_fn=None)

            if self.Norm:
                x = tf.contrib.layers.instance_norm(x)

            x = tf.nn.relu(x)

            if self.Dropout > 0:
                x = tf.nn.dropout(x, keep_prop=self.Dropout)

            return x

class conv_up():
    def __init__(self, channel_input, kernel=3, stride=1, padding='valid', Norm=True, Dropout=0.0):

        self.channel_output = channel_input // 2
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.Norm = Norm
        self.Dropout = Dropout

    def output(self, x):

        with tf.variable_scope('conv_up_%d' % self.channel_output):

            x = tf.contrib.layers.conv2d_transpose(x, self.channel_output, kernel_size=self.kernel, stride=self.stride,
                                         padding=self.padding, activation_fn=None)

            if self.Norm:
                x = tf.contrib.layers.instance_norm(x)

            x = tf.nn.relu(x)

            if self.Dropout > 0:
                x = tf.nn.dropout(x, keep_prop=self.Dropout)

            return x

class ResBlock():
    def __init__(self, input_channel, block_num):

        self.input_channel = input_channel
        self.block_num = block_num

    def output(self, input):
        with tf.variable_scope('resblock_%d' % self.block_num):
            paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            x = tf.pad(input, paddings, mode='REFLECT')
            x = tf.contrib.layers.conv2d(x, self.input_channel, 3, padding='valid', activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.relu(x)
            x = tf.pad(x, paddings, mode='REFLECT')
            x = tf.contrib.layers.conv2d(x, self.input_channel, 3, padding='valid', activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)

            x = x + input

            return x

class ResNet():
    def __init__(self, channel_in, channel_out, up_channel, name='resnet', n_downs=2, n_up=2, n_resnet=9):

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.up_channel = up_channel
        self.name = name

        self.conv_downs = []

        for i in range(n_downs):
            final_channel_ = up_channel * 2 ** i
            self.conv_downs.append(conv_down(channel_input=final_channel_))

        self.resnets = []
        final_channel_ *= 2
        for i in range(n_resnet):
            self.resnets.append(ResBlock(final_channel_, i))

        self.conv_ups = []
        for i in range(n_up):
            self.conv_ups.append(conv_up(channel_input=final_channel_))
            final_channel_ = final_channel_ // 2

    def output(self, x):
        with tf.variable_scope('%s' % self.name):

            paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]])
            #must specify each dimension unlike pytorch.

            x = tf.pad(x, paddings, mode='REFLECT')
            x = tf.contrib.layers.conv2d(x, self.up_channel, kernel_size=7, stride=1, padding='valid', activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.relu(x)

            for i in range(len(self.conv_downs)):
                x = self.conv_downs[i].output(x)

            for i in range(len(self.resnets)):
                x = self.resnets[i].output(x)

            for i in range(len(self.conv_ups)):
                x = self.conv_ups[i].output(x)

            x = tf.pad(x, paddings, mode='REFLECT')
            x = tf.contrib.layers.conv2d(x, self.channel_out, kernel_size=7, stride=1, padding='valid', activation_fn=None)
            x = tf.nn.sigmoid(x)

            return x

class PatchDiscrim():
    def __init__(self, channel_up, name):

        self.channel_up = channel_up
        self.name = name

    def discriminator(self, x):
        """Compute discriminator score for a batch of input images.

        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score
        for an image being real for each input image.
        """
        with tf.variable_scope('%s' % self.name):

            x = tf.contrib.layers.conv2d(x, self.channel_up, kernel_size=4, stride=2, padding='same', activation_fn=None)
            x = leaky_relu(x, alpha=0.01)
            x = tf.contrib.layers.conv2d(x, self.channel_up * 2, kernel_size=4, stride=2, padding='same', activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)
            x = leaky_relu(x, alpha=0.01)
            x = tf.contrib.layers.conv2d(x, self.channel_up * 3, kernel_size=4, stride=2, padding='same',
                                         activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)
            x = leaky_relu(x, alpha=0.01)
            x = tf.contrib.layers.conv2d(x, self.channel_up * 4, kernel_size=4, stride=2, padding='same',
                                         activation_fn=None)
            x = tf.contrib.layers.instance_norm(x)
            x = leaky_relu(x, alpha=0.01)
            x = tf.contrib.layers.conv2d(x, 1, kernel_size=4, padding='same', activation_fn=None)

            return x

