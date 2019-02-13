
import tensorflow as tf
from tf_util import *

from tensorflow.contrib import layers


class double_conv_down():
    def __init__(self, channel_input, channel_output):

        self.channel_input = channel_input
        self.channel_output = channel_output

    def output(self, x):

        with tf.variable_scope('encoder_%d' %self.channel_input):
            x = tf.contrib.layers.conv2d(x, self.channel_output, kernel_size=3, stride=1, padding='valid', activation_fn=tf.nn.relu)
            skip_out = tf.contrib.layers.conv2d(x, self.channel_output, kernel_size=3, stride=1, padding='valid', activation_fn=tf.nn.relu)

            down_out = tf.contrib.layers.max_pool2d(skip_out, 2, stride=2, padding='valid')

            return down_out, skip_out


class double_conv_up():
    def __init__(self, channel_input, channel_output, final_output=None, activation=True):

        if final_output == None:
            final_output = channel_output

        self.channel_input = channel_input
        self.channel_output = channel_output
        self.final_output = final_output
        self.activation = activation


    def output(self, up_input, skip_input):

        with tf.variable_scope('decoder_%d' %self.channel_input):

            x = tf.concat([up_input, skip_input], axis=3)

            x = tf.contrib.layers.conv2d_transpose(x, self.channel_output, kernel_size=3, stride=1, padding='valid',
                                                   activation_fn=tf.nn.relu)
            if self.activation:
                x = tf.contrib.layers.conv2d_transpose(x, self.final_output, kernel_size=3, stride=1, padding='valid', activation_fn=tf.nn.relu)
            else:
                x = tf.contrib.layers.conv2d_transpose(x, self.final_output, kernel_size=3, stride=1, padding='valid')

        return x


class AutoEncoder_Unet():
    def __init__(self, in_channel, out_channel):

        self.U_down1 = double_conv_down(channel_input=in_channel, channel_output=16)
        self.U_down2 = double_conv_down(channel_input=16, channel_output=32)

        self.U_up1 = double_conv_up(64, 32, 16, True)
        self.U_up2 = double_conv_up(32, 8, out_channel, False)


    def model(self, x):


        with tf.variable_scope('mlp'):

            input_down1, input_skip1 = self.U_down1.output(x)
            input_down2, input_skip2 = self.U_down2.output(input_down1)

            #input down should be 32 * 4 * 4
            x = tf.contrib.layers.flatten(input_down2)

            mlp = layers.fully_connected(x, num_outputs=128, activation_fn=tf.nn.relu)
            mlp = layers.fully_connected(mlp, num_outputs=32*4*4, activation_fn=tf.nn.relu)

            x = tf.reshape(mlp, [-1, 4, 4, 32])

            x = tf.contrib.layers.conv2d_transpose(x, 32, kernel_size=2, stride=2, padding='valid')
            input_up1 = self.U_up1.output(x, input_skip2)

            input_up1 = tf.contrib.layers.conv2d_transpose(input_up1, 16, kernel_size=2, stride=2, padding='valid')
            input_up2 = self.U_up2.output(input_up1, input_skip1)

            output = tf.nn.tanh(input_up2)

            return output








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
