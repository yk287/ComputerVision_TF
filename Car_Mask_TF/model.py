
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

        self.U_down1 = double_conv_down(channel_input=in_channel, channel_output=8)
        self.U_down2 = double_conv_down(channel_input=8, channel_output=16)
        self.U_down3 = double_conv_down(channel_input=16, channel_output=32)

        self.U_up1 = double_conv_up(64, 32, 16)
        self.U_up2 = double_conv_up(32, 16, 8)
        self.U_up3 = double_conv_up(16, 8, out_channel, False)

    def model(self, x):

        with tf.variable_scope('mlp'):

            input_down1, input_skip1 = self.U_down1.output(x)
            input_down2, input_skip2 = self.U_down2.output(input_down1)
            input_down3, input_skip3 = self.U_down3.output(input_down2)

            #input down should be 32 * 4 * 4
            x = tf.contrib.layers.flatten(input_down3)

            mlp = layers.fully_connected(x, num_outputs=32 * 21 * 21, activation_fn=tf.nn.relu)

            x = tf.reshape(mlp, [-1, 21, 21, 32])

            x = tf.contrib.layers.conv2d_transpose(x, 32, kernel_size=2, stride=2, padding='valid')
            input_up1 = self.U_up1.output(x, input_skip3)

            x = tf.contrib.layers.conv2d_transpose(input_up1, 16, kernel_size=2, stride=2, padding='valid')
            input_up2 = self.U_up2.output(x, input_skip2)

            x = tf.contrib.layers.conv2d_transpose(input_up2, 8, kernel_size=2, stride=2, padding='valid')
            input_up3 = self.U_up3.output(x, input_skip1)

            output = tf.nn.sigmoid(input_up3)

            return output
