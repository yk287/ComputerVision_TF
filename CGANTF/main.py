
from util import *
import tf_util
from model import *
from dataloader import *
from image_to_gif import *

from options import options

options = options()
opts = options.parse()

session = tf.InteractiveSession()

with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 784])
    x_masked = tf.placeholder(tf.float32, [None, 784])
    z = tf.placeholder(tf.float32, [None, 16])

with tf.name_scope('loss'):
    x_true = x_true
    x_masked = x_masked

    z, mu, sigma = encoder(x_masked)
    logits, reconstruction = decoder(z)

    latent_losses = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, axis=1)

    reconstruction_losses = tf_util.recon_loss(x_true, logits)
    loss = tf.reduce_mean(reconstruction_losses + latent_losses)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
    train = optimizer.minimize(loss)

# a giant helper function
def train_VAE(train_step, loss, reconstruction, show_every=777, print_every=1, batch_size=128, num_epoch=10):
    """
    function that trains VAE.
    :param train_step: an op that defines what to do with the loss function (minimize or maximize)
    :param loss: an op that defines the loss function to be minimized
    :param reconstruction: an op that defines how to reconstruct a target image
    :param show_every: how often to show an image to gauge training progress
    :param print_every: how often to print loss
    :param batch_size: batch size of training samples
    :param num_epoch: how many times to iterate over the training samples
    :return:
    """
    # compute the number of iterations we need
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    step= 0

    masked = []
    recon = []
    true = []

    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in mnist:

            x_true_masked = black_out(minibatch.reshape([-1, 28, 28, 1]), width=3).reshape([-1, 784])

            _, loss_curr, recon_image, true_image, masked_image = session.run([train_step, loss, reconstruction, x_true, x_masked], feed_dict={x_true: minibatch, x_masked: x_true_masked})

            if step % show_every == 0:
                '''for every show_every step, show reconstructed images from the training iteration'''

                masked_name = './img/masked_%s.png' %step
                recon_name = './img/recon_%s.png' %step
                true_name = './img/true_%s.png' %step

                masked.append(masked_name)
                recon.append(recon_name)
                true.append(true_name)

                show_images(masked_image[:16], masked_name)
                plt.show()

                show_images(recon_image[:16], recon_name)
                plt.show()

                show_images(true_image[:16], true_name)
                plt.show()

            step += 1
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}'.format(epoch, loss_curr))

    image_to_gif('', masked, duration=1, gifname='masked')
    image_to_gif('', recon, duration=1, gifname='recon')
    image_to_gif('', true, duration=1, gifname='true')

tf.global_variables_initializer().run()
train_VAE(train, loss, reconstruction, batch_size=opts.batch, num_epoch=opts.epoch)