
from util import *
import tf_util
from model import *
from dataloader import *

from options import options

options = options()
opts = options.parse()

session = tf.InteractiveSession()


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 784])
    z = tf.placeholder(tf.float32, [None, 16])

with tf.name_scope('loss'):
    z, mu, sigma = encoder(x_true)
    logits, reconstruction = decoder(z)

    latent_losses = 0.5 * tf.reduce_sum(tf.square(mu) +
                                        tf.square(sigma) -
                                        tf.log(tf.square(sigma)) - 1,
                                        axis=1)

    reconstruction_losses = tf_util.recon_loss(x_true, logits)
    loss = tf.reduce_mean(reconstruction_losses + latent_losses)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
    train = optimizer.minimize(loss)

#mnist = input_data.read_data_sets('MNIST_data')


# a giant helper function
def run_a_gan(D_train_step, D_loss, reconstruct, show_every=666, print_every=1, batch_size=128, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    step= 0
    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in mnist:
            _, D_loss_curr = session.run([D_train_step, D_loss], feed_dict={x_true: minibatch})

            if step % show_every == 0:
                z_validate = np.random.randn(1, 16)
                generated = reconstruction.eval(feed_dict={z: z_validate}).squeeze()
                plt.figure('results')
                plt.imshow(generated.reshape([28, 28]), clim=[0, 1], cmap='bone')
                plt.pause(0.001)
            step += 1
        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}'.format(epoch, D_loss_curr))

tf.global_variables_initializer().run()
run_a_gan(train, loss, reconstruction, num_epoch=opts.epoch)