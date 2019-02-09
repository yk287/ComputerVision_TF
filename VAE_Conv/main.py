
from util import *
import tf_util
from model import *
from dataloader import *

from options import options

options = options()
opts = options.parse()

session = tf.InteractiveSession()

with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
    z = tf.placeholder(tf.float32, [None, 128])

with tf.name_scope('loss'):
    z, mu, sigma = encoder(preprocess_img(x_true))
    logits, reconstruction = decoder(z)

    latent_losses = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, axis=1)

    reconstruction_losses = tf_util.recon_loss(x_true, logits)
    loss = tf.reduce_mean(reconstruction_losses + latent_losses)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
    train = optimizer.minimize(loss)

# a giant helper function
def train_VAE(train_step, loss, reconstruction, show_every=666, print_every=1, batch_size=128, num_epoch=10):
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
    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in mnist:
            minibatch = minibatch.reshape([-1, 28, 28, 1])

            _, loss_curr, recon_image = session.run([train_step, loss, reconstruction], feed_dict={x_true: minibatch})

            if step % show_every == 0:
                '''for every show_every step, show reconstructed images from the training iteration'''
                show_images(recon_image[:25])
                plt.show()

            if step % show_every == 0:
                '''for every show_every step, show images generated from sampling from prior'''
                z_validate = np.random.randn(1, 128)
                generated = reconstruction.eval(feed_dict={z: z_validate}).squeeze()
                plt.figure('results')
                plt.imshow(generated.reshape([28, 28]), clim=[0, 1], cmap='bone')
                #plt.pause(0.001)

            step += 1
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}'.format(epoch, loss_curr))

tf.global_variables_initializer().run()
train_VAE(train, loss, reconstruction, batch_size=opts.batch, num_epoch=opts.epoch)