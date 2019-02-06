
from util import *
from model import *
from dataloader import *

from options import options

options = options()
opts = options.parse()

tf.reset_default_graph()

# number of images for each batch
batch_size = opts.batch
# our noise dimension

x = tf.placeholder(tf.float32, [None, 784])

with tf.variable_scope("") as scope:
    """
    End to End data pipeline
    """
    processed_image = preprocess_img(x)
    latent = encoder(processed_image)
    # Re-use discriminator weights on new inputs
    #scope.reuse_variables()
    reconstruct = decoder(latent)

# Get the list of variables for the discriminator and generator

E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decoder')

# get our solver
_, AE_solver = get_solvers(learning_rate=opts.lr)

# get our loss
recon_loss = recon_loss(processed_image, reconstruct)

# setup training steps
D_train_step = AE_solver.minimize(recon_loss, var_list=[E_vars, D_vars])

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')


# a giant helper function
def run_a_gan(sess, D_train_step, D_loss, show_every=666, print_every=1, batch_size=128, num_epoch=10):
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

            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})

            if step % show_every == 0:
                samples = sess.run(reconstruct, feed_dict={x: minibatch})
                fig = show_images(samples[:16])
                plt.show()
            step += 1
        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}'.format(epoch, D_loss_curr))

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, D_train_step, recon_loss, num_epoch=opts.epoch)
