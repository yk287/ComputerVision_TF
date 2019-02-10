
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
noise_dim = 96

# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
# random noise fed into our generator
prior = sample_gauss_noise(batch_size, noise_dim)
# generated images

z, _, _ = encoder(preprocess_img(x))
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(prior)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(z)

loss = recon_loss(x, G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
E_Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr)
optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(loss, var_list=[G_vars, E_Vars])
E_train_step = optimizer.minimize(G_loss, var_list=E_Vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, \
              show_every=1000, print_every=1, batch_size=128, num_epoch=10):
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
    for epoch in range(num_epoch):
        steps = 0
        # every show often, show a sample result
        for (minibatch, minbatch_y) in mnist:
            # run a batch of data through the network

            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, recon_loss = sess.run([E_train_step, loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: minibatch})

            if steps % show_every == 0:
                samples = sess.run(G_sample, feed_dict={x: minibatch})
                fig = show_images(samples[:25])
                plt.show()
                fig = show_images(minibatch[:25])
                plt.show()


            steps += 1
        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step, batch_size=opts.batch, num_epoch=opts.epoch)
