
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
n_classes = 10

# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])
# random noise fed into our generator
z = sample_noise(batch_size, noise_dim)
fake_y = tf.placeholder(tf.float32, [None, n_classes])
# generated images

G_sample = generator(tf.concat([z, fake_y],  axis=1))

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(tf.concat([preprocess_img(x), y], axis=1))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(tf.concat([G_sample, fake_y], axis=1))

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr)

# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')


# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, \
              show_every=2, print_every=1, batch_size=128, num_epoch=10):
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
        # every show often, show a sample result

        for (minibatch, minbatch_y) in mnist:

            minbatch_y = one_hot_encoder(minbatch_y, n_classes)
            fake_labels = generate_fake_label(opts.batch, n_classes=n_classes)

            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, y: minbatch_y, fake_y: fake_labels})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={fake_y: fake_labels})

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % show_every == 0:

            fake_labels = generate_fake_label(opts.batch, n_classes=n_classes)

            samples = sess.run(G_sample, feed_dict={fake_y: fake_labels})
            fig = show_images(samples[:25])
            plt.show()
            print()

        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
    print('Final images')

    fake_labels = generate_fake_label(opts.batch, n_classes=n_classes)
    samples = sess.run(G_sample, feed_dict={fake_y: fake_labels})

    fig = show_images(samples[:25])
    plt.show()

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,batch_size=opts.batch, num_epoch=opts.epoch)
