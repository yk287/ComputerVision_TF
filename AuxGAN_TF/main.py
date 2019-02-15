
from util import *
from model import *
from dataloader import *

from options import options
from image_to_gif import *

from collections import deque

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
z = sample_noise(tf.shape(y)[0], noise_dim)
# generated images
G_sample = generator(tf.concat([z, y],  axis=1))

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real_src, logits_real_cls = discriminator(tf.concat([preprocess_img(x), y], axis=1), n_classes)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake_src, logits_fake_cls = discriminator(tf.concat([G_sample, y], axis=1), n_classes)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr)

# get our loss
D_loss_src, G_loss_src = gan_loss(logits_real_src, logits_fake_src)
D_loss_cls, G_loss_cls = classification_loss(logits_real_cls, logits_fake_cls, y)

D_loss = tf.reduce_mean(D_loss_src + D_loss_cls)
G_loss = tf.reduce_mean(G_loss_src + G_loss_cls)

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
    recon = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []

    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in mnist:

            minbatch_y = one_hot_encoder(minbatch_y, n_classes)

            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, y: minbatch_y})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={y: minbatch_y})

            last_100_loss_dq.append(G_loss_curr)
            last_100_loss.append(np.mean(last_100_loss_dq))
        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % show_every == 0:
            fake_labels = generate_fake_label(batch_size, n_classes=n_classes)

            samples = sess.run(G_sample, feed_dict={y: fake_labels})
            fig = show_images(samples[:25])
            plt.show()

        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
            raw_score_plotter(last_100_loss)

    raw_score_plotter(last_100_loss)
    print('Final images')

    for i in range(n_classes):
        fake_labels = generate_fake_label(batch_size, n_classes=n_classes, specific=i)
        samples = sess.run(G_sample, feed_dict={y: fake_labels})

        recon_name = './img/recon_%s.png' % i

        recon.append(recon_name)
        show_images(samples[:opts.batch], recon_name)

    image_to_gif('', recon, duration=1, gifname='recon')

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,batch_size=opts.batch, num_epoch=opts.epoch)
