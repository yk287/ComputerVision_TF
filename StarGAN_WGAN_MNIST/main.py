
from util import *
from model import *
from dataloader import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from options import options

from image_to_gif import image_to_gif

from collections import deque

options = options()
opts = options.parse()

tf.reset_default_graph()

session = tf.InteractiveSession()

# number of images for each batch
batch_size = opts.batch
# our noise dimension

# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])

label = tf.placeholder(tf.float32, [None, opts.num_classes])
label_domain = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.num_classes])

target_label = tf.placeholder(tf.float32, [None, opts.num_classes])
target_domain = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.num_classes])

Discrim = PatchDiscrim(32, name='discriminator', opts=opts)
generator = ResNet(opts.channel_in, opts.channel_out, up_channel=opts.channel_up, name='generator')

processed_img = preprocess_img(x)

with tf.variable_scope("") as scope:
    G_sample = generator.output(tf.concat([processed_img, target_domain],  axis=3))

    scope.reuse_variables()
    recon_cycle = generator.output(tf.concat([G_sample, label_domain],  axis=3))

recon_loss = recon_loss(processed_img, recon_cycle) * opts.cycle_lambda

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real_src, logits_real_cls = Discrim.discriminator(processed_img)

    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake_src, logits_fake_cls = Discrim.discriminator(G_sample)
    gradient_penalty = calc_gradient_penalty(Discrim.discriminator, processed_img, G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr, beta1=opts.beta1, beta2=opts.beta2)

# get our loss
D_loss_cls = classification_loss(logits_real_cls, label)
G_loss_cls = classification_loss(logits_fake_cls, target_label)

D_loss = - tf.reduce_mean(logits_real_src) + tf.reduce_mean(logits_fake_src) + opts.lamb * tf.reduce_mean(gradient_penalty) + opts.cls_lambda * D_loss_cls
G_loss = - tf.reduce_mean(logits_fake_src) + opts.cls_lambda * G_loss_cls + opts.cycle_lambda * recon_loss

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, \
              show_every=1, print_every=1, batch_size=128, num_epoch=10):
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

    # compute the number of iterations we need
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    recon = []
    file_list = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []
    step = 0
    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, og_label) in mnist:
            #
            og_label = one_hot_encoder(og_label, opts.num_classes)
            label_image = expand_spatially(og_label, opts.image_shape)

            target_labels = np.ones((og_label.shape[0], 1), dtype=int) * opts.target_domain
            target_labels = one_hot_encoder(target_labels, opts.num_classes)
            target_label_image = expand_spatially(target_labels, opts.image_shape)

            #transform the data to be the right shape
            minibatch = minibatch.reshape((-1, opts.image_shape, opts.image_shape, opts.channel_out))

            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, label: og_label, target_domain: target_label_image})
            step += 1

            if step % opts.d_steps == 0:
                _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: minibatch, label: og_label, label_domain: label_image, target_label: target_labels, target_domain: target_label_image})
                last_100_loss_dq.append(G_loss_curr)
                last_100_loss.append(np.mean(last_100_loss_dq))

            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
        if epoch % show_every == 0:

            samples = sess.run(G_sample, feed_dict={x: minibatch, target_domain: target_label_image})

            recon_name = './img/recon_%s.png' % epoch
            recon.append(recon_name)
            show_images(samples[:opts.batch], recon_name)
            plt.show()

            file_name = './img/original_%s.png' % epoch
            file_list.append(file_name)
            show_images(minibatch[:opts.batch], file_name)
            plt.show()

        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
            raw_score_plotter(last_100_loss)

    raw_score_plotter(last_100_loss)
    print('Final images')

    image_to_gif('', recon, duration=1, gifname='recon')
    image_to_gif('', file_list, duration=1, gifname='original')

tf.global_variables_initializer().run()
run_a_gan(session,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step, batch_size=opts.batch, num_epoch=opts.epoch)
