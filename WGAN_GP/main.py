
from util import *
from model import *
from dataloader import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from options import options

from image_to_gif import image_to_gif

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
data_size = tf.placeholder(tf.int32)
z = sample_noise(data_size, noise_dim)
#z = sample_noise(tf.shape(x)[0], noise_dim)
# generated images
G_sample = generator(z)

processed_img = preprocess_img(x)

interpolated_average = random_average(processed_img, G_sample, opts.batch)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(processed_img)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

    gradient_penalty = calc_gradient_penalty(discriminator, processed_img, G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr, beta1=opts.beta1, beta2=opts.beta2)

# get our loss

D_loss = - tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake) + opts.lamb * tf.reduce_mean(gradient_penalty)
G_loss = - tf.reduce_mean(logits_fake)

#clip the weights
#clip_D = [v.assign(tf.clip_by_value(v, -opts.weight_cap, opts.weight_cap)) for v in D_vars]

# setup training steps
#D_train_step = D_solver.apply_gradients(capped_gvs)
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, \
              show_every=500, print_every=1, batch_size=128, num_epoch=10):
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

    d_step = 0
    file_list = []

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epoch):
        for (minibatch, minbatch_y) in trainloader:
            minibatch = minibatch.view(-1, 28 * 28)

            "get the batch size: used due to uneven sample size"
            b_size = minibatch.shape[0]

            # every show often, show a sample result
            if d_step % show_every == 0:
                samples = sess.run(G_sample, feed_dict={data_size: b_size})
                file_name = './img/original_%s.png' % d_step
                file_list.append(file_name)
                show_images(samples[:25], opts, file_name)
                plt.show()

            d_step += 1
            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch, data_size: b_size})

            # clip the weights
            #sess.run(clip_D)

            if d_step % opts.d_steps == 0:
                _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={data_size: b_size})

        # print loss every so often.
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
    print('Final images')

    samples = sess.run(G_sample, feed_dict={data_size: 64})
    fig = show_images(samples[:25], opts)
    plt.show()

    image_to_gif('', file_list, duration=1, gifname='digits')


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step, batch_size=opts.batch, num_epoch=opts.epoch)
