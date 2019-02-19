
from util import *
import tf_util
from model import *
from image_to_gif import *

#using pytorch's dataloader
from data_loader import *
from image_folder import *
from torch.utils.data import DataLoader

from collections import deque

from options import options

options = options()
opts = options.parse()

session = tf.InteractiveSession()
D = PatchDiscrim(32)

unet = AutoEncoder_Unet(opts.channel, opts.channel)

with tf.name_scope('placeholders'):
    input_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel])
    target_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel])

with tf.name_scope('generate'):
    recon_image = unet.model(input_image)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = D.discriminator(tf.concat([target_image, input_image], axis=3))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = D.discriminator(tf.concat([recon_image, input_image], axis=3))

recon_loss = recon_loss(target_image, recon_image)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mlp')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=opts.lr)

# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)
G_loss = tf.reduce_mean(G_loss + opts.lamb * recon_loss)
loss = [D_loss, G_loss]
# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

train = [D_train_step, G_train_step]

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'mlp')

# a giant helper function
def train_UNET(train_step, loss, reconstruction, show_every=100, print_every=5, batch_size=128, num_epoch=10):
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
    image_dir = '/home/youngwook/Downloads/Carvana'
    folder_names = get_folders(image_dir)

    train_folder = folder_names[1]
    target_folder = folder_names[2]

    resize = 224

    train_data = Pix2Pix_AB_Dataloader(train_folder, target_folder, size=resize, randomcrop=opts.image_shape)

    train_loader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=12)

    step = 0
    recon = []
    true = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []

    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in train_loader:

            # run a batch of data through the network
            # logits= sess.run(logits_real, feed_dict={x:minibatch})
            _, D_loss_curr = session.run([train_step[0], loss[0]], feed_dict={input_image: minibatch, target_image: minbatch_y})
            _, G_loss_curr = session.run([train_step[1], loss[1]], feed_dict={input_image: minibatch, target_image: minbatch_y})

            last_100_loss_dq.append(G_loss_curr)
            last_100_loss.append(np.mean(last_100_loss_dq))

            step += 1

            if step % show_every == 0:
                '''for every show_every step, show reconstructed images from the training iteration'''

                recon_name = './img/recon_%s.png' % step
                true_name = './img/true_%s.png' % step

                #translate the image
                recon_images = session.run(recon_image, feed_dict={input_image: minibatch})

                recon.append(recon_name)
                true.append(true_name)

                show_images(recon_images[:opts.batch], opts, recon_name)
                show_images(minibatch[:opts.batch], opts, true_name)

            if step % print_every == 0:
                print('Epoch: {}, D: {:.4}'.format(epoch, G_loss_curr))
                raw_score_plotter(last_100_loss)

    raw_score_plotter(last_100_loss)
    image_to_gif('', recon, duration=0.5, gifname='recon')
    image_to_gif('', true, duration=0.5, gifname='true')

tf.global_variables_initializer().run()
train_UNET(train, loss, recon_image, batch_size=opts.batch, num_epoch=opts.epoch)