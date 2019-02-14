
from util import *
import tf_util
from model import *
from image_to_gif import *

#using pytorch's dataloader
from data_loader import *
from torch.utils.data import DataLoader

from collections import deque

from options import options

options = options()
opts = options.parse()

session = tf.InteractiveSession()

unet = AutoEncoder_Unet(opts.channel, opts.channel)

with tf.name_scope('placeholders'):
    input_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel])
    target_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel])

with tf.name_scope('loss'):
    reconstruction = unet.model(input_image)
    loss = tf.reduce_mean(tf_util.recon_loss(target_image, reconstruction))

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
    train = optimizer.minimize(loss)

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

    resize = 256

    train_data = Pix2Pix_AB_Dataloader(train_folder, target_folder, size=resize, randomcrop=opts.image_shape)

    train_loader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=12)

    step= 0
    recon = []
    true = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []

    for epoch in range(num_epoch):
        # every show often, show a sample result
        for (minibatch, minbatch_y) in train_loader:

            _, loss_curr, recon_image = session.run([train_step, loss, reconstruction], feed_dict={input_image: minibatch, target_image: minbatch_y})
            last_100_loss_dq.append(loss_curr)
            last_100_loss.append(np.mean(last_100_loss_dq))

            if step % show_every == 0:
                '''for every show_every step, show reconstructed images from the training iteration'''

                recon_name = './img/recon_%s.png' % step
                true_name = './img/true_%s.png' % step

                recon.append(recon_name)
                true.append(true_name)

                show_images(recon_image[:opts.batch], opts, recon_name)
                show_images(minibatch[:opts.batch], opts, true_name)

            step += 1
            if step % print_every == 0:
                print('Epoch: {}, D: {:.4}'.format(epoch, loss_curr))
                raw_score_plotter(last_100_loss)

    raw_score_plotter(last_100_loss)
    image_to_gif('', recon, duration=0.5, gifname='recon')
    image_to_gif('', true, duration=0.5, gifname='true')

tf.global_variables_initializer().run()
train_UNET(train, loss, reconstruction, batch_size=opts.batch, num_epoch=opts.epoch)