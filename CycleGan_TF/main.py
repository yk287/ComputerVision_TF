
import util
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

input_memory = deque(maxlen=opts.memory_len)
target_memory = deque(maxlen=opts.memory_len)

session = tf.InteractiveSession()

checkpoint_dir = './model'
Input_D = PatchDiscrim(32, name='input_discriminator')
Target_D = PatchDiscrim(32, name='target_discriminator')

ResNet_targetPred = ResNet(opts.channel_in, opts.channel_out, up_channel=opts.channel_up, name='resnet_target')
ResNet_inputPred = ResNet(opts.channel_in, opts.channel_out, up_channel=opts.channel_up, name='resnet_input')

with tf.name_scope('placeholders'):
    input_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])
    target_image = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])

    input_image_pred = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])
    target_image_pred = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])

    input_replay = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])
    target_replay = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])

    #Used for adaptive LR decay
    adaptive_lr = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope("") as scope:
    target_image_prediction = ResNet_targetPred.output(input_image)
    input_image_prediction = ResNet_inputPred.output(target_image)

    scope.reuse_variables()

    target_image_cycle = ResNet_targetPred.output(input_image_pred)
    input_image_cycle = ResNet_inputPred.output(target_image_pred)

with tf.variable_scope("") as scope:
    target_logits_real = Target_D.discriminator(target_image)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    target_logits_fake = Target_D.discriminator(target_replay)

with tf.variable_scope("") as scope:
    input_logits_real = Input_D.discriminator(input_image)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    input_logits_fake = Input_D.discriminator(input_replay)

recon_image = [target_image_cycle, input_image_cycle]

# Cycle Loss
ITI_cycle_loss = recon_loss(input_image, input_image_cycle)
TIT_cycle_loss = recon_loss(target_image, target_image_cycle)
cycle_loss = ITI_cycle_loss + TIT_cycle_loss

# Get the list of variables for the discriminator and generator
D_vars_input = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'input_discriminator')
D_vars_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_discriminator')
G_vars_input = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_input')
G_vars_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_target')

# get our solver
D_solver_input, G_solver = get_solvers(learning_rate=adaptive_lr)
D_solver_target, _ = get_solvers(learning_rate=adaptive_lr)

# get our loss
target_D_loss, target_G_loss = gan_loss(target_logits_real, target_logits_fake)
input_D_loss, input_G_loss = gan_loss(input_logits_real, input_logits_fake)

D_loss = [input_D_loss, target_D_loss]
G_loss = tf.reduce_mean(target_G_loss + input_G_loss + opts.lamb * cycle_loss)
#G_loss = tf.reduce_mean(target_G_loss + input_G_loss)# + opts.lamb * cycle_loss)

loss = [D_loss, G_loss]
# setup training steps
D_input_train = D_solver_input.minimize(input_D_loss, var_list=D_vars_input)
D_target_train = D_solver_input.minimize(target_D_loss, var_list=D_vars_target)

G_vars = [G_vars_input, G_vars_target]
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

train = [D_input_train, D_target_train, G_train_step]

# a giant helper function
def train_CycleGan(train_step, loss, reconstruction, show_every=opts.show_every, print_every=opts.print_every, batch_size=128, num_epoch=10):
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

    image_dir = '/home/youngwook/Downloads/edges2shoes'
    folder_names = get_folders(image_dir)

    train_folder = folder_names[2]
    val_folder = folder_names[1]

    train_data = AB_Combined_ImageLoader(train_folder, size=opts.resize, num_images=opts.num_images, randomcrop=opts.image_shape)
    train_loader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=12)

    step = 0
    target_pred_list = []
    input_pred_list = []
    input_true_list = []
    target_true_list = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []

    checkpoint_dir = './model'
    saver = tf.train.Saver()

    if opts.resume:
        #print('Loading Saved Checkpoint')
        tf_util.load_session(checkpoint_dir, saver, session, model_name=opts.model_name)

    for epoch in range(num_epoch):
        # every show often, show a sample result

        lr = util.linear_LR(epoch, opts)

        for (minibatch, minbatch_y) in train_loader:
            # run a batch of data through the network
            # logits= sess.run(logits_real, feed_dict={x:minibatch})

            target_pred, input_pred = session.run([target_image_prediction, input_image_prediction], feed_dict={input_image: minibatch, target_image: minbatch_y, adaptive_lr: lr})

            input_memory.append(input_pred)
            target_memory.append(target_pred)

            target_replay_images = np.vstack(target_memory)
            input_replay_images = np.vstack(input_memory)

            #train the Generator
            _, G_loss_curr = session.run([train_step[2], loss[1]], feed_dict={input_image: minibatch, target_image: minbatch_y, input_replay: input_replay_images, target_replay: target_replay_images, input_image_pred : input_pred, target_image_pred: target_pred, adaptive_lr: lr})

            #train the discriminator
            _, D_loss_curr = session.run([train_step[0], loss[0][0]], feed_dict={input_image: minibatch, input_replay: input_replay_images, adaptive_lr: lr})
            _, D_loss_curr = session.run([train_step[1], loss[0][1]], feed_dict={target_image: minbatch_y,  target_replay: target_replay_images, adaptive_lr: lr})

            last_100_loss_dq.append(G_loss_curr)
            last_100_loss.append(np.mean(last_100_loss_dq))

            step += 1
            if step % show_every == 0:
                '''for every show_every step, show reconstructed images from the training iteration'''

                target_name = './img/target_pred_%s.png' % step
                input_name = './img/input_pred_%s.png' % step
                input_true_name = './img/true_input_%s.png' % step
                target_true_name = './img/true_target_%s.png' % step

                #translate the image
                target_pred, input_pred = session.run([target_image_prediction, input_image_prediction],
                                                      feed_dict={input_image: minibatch, target_image: minbatch_y})

                target_pred_list.append(target_name)
                input_pred_list.append(input_name)
                input_true_list.append(input_true_name)
                target_true_list.append(target_true_name)

                util.show_images(target_pred[:opts.batch], opts, target_name)
                util.plt.show()
                util.show_images(minbatch_y[:opts.batch], opts, target_true_name)
                util.plt.show()

                util.show_images(input_pred[:opts.batch], opts, input_name)
                util.plt.show()
                util.show_images(minibatch[:opts.batch], opts, input_true_name)
                util.plt.show()

            if step % print_every == 0:
                print('Epoch: {}, D: {:.4}'.format(epoch, G_loss_curr))
                util.raw_score_plotter(last_100_loss)

        #save the model after every epoch
        if opts.save_progress:
            tf_util.save_session(saver, session, checkpoint_dir, epoch, model_name=opts.model_name)

    util.raw_score_plotter(last_100_loss)

    image_to_gif('', target_pred_list, duration=0.5, gifname='target_pred')
    image_to_gif('', input_pred_list, duration=0.5, gifname='input_pred')
    image_to_gif('', input_true_list, duration=0.5, gifname='input_true')
    image_to_gif('', target_true_list, duration=0.5, gifname='target_true')

tf.global_variables_initializer().run()
train_CycleGan(train, loss, recon_image, batch_size=opts.batch, num_epoch=opts.epoch)