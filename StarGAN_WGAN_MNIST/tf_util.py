import tensorflow as tf


def get_session():
    '''
    makes it so that tf session can allocate more gpu memory as needed
    :return:
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #allocates more gpu memory as needed during run time
    session = tf.Session(config=config)
    return session


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(x, tf.multiply(x, alpha))

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """

    return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)

def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar

    HINT: for the discriminator loss, you'll want to do the averaging separately for
    its two components, and then add them together (instead of averaging once at the very end).
    """
    G_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(logits_fake), predictions=logits_fake))
    D_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(logits_real), predictions=logits_real)) \
             + tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(logits_fake), predictions=logits_fake))

    return D_loss, G_loss

def get_solvers(learning_rate= 0.00005, beta1=0.5, beta2=0.999):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """

    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

    return D_solver, G_solver

def random_average(real_data, fake_data, batch_size):

    if len(real_data.shape) == 2:
        epsilon = tf.random.uniform((batch_size, 1), maxval=1)

    interpolated_image = epsilon * real_data + (1-epsilon) * fake_data

    return interpolated_image

def calc_gradient_penalty(Disc, real_data, fake_data):
    '''https://discuss.pytorch.org/t/how-to-implement-gradient-penalty-in-pytorch/1656/12'''

    if len(real_data.shape) == 2:
        epsilon = tf.random.uniform((tf.shape(real_data)[0], 1), maxval=1)

    """not yet tested on images"""
    if len(real_data.shape) == 4:
        epsilon = tf.random.uniform((tf.shape(real_data)[0], 1, 1, 1), maxval=1)

    interpolated_image = epsilon * real_data + (1-epsilon) * fake_data
    gradients = tf.gradients(Disc(interpolated_image)[0], [interpolated_image])
    gradient_penalty = tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    return gradient_penalty

def classification_loss(logits_real, class_array):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar

    HINT: for the discriminator loss, you'll want to do the averaging separately for
    its two components, and then add them together (instead of averaging once at the very end).
    """
    cls_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=class_array, logits=logits_real))

    return cls_loss


def recon_loss(input_image, recon_image):
    """
    Computes reconstruction loss
    :param input_image:
    :param recon_image:
    :return:
    """

    #recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_image, logits=recon_image), axis=[1,2,3])
    recon_loss = tf.reduce_sum(tf.losses.absolute_difference(labels=input_image, predictions=recon_image))
    return recon_loss
