
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def one_hot_encoder(index_array, n_classes = 10):
    '''
    One hot encoder that takes in an array of labels (ex.  array([1, 0, 3])) and returns
    a n-dimensional array that is one hot encoded
    :param index_array (array of ints): an array that holds class labels
    :param n_classes (int): number of classes
    :return:
    '''

    batch_size = index_array.shape[0]

    b = np.zeros((batch_size, n_classes))
    b[np.arange(batch_size), index_array] = 1
    return np.float32(b)

def categorical_label_generator(batch_size=128, n_classes=10):
    '''
    Function that returns an array with a label between [0, 10] for an element with size batch_size
    :param batch_size (int): length of an array
    :param n_classes (int): total number of classes
    :return:
    '''

    array = np.random.choice(n_classes, batch_size)

    return array

def generate_fake_label(batch_size=128, n_classes=10):

    fake_labels = categorical_label_generator(batch_size, n_classes=n_classes)
    fake_labels = one_hot_encoder(fake_labels, n_classes)

    return fake_labels


