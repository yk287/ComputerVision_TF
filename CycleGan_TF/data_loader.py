#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from image_folder import get_images

import torchvision.transforms.functional as TVF

class AB_Combined_ImageLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    This works so that an image of 2 images A, B that are concatenated horizontally can be split up with the right transformations
    """

    def __init__(self, img_folder, shuffle=True, train=True, num_images=2, size=256, randomcrop=196, hflip=0.5, vflip=0.5):
        '''

        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.img_folder = img_folder
        self.size = size
        self.num_images = num_images #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.shuffle = shuffle
        self.train = train

        self.file_names = get_images(img_folder)
        #self.target_names = get_images(target_folder)

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        #TODO: Add some randomization to shuffle paried data.

        combined_image = Image.open(self.file_names[index]).convert('RGB')

        resize = transforms.Resize(size=(self.size, 2 * self.size))
        combined_image = resize(combined_image)

        combined_image = crop_PIL(combined_image, self.num_images, crop_size=self.randomCrop, random=False)

        left_image = combined_image[0]
        right_image = combined_image[1]

        if self.train:

            if random.random() >= self.hflip:
                left_image = TVF.hflip(left_image)
                right_image = TVF.hflip(right_image)

            if random.random() >= self.vflip:
                left_image = TVF.vflip(left_image)
                right_image = TVF.vflip(right_image)

        left_image = np.array(left_image) / 255.0
        right_image = np.array(right_image) / 255.0

        return left_image, right_image

def crop_PIL(image, num_image, crop_size=0, random=False):

    #assumes channel X Height X Width

    w = image.size[0]
    h = image.size[1]

    assert w % num_image == 0, "The Width is not a multiple of the number of splits"
    w_cutoff = w // num_image

    w_crop = 0
    h_crop = 0

    if random != False:
        w_crop = np.random.randint(0, w_cutoff - crop_size)
        h_crop = np.random.randint(0, h - crop_size)

    image_list = []
    for i in range(num_image):
        starting_point = i * w_cutoff + w_crop
        image_list.append(image.crop((starting_point, h_crop, crop_size + starting_point, crop_size + h_crop)))

    return image_list
