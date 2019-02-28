import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=25, help='batch size to be used')
        self.parser.add_argument('--lr', type=float, nargs='?', default=0.001, help='learning rate')
        self.parser.add_argument('--beta1', type=float, nargs='?', default=0.5, help='learning rate')
        self.parser.add_argument('--beta2', type=float, nargs='?', default=0.999, help='learning rate')

        #Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=150, help='total number of training episodes')
        self.parser.add_argument('--d_steps', type=int, nargs='?', default=5, help='number of discriminator updates before updating the generator')
        self.parser.add_argument('--weight_cap', type=float, nargs='?', default=0.01, help='value used to cap gradient')
        self.parser.add_argument('--lamb', type=float, nargs='?', default=10, help='lambda used gradient penalty part')

        #Image
        self.parser.add_argument('--resize', type=int, nargs='?', default=28, help='Image Resize size')
        self.parser.add_argument('--num_images', type=int, nargs='?', default=2, help='number of images in a single image file')
        self.parser.add_argument('--image_shape', type=int, nargs='?', default=28, help='height of a square image')
        self.parser.add_argument('--channel_in', type=int, nargs='?', default=1, help='number of input channels')
        self.parser.add_argument('--channel_out', type=int, nargs='?', default=1, help='number of output channels')
        self.parser.add_argument('--channel_up', type=int, nargs='?', default=8, help='initial channel increasing')

        #Training Options
        self.parser.add_argument('--model_name', type=str, nargs='?', default='cyclegan', help='name of the model to be saved')
        self.parser.add_argument('--resume', type=bool, nargs='?', default=False, help='resume training by loading saved checkpoints')
        self.parser.add_argument('--save_progress', type=bool, nargs='?', default=True,
                                 help='save training progress')
        self.parser.add_argument('--memory_len', type=int, nargs='?', default=50, help='size of memory from which the Discriminator is used to update')
        self.parser.add_argument('--show_every', type=int, nargs='?', default=15000, help='how often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=3000, help='how often to print losses')
        self.parser.add_argument('--const_epoch', type=int, nargs='?', default=100, help='number of epochs where LR is constant')
        self.parser.add_argument('--adaptive_epoch', type=int, nargs='?', default=100, help='number of epochs where LR changes')
        self.parser.add_argument('--num_classes', type=int, nargs='?', default=10, help='number of classes')
        self.parser.add_argument('--cycle_lambda', type=int, nargs='?', default=10, help='lambda for cycle')
        self.parser.add_argument('--cls_lambda', type=int, nargs='?', default=1, help='lambda for class')

        #MNIST specific
        self.parser.add_argument('--target_domain', type=int, nargs='?', default=8, help='target domain for StarGAN')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""

options = options()

opts = options.parse()
batch = opts.batch
"""