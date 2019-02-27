import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=64, help='batch size to be used')
        self.parser.add_argument('--lr', type=float, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--beta1', type=float, nargs='?', default=0.0, help='learning rate')
        self.parser.add_argument('--beta2', type=float, nargs='?', default=0.9, help='learning rate')

        #Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=50, help='total number of training episodes')
        self.parser.add_argument('--d_steps', type=int, nargs='?', default=5, help='number of discriminator updates before updating the generator')
        self.parser.add_argument('--weight_cap', type=float, nargs='?', default=0.01, help='value used to cap gradient')
        self.parser.add_argument('--lamb', type=float, nargs='?', default=10, help='lambda used gradient penalty part')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""

options = options()

opts = options.parse()
batch = opts.batch
"""