import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=128, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.00005, help='learning rate')

        #Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=200, help='total number of training episodes')
        self.parser.add_argument('--d_steps', type=int, nargs='?', default=5, help='number of discriminator updates before updating the generator')
        self.parser.add_argument('--weight_cap', type=float, nargs='?', default=0.01, help='value used to cap gradient')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""

options = options()

opts = options.parse()
batch = opts.batch
"""