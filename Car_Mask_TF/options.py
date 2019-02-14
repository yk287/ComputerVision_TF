import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=36, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0005, help='learning rate')

        #Image
        self.parser.add_argument('--image_shape', type=int, nargs='?', default=196, help='height of a square image')
        self.parser.add_argument('--channel', type=int, nargs='?', default=3, help='number of channels')

        #Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=20, help='total number of training episodes')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt

