import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=64, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.001, help='learning rate')

        #Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=50, help='total number of training episodes')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt

