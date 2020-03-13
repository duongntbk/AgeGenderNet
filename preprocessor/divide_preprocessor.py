# -*- coding: utf-8 -*-

class DividePreprocessor:
    '''
    Divide the value of each pixel by *rate* to scale back input.
    '''

    def __init__(self, rate):
        self.rate = rate

    def process(self, images):
        return images / self.rate
