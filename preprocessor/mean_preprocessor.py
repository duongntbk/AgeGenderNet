# -*- coding: utf-8 -*-

import numpy as np
import json

class MeanPreprocessor:
    '''
    For each channel in image, subtract the corresponding
    mean values of R, G or B channels of training data
    to zero-center pixel's values.
    Mean values of R, G and B channels are loaded from json file.
    '''

    def __init__(self, mean_path='hdf5/mean.json'):
        with open(mean_path, 'r') as f:
            text = f.read()
        
        json_val = json.loads(text)
        r, g, b = json_val['R'], json_val['G'], json_val['B']
        self.mean = np.array([r, g, b])

    def process(self, images):
        return images - self.mean