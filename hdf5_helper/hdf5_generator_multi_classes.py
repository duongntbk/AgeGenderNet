# -*- coding: utf-8 -*-

import h5py
import numpy as np
from keras.utils import to_categorical


class HDF5GeneratorMultiClasses:
    '''
    Generator to generate training/validation/test data from hdf5 file.
    Each data point has 2 labels, one for gender and one for age.
    '''

    def __init__(self, db_path, is_categorical, batch_size=128, preprocessors=None, augmentator=None):
        self.batch_size = batch_size
        self.label_counts = 2
        self.is_categorical = is_categorical
        self.preprocessors = preprocessors
        self.augmentator = augmentator

        self.db = h5py.File(db_path)
        self.db_size = self.db['labels_0'].shape[0]

    def generator(self, max_epochs=np.inf):
        '''
        Create generator object to load and process data from hdf5 file.
        '''

        epochs = 0

        # If max_epochs is set, only generate data for maximum "max_epochs" times.
        while epochs < max_epochs:
            for i in np.arange(0, self.db_size, self.batch_size):
                images = self.db['data'][i:i+self.batch_size]
                labels_dict = {}
                labels_dict['age_output'] = self.db['labels_0'][i:i+self.batch_size]
                labels_dict['gender_output'] = self.db['labels_1'][i:i+self.batch_size]

                # Use one-hot encoding on labels if needed
                if self.is_categorical:
                    labels_dict['age_output'] = to_categorical(labels_dict['age_output'])
                    labels_dict['gender_output'] = to_categorical(labels_dict['gender_output'])

                # Pass data through all preprocessor object to normalize data
                if self.preprocessors is not None:
                    for preprocessor in self.preprocessors:
                        images = preprocessor.process(images)

                # Perform data augmentation if needed
                if self.augmentator is not None:
                    images, labels_dict = next(self.augmentator.flow(images, labels_dict, batch_size=self.batch_size))
                
                yield images, labels_dict
            
            epochs += 1
    
    def close(self):
        self.db.close()
