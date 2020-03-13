# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitorMultiClasses(BaseLogger):
    '''
    Log trainning loss/accuracy and validation loss/accuracy to json and png file.
    '''

    def __init__(self, output_path, start_epoch=0):
        super(TrainingMonitorMultiClasses, self).__init__()
        self.output_path = output_path
        self.json_file_path = os.path.join(output_path, 'monitor.json')
        self.start_epoch = start_epoch

    def on_train_begin(self, logs=None):
        '''
        Initialize necessary properties when training starts.
        '''

        self.history = {}

        # If json file already existed, trim history up to *start_epoch*
        if self.json_file_path is not None:
            if os.path.exists(self.json_file_path):
                with open(self.json_file_path, 'r') as f:
                    self.history = json.loads(f.read())
                
                # Trim history to start_epoch
                if self.start_epoch > 0:
                    for key in self.history.keys():
                        self.history[key] = self.history[key][:self.start_epoch]

    def on_epoch_end(self, epoch, logs={}):
        '''
        At the end of each epoch, append current training loss/accuracy and
        validation loss/accuracy to json file.
        Also update png graph.
        '''

        for (key, val) in logs.items():
            hit = self.history.get(key, []) # return empty array if this key does not exist

            if type(val) is np.float32:
                val = val.astype('float64')

            hit.append(val)
            self.history[key] = hit

        if self.json_file_path is not None:
            with open(self.json_file_path, 'w') as f:
                f.write(json.dumps(self.history))

        if len(self.history['loss']) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.history['loss']))
            plt.style.use('ggplot')
            plt.figure()

            plt.plot(N, np.log(self.history['age_output_loss']), label='age_output_loss')
            plt.plot(N, np.log(self.history['val_age_output_loss']), label='val_age_output_loss')
            plt.plot(N, self.history['age_output_mae'], label='age_output_mae')
            plt.plot(N, self.history['val_age_output_mae'], label='val_age_output_mae')
            plt.title('Training Loss and MAE  [Epoch {}]'.format(
                len(self.history['loss'])))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/MAE')
            plt.legend()

            # save the age figure
            plt.savefig(os.path.join(self.output_path, 'age.png'))
            plt.close()

            plt.plot(N, self.history['gender_output_loss'], label='gender_output_loss')
            plt.plot(N, self.history['val_gender_output_loss'], label='val_gender_output_loss')
            plt.plot(N, self.history['gender_output_acc'], label='gender_output_acc')
            plt.plot(N, self.history['val_gender_output_acc'], label='val_gender_output_acc')
            plt.title('Training Loss and Accuracy  [Epoch {}]'.format(
                len(self.history['loss'])))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()

            # save the age figure
            plt.savefig(os.path.join(self.output_path, 'gender.png'))
            plt.close()
