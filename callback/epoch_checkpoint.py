# -*- coding: utf-8 -*-

from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    '''
    At given epoch interval, save current model to disk.
    '''

    def __init__(self, outputPath, every = 5, start_at = 0):
        '''
        Arguments:
        - every: checkpoint interval
        - start_at: in case training was resumed from a different checkpoint,
        you can specify the current epoch number here.
        '''

        # call the parent instructor
        super().__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = start_at

    def on_epoch_end(self, epochs, logs = {}):
        '''
        At the end of each epoch,
        check to see if the model should be serialized to disk
        '''

        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                "epochs_{}.h5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite = True)

        # increment internal epoch counter
        self.intEpoch += 1
