# -*- coding: utf-8 -*-

from os.path import join as path_join

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam

from callback.epoch_checkpoint import EpochCheckpoint
from callback.training_monitor_multi_classes import TrainingMonitorMultiClasses
from hdf5_helper.hdf5_generator_multi_classes import HDF5GeneratorMultiClasses
from image_generator.multi_output_data_generator import \
    MultiOutputDataGenerator
from preprocessor.mean_preprocessor import MeanPreprocessor
from preprocessor.divide_preprocessor import DividePreprocessor


class AgeGenderNetTrainer:
    '''
    Helper class to train and evaluate AgeGenderNet.
    '''

    def __init__(self, train_generator=None, validation_generator=None, test_generator=None):
        '''
        If train_generator is not specified,
        a generator which apply augmentation and both preprocessors will be created.
        Validation_generator and test_generator can also be created automatically,
        but we do not apply augmentation on validation_generator and test_generator.
        '''

        K.set_learning_phase(0) # must set this to enable training on ResNet
        mean_json_path = path_join('hdf5', 'mean.json')

        if train_generator is None:
            aug = MultiOutputDataGenerator(rotation_range=20, zoom_range=0.15,
                width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                horizontal_flip=True, fill_mode="nearest")

            training_path = path_join('hdf5', 'training.hdf5')
            self.train_generator = HDF5GeneratorMultiClasses(training_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)],
                augmentator=aug)
        else:
            self.train_generator = train_generator

        if validation_generator is None:
            validation_path = path_join('hdf5', 'validation.hdf5')
            self.validation_generator = HDF5GeneratorMultiClasses(validation_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)])
        else:
            self.validation_generator = validation_generator
        
        if test_generator is None:
            test_path = path_join('hdf5', 'test.hdf5')
            self.test_generator = HDF5GeneratorMultiClasses(test_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)])
        else:
            self.test_generator = test_generator

    def compile(self, model, lr=1e-3):
        '''
        Age guessing branch: mse (mean squared error) as loss function
        and mae (mean absolute error) as metric.
        Gender prediction branch: binary crossentropy as loss function
        and accuracy as metric.

        Return a compiled model, ready for training.
        '''

        losses = {
            'age_output': 'mse',
            'gender_output': 'binary_crossentropy',
        }

        lossWeights = {'age_output': 1.0, 'gender_output': 500.0}
        metrics = {'age_output': "mae", 'gender_output': "acc"}

        opt = Adam(lr=lr)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
            metrics=metrics)

        return model
    
    def build_tuning_model(self, warmup_path, lr=5e-4):
        '''
        Build AgeGenderNet model to be used in fine-tuning.
        *warmup_path* should be an AgeGenderNet model
        whose all conv_base layers are set to untrainabled.

        Load an AgeGenderNet model from disk,
        set trainable flags of all layers in age guessing
        and gender prediction branches to True,
        while keep trainable flags of all layers in
        common part as False.
        '''

        model = load_model(warmup_path)

        for layer in model.layers:
            if 'age_branch_' in layer.name or 'gender_branch' in layer.name:
                layer.trainable = True

        model = self.compile(model, lr=lr)
        return model

    def fit(self, model, epochs=50, start_epoch=0):
        '''
        Start training on *model*.
        This method can be used for both warming up and fine-tuning.
        If we are resuming training from a checkpoint,
        set *start_epoch* to the resumed epoch.
        '''

        training_monitor = TrainingMonitorMultiClasses(output_path='history', start_epoch=start_epoch)
        epoch_checkpoint = EpochCheckpoint('model', every=2, start_at=start_epoch)
        class_weights = {'age_output': None, 'gender_output': [1.632, 1]}

        model.fit_generator(
            self.train_generator.generator(),
            steps_per_epoch=self.train_generator.db_size // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator.generator(),
            validation_steps=self.validation_generator.db_size // self.validation_generator.batch_size,
            class_weight=class_weights,
            callbacks=[epoch_checkpoint, training_monitor]
        )
    
    def test(self, model_path):
        '''
        Load trained AgeGenderNet model and test it on test dataset.
        '''

        model = load_model(model_path)
        return model.evaluate_generator(
            self.test_generator.generator(), 
            steps=self.test_generator.db_size // self.test_generator.batch_size, 
        )
