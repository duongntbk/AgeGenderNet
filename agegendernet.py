# -*- coding: utf-8 -*-

import copy

from keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from keras.models import Model


class AgeGenderNet:
    '''
    This class is used to build AgeGenderNet model.
    It does not perform any training or evaluation.
    AgeGenderNet is based on VGG19 architecture,
    with separate branches for age guessing and gender prediction.
    Result of gender prediction is integrated into age guessing branch
    to increase accuracy.
    '''

    @staticmethod
    def build_root(conv_base, inputs, split_from_top=4):
        '''
        Build the common part of both age guessing branch and
        gender prediction branch.
        The weights in common part are loaded directly from model
        pre-trained on ImageNet dataset and will not be modified.
        '''

        base = copy.deepcopy(conv_base)
        base_depth = len(base.layers)
        x = inputs
        for layer in base.layers[1:base_depth-split_from_top]:
            layer.trainable = False
            x = layer(x)

        return x

    @staticmethod
    def build_gender_branch(conv_base, root, split_from_top=4, dropout=0.4):
        '''
        Build the gender prediction branch.
        While performing fine-tuning, we will update the weights in this branch
        but at first we set all parameters to trainable==False so that
        we can warm up the fully conntected layers first.
        '''

        base = copy.deepcopy(conv_base)
        base_depth = len(base.layers)
        x = root
        for layer in base.layers[base_depth-split_from_top:base_depth]:
            name = layer.name
            layer.name = "gender_branch_" + name
            layer.trainable = False
            x = layer(x)

        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='sigmoid', name='gender_output')(x)

        return x

    @staticmethod
    def build_age_branch(conv_base, root, gender_branch, split_from_top=4, dropout=0.4):
        '''
        The age guessing branch.
        Because the features to guess age for female and male are different,
        we concatenate gender information into the fully connected layers of age guessing.
        If gender prediction is accurate enough,
        this should increase age guessing accuracy.
        While performing fine-tuning, we will update the weights in this branch
        but at first we set all parameters to trainable==False so that
        we can warm up the fully conntected layers first.
        '''

        base = copy.deepcopy(conv_base)
        base_depth = len(base.layers)
        x = root
        for layer in base.layers[base_depth-split_from_top:base_depth]:
            name = layer.name
            layer.name = "age_branch_" + name
            layer.trainable = False
            x = layer(x)

        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Concatenate()([x, gender_branch])
        x = Dense(1, name='age_output')(x)

        return x

    @staticmethod
    def build(conv_base, split_from_top=4, age_dropout=0.4, gender_dropout=0.2):
        '''
        Build AgeGenderNet model.
        All layers in conv_base will be set to trainable=False at first.
        When perform fine-tuning, we need to change trainable property.
        '''

        input_shape = conv_base.layers[0].input_shape[1:]
        inputs = Input(shape=input_shape, name='root_input')
        root = AgeGenderNet.build_root(conv_base, inputs, split_from_top=split_from_top)
        gender_branch = AgeGenderNet.build_gender_branch(conv_base, root,
                            split_from_top=split_from_top, dropout=age_dropout)
        age_branch = AgeGenderNet.build_age_branch(conv_base, root, gender_branch,
                            split_from_top=split_from_top, dropout=gender_dropout)
        return Model(inputs, [age_branch, gender_branch])
