# -*- coding: utf-8 -*-

'''
Helper methods to read AFAD dataset from disk,
convert to tensor and write tensor to hdf5 file.
Because our training set is quite big,
we perform training on hdf5 file to minimize disk read.
'''

import json
import os
from math import floor

import cv2
import numpy as np
import progressbar
from imutils import paths
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hdf5_helper.hdf5_writer_multi_classes import HDF5WriterMultiClasses


def split_by_gender_age(data_dir):
    '''
    Label each image as either male or female and by age.
    All female images are stored in folder named "112",
    while male images are stored in folder named "111".
    The folder where each image is stored is the age.

    Returns: a tuple of 3 arrays of the same length.
    The first array is the path to each image,
    while the second array is gender, the third array is age.
    '''

    data_paths = list(paths.list_images(data_dir))

    if len(data_paths) == 0:
        return None, None

    age_labels = [int(n.split(os.path.sep)[-3]) for n in data_paths]

    gender_labels = ['0' if n.split(os.path.sep)[-2] == '111' else '1' for n in data_paths]

    # Because folder name is string, we need to encode it before using it as label.
    le = LabelEncoder()
    gender_labels = le.fit_transform(gender_labels)

    return data_paths, age_labels, gender_labels

def write_data_to_hdf5(data_dir, split_method, output_dir, set_split=0.2, channels_first=False):
    '''
    Read images data, convert to tensor and write both tensor data and label to one hdf5 file.
    Image can be either channels first or channels_last. 
    '''

    data_paths, age_labels, gender_labels = split_method(data_dir)

    if data_paths is None:
        print('Cannot find any image in {0}'.format(data_dir))
        return

    test_size = floor(len(data_paths) * set_split)
    train_val_paths, test_paths, train_val_age_labels, test_age_lables, train_val_gender_labels, test_gender_lables = train_test_split(
        data_paths, age_labels, gender_labels, test_size=test_size)

    val_size = floor(len(train_val_paths) * set_split)
    train_paths, val_paths, train_age_labels, val_age_lables, train_gender_labels, val_gender_lables = train_test_split(
        train_val_paths, train_val_age_labels, train_val_gender_labels, test_size=val_size)

    write_set_to_hdf5(train_paths, [train_age_labels, train_gender_labels], output_dir, 'training.hdf5', channels_first=channels_first)
    write_set_to_hdf5(val_paths, [val_age_lables, val_gender_lables], output_dir, 'validation.hdf5', channels_first=channels_first)
    write_set_to_hdf5(test_paths, [test_age_lables, test_gender_lables], output_dir, 'test.hdf5', channels_first=channels_first)

def write_set_to_hdf5(data_paths, labels_list, output_dir, output_name, channels_first=False):
    '''
    Read images data, convert to tensor and write both tensor data and label to one hdf5 file.
    Image can be either channels first or channels_last. 
    '''

    print('Writing to {0}...'.format(output_name))
    label_counts = len(labels_list)

    dims = (len(data_paths), 3, 150, 150) if channels_first else (len(data_paths), 150, 150, 3)
    writer = HDF5WriterMultiClasses(dims, output_dir, output_name, label_counts=2)

    # Progress bar to track writing process
    widgets = ['Processing:', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(data_paths), widgets=widgets).start()

    R, G, B = [], [], [] # save rgb values for mean normalization, only for training set
    images_buffer = []
    labels_buffer = []
    for i in range(label_counts):
        labels_buffer.append([])

    # For each path in image path's list
    for i in np.arange(len(data_paths)):
        # Read image, resize to 150x150 pixel
        image = load_img(
            path = data_paths[i],
            grayscale=False,
            color_mode='rgb',
            target_size=(150,150),
            interpolation='nearest'
        )

        # Convert image to tensor
        data_format = 'channels_first' if channels_first else 'channels_last'
        image = img_to_array(image, data_format=data_format)

        # Append current mean R, G, B values to array
        if output_name == 'training.hdf5':
            axis = (1, 2) if channels_first else (0, 1) 
            r, g, b = image.mean(axis=axis)
            R.append(r)
            G.append(g)
            B.append(b)
        
        images_buffer.append(image)
        for j in range(label_counts):
            labels_buffer[j].append(labels_list[j][i])

        # If buffer is full, write all data in buffer to hdf5 file
        if (i + 1) % writer.buffer_size == 0:
            writer.write(images_buffer, labels_buffer)
            images_buffer = []
            labels_buffer = []
            for i in range(label_counts):
                labels_buffer.append([])

        pbar.update(i) # update progress bar

    # If there is any data left in buffer, flush it to hdf5 file
    if len(images_buffer) > 0:
        writer.write(images_buffer, labels_buffer)

    writer.close()

    # For training set, calculate to mean values of R, G, B channels and write to json file
    if output_name == 'training.hdf5':
        print('Calculating color means...')
        mean = {'R': np.mean(R).astype('float'), 'G': np.mean(G).astype('float'), 'B': np.mean(B).astype('float')} # cannot dumps float32

        mean_path = os.path.join(output_dir, 'mean.json')
        with open(mean_path, 'w') as f:
            f.write(json.dumps(mean))

    pbar.finish() # update progress bar
