# -*- coding: utf-8 -*-

import os

import h5py


class HDF5WriterMultiClasses:
    '''
    Helper class to convert data into hdf5 format.
    '''

    def __init__(self, dims, output_dir, output_name, label_counts, buffer_size=1000):
        '''
        Set output path, dimensions and buffer size for writer.
        Output folder will be created if needed.
        '''
        
        self.label_counts = label_counts

        # Check and create output folder if it does not exist.
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        # Create database object with 2 datasets.
        # We will normalize out images data, because of that we use dtype==float32
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset('data', dims, dtype='float32')

        self.labels_list = []
        for i in range(self.label_counts):
            self.labels_list.append(self.db.create_dataset('labels_{0}'.format(i), (dims[0],), dtype='int'))

        self.buffer_size = buffer_size
        self.buffer = {
            'data': [],
        }
        for i in range(self.label_counts):
            self.buffer['labels_{0}'.format(i)] = []

        self.idx = 0 # Index in database

    def flush(self):
        '''
        Write data from buffer to disk and reset buffer.
        '''

        # Write buffer to disk
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']

        for j in range(self.label_counts):
            self.labels_list[j][self.idx:i] = self.buffer['labels_{0}'.format(j)]
        self.idx = i

        # Reset buffer
        self.buffer = {
            'data': [],
        }
        for i in range(self.label_counts):
            self.buffer['labels_{0}'.format(i)] = []

    def write(self, rows, labels_list):
        '''
        Write multiple rows to buffer,
        then flush buffer to disk if fulls.
        '''

        self.buffer['data'].extend(rows)
        for i in range(self.label_counts):
            self.buffer['labels_{0}'.format(i)].extend(labels_list[i])

        # Buffer is full, flush data to disk
        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    def close(self):
        '''
        Finish writing to hdf5 db.
        '''

        # If buffer still contains data, flush it all to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # Close database
        self.db.close()
