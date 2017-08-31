#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:47:34 2017

@author: ajaver
"""
import numpy as np
import tables
import scipy.ndimage as ndi
from scipy.ndimage.filters import gaussian_filter
import threading
from keras import backend as K

from skelxy2ang import transform2skelangles, transform2skelxy

def random_rotation(rg, h, w):
    
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)    
    return transform_matrix


def random_shift(shift_range, h, w):
    
    tx = np.random.uniform(-shift_range, shift_range) * h
    ty = np.random.uniform(-shift_range, shift_range) * w
    translation_matrix_x = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    translation_matrix_y = np.array([[1, 0, -ty],
                                   [0, 1, -tx],
                                   [0, 0, 1]])
    return translation_matrix_x, translation_matrix_y


def random_zoom(zoom_range, h, w):
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix_x = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix_x = transform_matrix_offset_center(zoom_matrix_x, h, w)
    
    
    zoom_matrix_y = np.array([[1/zy, 0, 0],
                            [0, 1/zx, 0],
                            [0, 0, 1]])
    transform_matrix_y = transform_matrix_offset_center(zoom_matrix_y, h, w)
    return transform_matrix_x, transform_matrix_y

def apply_transform_coord(y, transform_matrix):
    yr = np.ones((y.shape[0],y.shape[1]+1)); 
    yr[:,:-1] = y
    yr = np.dot(transform_matrix, yr.T).T
    return yr[:, :-1]

def apply_transform_img(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x




def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_transform(xx, yy, 
                     rotation_range=90, 
                     shift_range = 0.1,
                     zoom_range = (0.75, 1.25),
                     horizontal_flip=True,
                     vertical_flip=True):
    h, w = xx.shape[:-1]
    rot_mat = random_rotation(rotation_range, h, w)
    shift_mat_x, shift_mat_y = random_shift(shift_range, h, w)
    zoom_mat_x, zoom_mat_y = random_zoom(zoom_range, h, w)
    
    transform_x = rot_mat.copy()
    for mat in [shift_mat_x, zoom_mat_x]:
        transform_x = np.dot(transform_x, mat)
    
    transform_y = rot_mat.copy()
    for mat in [shift_mat_y, zoom_mat_y]:
        transform_y = np.dot(mat, transform_y)
    
    img = apply_transform_img(xx, transform_x, 2)
    yr = apply_transform_coord(yy, transform_y)

    if horizontal_flip and np.random.random() < 0.5:
        img = img[::-1, :, :]
        yr[:, 1] = -yr[:, 1] + img.shape[1]
    
    if vertical_flip and np.random.random() < 0.5:
        img = img[:, ::-1, :]
        yr[:, 0] = -yr[:, 0] + img.shape[0]

    return img, yr


class Iterator(object):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class ImageSkeletonsGenerator(Iterator):
    
    def __init__(self, sample_file, 
                 field_x = '/mask', 
                 field_y = '/skeleton',
                 field_indexes = '/index_groups/train',
                 batch_size=32, 
                 shuffle=True, 
                 seed=None,
                 do_intensity = False,
                 transform_ang = False):
       
        self.sample_file = sample_file
        self.fid = tables.File(sample_file, 'r') 
        
        
        
        self.X = self.fid.get_node(field_x)
        self.Y = self.fid.get_node(field_y)
        self.sample_indexes = self.fid.get_node(field_indexes)
        
        self.roi_size = self.X.shape[1]
        self.roi_size_half = self.roi_size/2.
        #self.n_skel_points = self.Y.shape[1]
        self.tot_samples = self.sample_indexes.shape[0]
        self.do_intensity = do_intensity
        self.transform_ang = transform_ang
        
        super(ImageSkeletonsGenerator, self).__init__(self.tot_samples, batch_size, shuffle, seed)

    def next(self):
        """
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size, self.roi_size, self.roi_size, 1), dtype=K.floatx())
        
        
        batch_y = np.zeros((current_batch_size, 49, 2), dtype=K.floatx())
        
        for i, j in enumerate(index_array):
            ind  = self.sample_indexes[j]
            xx = self.X[ind][:, :, np.newaxis]
            yy = self.Y[ind]
            
            xr, yr = random_transform(xx, yy, 
                     rotation_range=90, 
                     shift_range = 0.1,
                     zoom_range = (0.75, 1.25),
                     horizontal_flip=True,
                     vertical_flip=True)
            yr = (yr-self.roi_size_half)/self.roi_size_half
            
            
            if self.do_intensity:
                gamma = np.random.uniform(0.5, 3)
                alpha = np.random.uniform(0.5, 2)
                sigma_blur = np.random.uniform(0, 1.5)
                
                img_r = np.round((np.abs(xr)**gamma)*255*alpha)
                img_r = -np.clip(img_r, 0, 255)/255
                img_r = gaussian_filter(img_r, sigma=sigma_blur)
                xr = img_r
            
            
            batch_x[i] = xr
            batch_y[i] = yr
            
        if self.transform_ang:
            skel_angles, mean_angles, segment_sizes, ini_coord = transform2skelangles(batch_y)
            batch_y = np.concatenate((skel_angles, 
                                      mean_angles[:,np.newaxis], 
                                      segment_sizes[:,np.newaxis], 
                                      ini_coord), axis=1)
            
        return batch_x, batch_y

if __name__ == '__main__':
    import os
    import matplotlib.pylab as plt
    
    
    rand_seed = 1337
    SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/'
    sample_file = 'N2 on food R_2011_03_09__11_58_06___6___3_sample.hdf5'
    sample_file = os.path.join(SAVE_DIR, 'data', sample_file)
    
    
    img_generator = ImageSkeletonsGenerator(sample_file, 
                 batch_size=128, 
                 shuffle=True, 
                 seed=rand_seed,
                 transform_ang = True)
    #%%
    batch_x, batch_y = next(img_generator)
    batch_y = transform2skelxy(batch_y[:, :48], batch_y[:, 48], batch_y[:, 49], batch_y[:, 50:])
    
    n_rows, n_cols = 3,3 
    tot = n_rows*n_cols
    
    
    plt.figure()    
    for mm in range(tot):
        img = batch_x[mm]
        yr = batch_y[mm]
        
        
        
        yr = yr*img.shape[1]/2 + img.shape[1]/2.
        plt.subplot(n_rows,n_cols,mm+1)
        plt.imshow(np.squeeze(img), interpolation='none', cmap='gray')
        plt.plot(yr[:,0], yr[:,1], 'r')
        plt.plot(yr[0,0], yr[0,1], 'x')
#%%
#from skimage.filters import rank
#
#plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(np.squeeze(img*255), interpolation='none', cmap='gray')
#plt.subplot(1,2,2)
#
#img_r = np.round((np.abs(img)**3)*255*2)
#img_r = -np.clip(img_r, 0, 255)/255
#img_r = gaussian_filter(img_r, sigma=1.5)
#plt.imshow(np.squeeze(img_r), interpolation='none', cmap='gray')




