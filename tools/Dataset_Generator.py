from __future__ import print_function

import time
import warnings
import numpy as np
import glob
import os
from keras.utils import to_categorical
from random import shuffle, randint
from scipy.misc import imread
import skimage.transform
from utils import compute_mean_std, load_img

from utils import shuffle


class Dataset_Generator(object):
    def __init__(self, cf, dataset_images_path, n_classes=2, batch_size=5, resize_image=(224, 224), shuffle=True,
                 apply_augmentation=False, sampling_score=None, data_path='data', data_path2=None, mode='train'):

        self.dataset_images_path = dataset_images_path
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize_image = resize_image
        self.rescale = cf.norm_rescale
        self.featurewise_center = cf.norm_featurewise_center
        self.featurewise_std_normalization = cf.norm_featurewise_std_normalization

        self.history_batch_fnames = np.array([''])
        self.history_batch_labels = np.array([], dtype=np.int32)

        self.apply_augmentation = apply_augmentation
        self.sampling_score = sampling_score

        if mode not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid mode: ', mode, '; expected \'train\', \'validation\' or \'test\'')

        # Load training set
        print('\n > Reading: ' + mode + ' set...')
        print('   Data path: ' + data_path)
        print('   Dataset images path: ' + dataset_images_path)


        # Load Neop: X_train/validation/test_neop.npy, y_train/validation/test_neop.npy
        self.X_neop = np.load(data_path + '_X_' + mode + '_neop.npy')
        self.y_neop = np.load(data_path + '_y_' + mode + '_neop.npy')

        # Load NOneo: X_train/validation/test_noneo.npy, y_train/validation/test_noneo.npy
        self.X_noneo = np.load(data_path + '_X_' + mode + '_noneo.npy')
        self.y_noneo = np.load(data_path + '_y_' + mode + '_noneo.npy')

        # We have only 2 classes
        # noneo = class 0
        # neo   = class 1
        self.y_noneo_class = np.zeros((len(self.y_noneo),), dtype=np.int32)
        self.y_neop_class = np.ones((len(self.y_neop),), dtype=np.int32)  # np.ones((5,), dtype=int)

        # Only for train: compute the mean and std, and save them.
        if mode == 'train':
            # Compute the mean and std using only the train set (all the images in the train set)
            t = time.time()
            X_all = np.concatenate((self.X_neop, self.X_noneo), axis=0)
            self.preprocess(cf, X_all)
            print("   Time to compute mean and std: {0:.2f} seconds.".format(time.time() - t))
            # Save the mean and std
            np.save(data_path + '_X_' + mode + '_mean_std', np.array([self.rgb_mean, self.rgb_std], dtype=np.float32))
            print("   Mean and std saved to ", data_path + '_X_' + mode + '_mean_std.npy')

        else: # for validation or test
            print("   Reading mean and std from " + data_path2 + '_X_' + 'train' + '_mean_std.npy')
            tmp = np.load(data_path2 + '_X_' + 'train' + '_mean_std.npy')
            self.rgb_mean = tmp[0]
            self.rgb_std = tmp[1]


        # Statistic
        self.noneo_size = len(self.X_noneo)
        self.neop_size = len(self.X_neop)

        self.total_images = self.noneo_size + self.neop_size

        self.proportion_class_noneo = (len(self.X_noneo) * 100) / self.total_images  # it is the lowest
        # proportion_class_neop = (len(self.X_neop) * 100) / self.total_images

        self.noneo_batch_size = int(round((self.proportion_class_noneo * self.batch_size) / 100))

        self.neop_batch_size = self.batch_size - self.noneo_batch_size

        # Load training set
        print('\n   Information: ' + mode + ' set')
        print('   Total images: ' + str(self.total_images))
        print('   Batch size: ' + str(self.batch_size))

        print('   Noneo images: ' + str(self.noneo_size))
        print('   Noneo batch size: ' + str(self.noneo_batch_size))

        print('   Neop images: ' + str(self.neop_size))
        print('   Neop batch size: ' + str(self.neop_batch_size))


    def preprocess(self, cf, X_all):

        # Compute mean
        if self.featurewise_center:
            self.rgb_mean = compute_mean_std(self.dataset_images_path, X_all, self.resize_image, self.rescale, method='mean',
                                             mean=None)
            # Broadcast the shape
            broadcast_shape = [1, 1, 3]
            self.mean = np.reshape(self.rgb_mean, broadcast_shape)
            print('   Mean {}: {}'.format(self.mean.shape, self.rgb_mean, self.mean))

        # Compute std
        if self.featurewise_std_normalization:
            if not cf.norm_featurewise_center:
                self.rgb_mean = compute_mean_std(self.dataset_images_path, X_all, self.resize_image, self.rescale, method='mean',
                                                 mean=None)

            var = compute_mean_std(self.dataset_images_path, X_all, self.resize_image, self.rescale, method='var', mean=self.rgb_mean)
            self.rgb_std = np.sqrt(var)
            # Broadcast the shape
            broadcast_shape = [1, 1, 3]
            self.std = np.reshape(self.rgb_std, broadcast_shape)
            print('   Std {}: {}'.format(self.std.shape, self.rgb_std))


    def standardize(self, x):

        # Normalize
        if self.rescale:
            x *= self.rescale

        # Standardize
        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This Data_Generator specifies `featurewise_center`, but it hasn\'t'
                              'been fit on any training data')

        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This Data_Generator specifies `featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data.')

        return x


    # strategy 1: keep unbalanced each batch as the dataset.
    def generate(self):

        while True:

            for nFiles in range(self.total_images // self.batch_size):
                # if ((nFiles + 1) * self.noneo_batch_size) > self.noneo_size:
                #     print("X_noneo ended!")
                #     continue
                #
                # if ((nFiles + 1) * self.neop_batch_size) > self.neop_size:
                #     print("X_neop ended!")
                #     #continue

                X_noneo_names = self.X_noneo[nFiles * self.noneo_batch_size:(nFiles + 1) * self.noneo_batch_size]
                y_noneo_labels = self.y_noneo_class[nFiles * self.noneo_batch_size:(nFiles + 1) * self.noneo_batch_size]

                X_neop_names = self.X_neop[nFiles * self.neop_batch_size:(nFiles + 1) * self.neop_batch_size]
                y_neop_labels = self.y_neop_class[nFiles * self.neop_batch_size:(nFiles + 1) * self.neop_batch_size]

                # Checking special case for the last batch size
                # if we there is no more noneo items
                if len(X_noneo_names) < self.noneo_batch_size:
                    # fill with the other class
                    #print ("++")
                    missing = self.batch_size - len(X_noneo_names)
                    X_neop_names = self.X_neop[nFiles * self.neop_batch_size:(nFiles + 1) * self.neop_batch_size + missing]
                    y_neop_labels = self.y_neop_class[nFiles * self.neop_batch_size:(nFiles + 1) * self.neop_batch_size + missing]

                elif len(X_neop_names) < self.neop_batch_size:   #if we there is no more neop items
                    # fill with the other class
                    #print("+++++")
                    missing = self.batch_size - len(X_neop_names)
                    X_noneo_names = self.X_noneo[nFiles * self.noneo_batch_size:(nFiles + 1) * self.noneo_batch_size + missing]
                    y_noneo_labels = self.y_noneo_class[nFiles * self.noneo_batch_size:(nFiles + 1) * self.noneo_batch_size + missing]


                batch_fnames = np.concatenate((X_noneo_names, X_neop_names), axis=0)
                batch_labels = np.concatenate((y_noneo_labels, y_neop_labels), axis=0)

                self.history_batch_fnames = np.concatenate((self.history_batch_fnames, batch_fnames), axis=0)
                self.history_batch_labels = np.concatenate((self.history_batch_labels, batch_labels), axis=0)

                assert len(batch_fnames) == self.batch_size



                #print(">>>>>>>> batch_fnames = ", batch_fnames)
                #print(">>>>>>>> batch_labels = ", batch_labels)
                img_batch = []
                lab_batch = []

                if self.shuffle:
                    shuffle(batch_fnames, batch_labels)

                # Create the batch_x and batch_y
                for idx, image_name in enumerate(batch_fnames):
                    #print("\n Reading images")
                    # image = imread(os.path.join(self.dataset_images_path, image_name))  # Build batch of image data
                    #
                    # if self.resize_image is not None:
                    #     image = skimage.transform.resize(image, self.resize_image, order=1, preserve_range=True)
                    #     #print("resized")

                    image = load_img(os.path.join(self.dataset_images_path, image_name), resize=self.resize_image)

                    # Add images to batches
                    img_batch.append(image)     
                    # Build batch of label data, reshape and add to batch
                    lab_batch.append(to_categorical(batch_labels[idx], self.n_classes).reshape(self.n_classes))
                
                yield (np.array(img_batch), np.array(lab_batch))
