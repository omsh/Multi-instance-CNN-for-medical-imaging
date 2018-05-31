import tensorflow as tf
import numpy as np

from utils.img_utils import read_images_labels, get_patches_from_images_tensor, split_train_val, pre_augment_images

import logging
import pprint
import random

class DatasetLoader:
    """
    
    Loading images using the Dataset API
    Two Datasets are initialized, one for training and one for validation
    
    """

    def __init__(self, config):
        self.config = config
        
        # Read the PNG images from disk, whether a subset or not
        
        if (config.train_on_subset):
            images, labels = read_images_labels(n = int(self.config.subset_size/4))
        else:
            images, labels = read_images_labels()
        
        # Split training data (since reading is sequential in terms of classes)
        
        train_images, train_labels, val_images, val_labels = split_train_val(images,
                                                 labels,
                                                 ratio = self.config.train_val_split, 
                                                 pre_shuffle = True)

        del images, labels
        
        # Extract patches on the fly in case training is on image patches
        
        
        if (config.train_on_patches):
            train_images = pre_augment_images(train_images,
                                              angles = self.config.rotation_angles)
            
            train_images, train_labels = DatasetLoader.convert_to_patches_repeat_labels(train_images,
                                                                          train_labels,
                                                                          self.config.patch_size)

            val_images, val_labels = DatasetLoader.convert_to_patches_repeat_labels(val_images,
                                                                          val_labels,
                                                                          self.config.patch_size)
            
            logging.info(f"Shape of train images after getting all patches:{pprint.pformat(train_images.shape)}")
            
            logging.info(f"Shape of val images after getting all patches: {pprint.pformat(val_images.shape)}")
                
        logging.info(f"Number of Training Images and Labels: {pprint.pformat(train_labels.shape[0])}")
        logging.info(f"Number of    Val   Images and Labels: {pprint.pformat(val_labels.shape[0])}")
        
        # training dataset
        n = train_images.shape[0]
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        self.train_dataset = self.train_dataset.map(self.preprocess_train,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        self.train_dataset = self.train_dataset.shuffle(n, reshuffle_each_iteration = False)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                           self.train_dataset.output_shapes)
        
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        
        # validation dataset
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        self.val_dataset = self.val_dataset.map(self.preprocess_val,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        self.val_dataset = self.val_dataset.batch(self.config.batch_size)
                
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        
        self.len_x_train = train_labels.shape[0]
        self.len_x_val = val_labels.shape[0]

        self.num_iterations_train = self.len_x_train // self.config.batch_size
        self.num_iterations_val = self.len_x_val // self.config.batch_size
        
        del train_images, train_labels, val_images, val_labels
        
    
    
    def preprocess_train(self, image, label):
        #img = tf.random_crop(image, [224, 224, 3])
        img = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        if (not self.config.train_on_patches):
            n_times_90 = int(random.choice[np.array(self.config.rotation_angles, dtype=np.int16) / 90])
            img = tf.image.rot90(img, k = n_times_90)
        return tf.cast(img, tf.float32), tf.cast(label, tf.int64)
    
    
    def preprocess_val(self, image, label):
            img = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            return tf.cast(img, tf.float32), tf.cast(label, tf.int64)
    
    
    def preprocess_patched_train(self, image, label):
        # to be implemented if needed
        return
        
    
    def preprocess_patched_val(self, image, label):
        # to be implemented if needed
        return
    
    @staticmethod
    def convert_to_patches_repeat_labels(images, labels, patch_size, train = True):
        
        patches, n_patches = get_patches_from_images_tensor(images, size=(patch_size, patch_size))
        
        logging.info(f"Shape of patches: {pprint.pformat(patches.shape)}")
        logging.info(f"Shape of labels: {pprint.pformat(labels.shape)}")
        logging.info(f"Number of patches extracted: {pprint.pformat(n_patches)}")        
            
        # repeat labels for patches
        labels = np.asarray([l for l in labels for _ in range(n_patches)])
        logging.info(f"Shape of labels after patching (repeat): {pprint.pformat(labels.shape)}")
                                                     
        # squeeze 1st and 2nd dimensions via reshape (this) or sampling (later)
        images = tf.reshape(patches, shape=(-1, *patches.shape[2:]))
        
        return images, labels
    
    def initialize(self, sess, train = True):
        if (train):
            sess.run(self.training_init_op)
        else:
            sess.run(self.val_init_op)

    def get_input(self):
        return self.iterator.get_next()