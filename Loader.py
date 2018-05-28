import tensorflow as tf
import numpy as np
from utils.img_utils import read_images_labels, get_patches_from_images_tensor
import logging
import pprint

class DatasetLoader:
    """
    
    Loading images using the Dataset API
    Two Datasets are initialized, one for training and one for validation
    
    """

    def __init__(self, config):
        self.config = config
        
        if (config.train_on_subset):
            images, labels = read_images_labels(n = int(self.config.subset_size/4))
        else:
            images, labels = read_images_labels()
        
        # How to integrate augmentation for original images before patching?
        # * do that here before extracting patches
        # * ???
        
        if (config.train_on_patches):
            patches, n_patches = get_patches_from_images_tensor(images, size=(config.patch_size, config.patch_size))
            print("Shape of patches: ", patches.shape)
            print("Shape of labels: ", labels.shape)
            print("Number of patches: ", n_patches)
            
            # repeat labels for patches
            labels = np.asarray([l for l in labels for _ in range(n_patches)])
            print("Shape of labels after patching: ", labels.shape)
                                                     
            # squeeze 1st and 2nd dimensions via reshape (this) or sampling (later)
            images = tf.reshape(patches, shape=(-1, *patches.shape[2:]))
            print("Shape of images after getting all patches: ", images.shape)
            del patches
            
        n = labels.shape[0]
        
        logging.info(f"Number of Images and Labels loaded: {pprint.pformat(n)}")
        
        s = int(n * self.config.train_val_split)
        
        # training dataset
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((images[0:s], labels[0:s]))
        self.train_dataset = self.train_dataset.map(DatasetLoader.preprocess_train,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        self.train_dataset = self.train_dataset.shuffle(n, reshuffle_each_iteration = True)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                           self.train_dataset.output_shapes)
        
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        
        # validation dataset
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((images[s:], labels[s:]))
        self.val_dataset = self.val_dataset.map(DatasetLoader.preprocess_val,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        self.val_dataset = self.val_dataset.batch(self.config.batch_size)
                
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        
        self.len_x_train = labels[0:s].shape[0]
        self.len_x_val = labels[s:].shape[0]

        self.num_iterations_train = self.len_x_train // self.config.batch_size
        self.num_iterations_val = self.len_x_val // self.config.batch_size
        
        del images, labels
        
    
    @staticmethod
    def preprocess_train(image, label):
        #img = tf.random_crop(image, [224, 224, 3])
        img = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return tf.cast(img, tf.float32), tf.cast(label, tf.int64)
    
    @staticmethod
    def preprocess_val(image, label):
            img = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            return tf.cast(img, tf.float32), tf.cast(label, tf.int64)
    
    @staticmethod
    def preprocess_patched_train(image, label):
        # to be implemented if needed
        return
        
    @staticmethod
    def preprocess_patched_val(image, label):
        # to be implemented if needed
        return

    
    def initialize(self, sess, train = True):
        if (train):
            sess.run(self.training_init_op)
        else:
            sess.run(self.val_init_op)
                

    def get_input(self):
        return self.iterator.get_next()