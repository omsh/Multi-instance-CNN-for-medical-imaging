import tensorflow as tf
import numpy as np
from PIL import Image
from utils.img_utils import get_images_pathlist_labels, extract_patches_from_tensor, split_train_val
import logging
import pprint
import random

class DatasetFileLoader:
    """
    
    Loading images using the Dataset API
    Image path as input to the dataset API
    Two Datasets are initialized, one for training and one for validation
    
    """

    def __init__(self, config):
        self.config = config
        
        # Get the paths of PNG images and the labels, whether a subset or not
        
        if (config.train_on_subset):
            train_images, train_labels, train_bi, val_images, val_labels, val_bi = split_train_val(
                *get_images_pathlist_labels(n = int(self.config.subset_size/4)),
                ratio = self.config.train_val_split, 
                pre_shuffle = True)
        else:
            train_images, train_labels, train_bi, val_images, val_labels, val_bi = split_train_val(
                *get_images_pathlist_labels(),
                ratio = self.config.train_val_split, 
                pre_shuffle = True)
                
        logging.info(f"Number of Training Images and Labels: {pprint.pformat(train_labels.shape[0])}")
        logging.info(f"Number of    Val   Images and Labels: {pprint.pformat(val_labels.shape[0])}")
        
        # training dataset
        n = train_images.shape[0]
        n_val = val_images.shape[0]
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_bi))
        
        self.train_dataset = self.train_dataset.shuffle(n, reshuffle_each_iteration = True).repeat()
        
        self.train_dataset = self.train_dataset.map(self.read_images,
                                                 num_parallel_calls = self.config.num_parallel_cores)
        
        self.train_dataset = self.train_dataset.map(self.preprocess_train,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)
        
        if (self.config.train_on_patches):
            self.train_dataset = self.train_dataset.map(
                self.get_patches_train, num_parallel_calls = self.config.num_parallel_cores)

        self.train_dataset = self.train_dataset.map(self.color_augment,
                                                    num_parallel_calls = self.config.num_parallel_cores)
        
        self.train_dataset = self.train_dataset.prefetch(3)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                           self.train_dataset.output_shapes)
        
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        
        
        # validation dataset
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels, val_bi)).repeat()
        
        self.val_dataset = self.val_dataset.map(self.read_images,
                                                    num_parallel_calls = self.config.num_parallel_cores)
        
        self.val_dataset = self.val_dataset.map(self.preprocess_val,
                                                    num_parallel_calls = self.config.num_parallel_cores)

        #if (self.config.patch_generation_scheme == 'random_crops' and self.config.train_on_patches):
        #    self.val_dataset = self.val_dataset.map(
        #        self.get_random_crop, num_parallel_calls = self.config.num_parallel_cores)
            
        #    self.val_dataset = self.val_dataset.batch(
        #        self.config.batch_size * self.config.n_random_patches)
        #else:    
        self.val_dataset = self.val_dataset.batch(1)

        if (self.config.train_on_patches):
            self.val_dataset = self.val_dataset.map(
                self.get_patches_val, num_parallel_calls = self.config.num_parallel_cores)                
        self.val_dataset = self.val_dataset.prefetch(3)
                
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
                
        self.len_x_train = train_labels.shape[0] 
        self.num_iterations_train = self.len_x_train // self.config.batch_size
        
        self.len_x_val = val_labels.shape[0]
        self.num_iterations_val = self.len_x_val // self.config.batch_size

        print("Iterations Train: ", self.num_iterations_train)
        print("Iterations Val: ", self.num_iterations_val)
        
    
    def read_images(self, image_path, label, bag_index):
        image = tf.image.decode_png(tf.read_file(image_path), channels = 3)
        image.set_shape([None, None, 3])
        
        return image, label, bag_index
        
    
    def preprocess_train(self, image, label, bag_index):
        # Rotation is done (for patching mode --> pre-augment whole images, else --> augment)
        
        n_times_90 = int(random.choice(np.array(self.config.rotation_angles, dtype=np.int16) / 90))
        image = tf.image.rot90(image, k = n_times_90)
        
        # If not in patching mode, make sure size is 224 
        if (not self.config.train_on_patches):
            image = tf.image.resize_images(image, [224, 224])
            image = tf.cast(image, dtype=tf.float32)

        
        logging.info(f"Shape of image: {pprint.pformat(tf.shape(image))}")

        return image, tf.cast(label, tf.int64), bag_index
    
    
    def preprocess_val(self, image, label, bag_index):
        # No Rotation is done 
        
        # If not in patching mode, make sure size is 224 
        if (not self.config.train_on_patches):
            image = tf.image.resize_images(image, [224, 224])
            image = tf.cast(image, dtype=tf.float32)

        
        return image, tf.cast(label, tf.int64), bag_index
    
    def get_patches_train(self, images, labels, bag_index):
        # just getting a cleaner code below
        n_patches = self.config.n_random_patches
        p = self.config.patch_size
        c = self.config.channels
        
        if (self.config.patch_generation_scheme == 'random_crops'):
            images = tf.reshape(
                tf.map_fn(
                    lambda z: tf.stack([tf.random_crop(
                        z, size = [p, p, c]) for _ in range(n_patches)]),
                    images),
                shape=[-1, p, p, c])
        else:
            images, n_patches = extract_patches_from_tensor(
                images, size=(p, p),
                overlap = self.config.patches_overlap)
        
        # squeeze 1st and 2nd dimensions via reshape or sampling
                            
        if (self.config.patch_generation_scheme == 'sequential_randomly_subset'):
            i = tf.random_uniform(shape = [1], minval = 0, maxval = tf.shape(images)[1] - n_patches,
                                  dtype = tf.int32, seed = self.config.random_seed)
            j = i + n_patches
            
            images, labels = tf.cond(
                tf.greater(tf.shape(images)[1], n_patches),
                lambda: (tf.map_fn(lambda x: x[i[0] : j[0], :, :], images),
                         tf.reshape(tf.map_fn(lambda x: tf.tile([x], [n_patches]), labels), shape=(-1,))),
                lambda: (images, labels))
        else:
            labels = tf.reshape(tf.map_fn(lambda x: tf.tile([x], [n_patches]), labels), shape=(-1,))
                                
        images = tf.reshape(images, shape=(-1, p, p, c))
        
        images = tf.image.resize_images(images, [224, 224])
        
        return tf.cast(images, dtype = tf.float32), labels, bag_index
    
    def get_patches_val(self, images, labels, bag_index):
        p = self.config.patch_size
        c = self.config.channels
        
        images, n_patches = extract_patches_from_tensor(images, size=(p, p),
                                                        overlap = self.config.patches_overlap)
                                                             
        # squeeze 1st and 2nd dimensions via reshape (validation) and repeat labels
        
        
        labels = tf.reshape(tf.map_fn(lambda x: tf.tile([x], [n_patches]), labels), shape=(-1,))       
        images = tf.reshape(images, shape=(-1, p, p, c))
        images = tf.image.resize_images(images, [224, 224])
        
        return tf.cast(images, dtype = tf.float32), labels, bag_index

    
    def get_random_crop(self, images, labels, bag_index):
        images = tf.random_crop(images, size = [self.config.patch_size, self.config.patch_size, 3])
        return tf.cast(images, dtype = tf.float32), labels, bag_index
    
    def color_augment(self, images, labels, bag_index):
        if (self.config.random_brightness):
            images = tf.image.random_brightness(images, 0.5)
        if (self.config.random_contrast):
            images = tf.image.random_contrast(images, 0.75, 1)
        if (self.config.random_saturation):
            images = tf.image.random_saturation(images, 0.75, 1)
        if (self.config.random_hue):
            images = tf.image.random_hue(images, 0.05)
            
        return tf.cast(images, dtype = tf.float32), labels, bag_index
        
    
    def initialize(self, sess, train = True):
        if (train):
            sess.run(self.training_init_op)
        else:
            sess.run(self.val_init_op)
    
    def get_input(self):
        return self.iterator.get_next()