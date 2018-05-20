import tensorflow as tf
import numpy as np
import glob
from os import listdir
from os.path import join
from PIL import Image
import logging
import pprint

class DatasetLoader:
    """
    DataSetAPI - Load Imgs from the disk
    """

    def __init__(self, config):
        self.config = config
        
        if (config.train_on_subset):
            images, labels = DatasetLoader.read_images_labels(n = int(self.config.subset_size/4))
        else:
            images, labels = DatasetLoader.read_images_labels()
            
        n = labels.shape[0]
        
        logging.info(f"Number of Images and Labels loaded: {pprint.pformat(n)}")
        
        s = int(n * self.config.train_val_split)
        
        # training dataset
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((images[0:s], labels[0:s]))
        self.train_dataset = self.train_dataset.map(DatasetLoader.preprocess_train,
                                                    num_parallel_calls = self.config.batch_size)

        self.train_dataset = self.train_dataset.shuffle(n, reshuffle_each_iteration = True)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                           self.train_dataset.output_shapes)
        
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        
        # validation dataset
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((images[s:], labels[s:]))
        self.val_dataset = self.val_dataset.map(DatasetLoader.preprocess_val,
                                                    num_parallel_calls = self.config.batch_size)

        self.val_dataset = self.val_dataset.batch(self.config.batch_size)
                
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        
        self.len_x_train = labels[0:s].shape[0]
        self.len_x_val = labels[s:].shape[0]

        self.num_iterations_train = self.len_x_train // self.config.batch_size
        self.num_iterations_val = self.len_x_val // self.config.batch_size
        
    
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
    def read_images_labels(dir_names=["0_Benign_PNGs", "1_Cnormal_PNGs", "2_InSitu_PNGs",
                                "3_Invasive_PNGs"], n = 100):
        if (n > 100):
            n = 100

        images = []
        labels = []
        files_list = []
        dir_listings = [glob.glob(join(x, '*.png')) for x in dir_names]

        for d_list in dir_listings:
            for file in d_list[:n]:
                files_list.append(file)

        for file in files_list:
            im = np.asarray(Image.open(file))
            label = int(len(images)/n)
            images.append(im)
            labels.append(label)

        return np.asarray(images), np.asarray(labels)

    
    def initialize(self, sess, train = True):
        if (train):
            sess.run(self.training_init_op)
        else:
            sess.run(self.val_init_op)
                

    def get_input(self):
        return self.iterator.get_next()