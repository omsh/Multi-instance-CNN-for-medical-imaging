import glob
from PIL import Image
from sklearn.utils import shuffle
from os.path import join
from os import listdir
import numpy as np
import tensorflow as tf
import random


# Function to read the converted png images and labels from disk and return numpy arrays

def read_images_labels(dir_names=["data/0_Benign_PNGs", "data/1_Cnormal_PNGs", "data/2_InSitu_PNGs",
                                "data/3_Invasive_PNGs"], n = 100, pre_shuffle = True, seed = 1):
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
        
        images, labels = np.asarray(images), np.asarray(labels)
        
        if (pre_shuffle):
            images, labels = shuffle(images, labels, random_state = seed)
            
        return images, labels 

# -------------------------------------------------------------------------------------

# Function to split numpy arrays of images and labels with split a ratio
# returns 4 arrays images, labels for train and val

def split_train_val(images, labels, ratio = 0.8, pre_shuffle = True, seed = 1):
        n = labels.shape[0]
        s = int(n * ratio)
        if (pre_shuffle):
            images, labels = shuffle(images, labels, random_state = seed)
        return images[:s], labels[:s], images[s:], labels[s:]
    

# -------------------------------------------------------------------------------------

# Function to pre-augment whole images (needed before patching)

    
def pre_augment_images(images, angles=[0, 90, 180, 270]):
    rotation_vector = [random.choice(angles) for _ in range(images.shape[0])]
    rotation_vector = np.asarray(rotation_vector) * np.pi / 180
    
    with tf.Session() as sess:
        images = tf.contrib.image.rotate(images, angles = rotation_vector)
    return images
# -------------------------------------------------------------------------------------

# Function to create patches for a 4-D tensor of images:
# args:
    # "images": 4-D tensor of shape: n_images X image_width X image_height X image_channels
    # "size": tuple of patch size --> number of patches then determined accordingly
    # "n_patches": if supplied, try to find the best patch size to match this total number given the "delta"

# returns:
    # "patches": 5-D tensor of shape: n_images X n_patches_per_image X patch_width X patch_height X patch_channels   
    # "number_of_patches_per_image": actual number of patches per image

def get_patches_from_images_tensor(images, size=(224, 224), n_patches=-1, delta=5):
    n = images.shape[0]
    with tf.Session() as sess:
        if (n_patches != -1):
            h, w = images.shape[1:3]
            x = h * w / n_patches
            size_w = size_h = np.floor(np.sqrt(x))
            actual_patches = 0
            while np.abs(n_patches - actual_patches) >= delta:
                size_w += 1
                size_h += 1
                print("Searching for best split")
                patches = tf.extract_image_patches(images,
                                              [1, size_w, size_h, 1],
                                              [1, size_w, size_h, 1],
                                              [1, 1, 1, 1],
                                              "SAME")
                new_shape = (n, -1, size_w, size_h, 3)
                patches = tf.reshape(patches, shape = new_shape)
                actual_patches = int(patches.shape[1])                   
        else:
            size_w, size_h = size

        patches = tf.extract_image_patches(images,
                                              [1, size_w, size_h, 1],
                                              [1, size_w, size_h, 1],
                                              [1, 1, 1, 1],
                                              "SAME")

        new_shape = (n, -1, size_w, size_h, 3)
        patches = tf.reshape(patches, shape=new_shape) 
        number_of_patches_per_image = patches.shape[1]
    return patches, number_of_patches_per_image
