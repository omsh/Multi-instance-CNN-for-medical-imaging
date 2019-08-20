import glob
from PIL import Image
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from os.path import join
from os import listdir
import numpy as np
import tensorflow as tf
import random
import ntpath
import re


def get_images_pathlist_labels(dir_names=["data/0_Benign_PNGs", "data/1_Cnormal_PNGs", "data/2_InSitu_PNGs", "data/3_Invasive_PNGs"], n = 100, pre_shuffle = True, seed = 1):
    if (n > 100):
        n = 100

    labels = []
    files_list = []
    dir_listings = [glob.glob(join(x, '*.png')) for x in dir_names]

    for d_list in dir_listings:
        for file in d_list[:n]:
            files_list.append(file)
            label = re.search("^[a-z]*", ntpath.basename(file))
            labels.append(label.group(0))

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    files_list, labels = np.asarray(files_list), np.asarray(labels)
    bag_index = np.asarray(list(range(len(labels))))

    if (pre_shuffle):
        files_list, labels = shuffle(files_list, labels, random_state = seed)

    return files_list, labels, bag_index
        

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
    bag_index = np.asarray(list(range(len(labels))))
    
    if (pre_shuffle):
        images, labels = shuffle(images, labels, random_state = seed)

    return images, labels, bag_index

# -------------------------------------------------------------------------------------

# Function to split numpy arrays of images and labels with split a ratio
# returns 4 arrays images, labels for train and val

def split_train_val(images, labels, bag_index, ratio = 0.8, pre_shuffle = True, seed = 1, per_class_split = True):
        if (pre_shuffle):
            images, labels, bag_index = shuffle(images, labels, bag_index, random_state = seed)
            
        if (per_class_split):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-ratio), random_state = seed)
            for train_index, val_index in sss.split(images, labels):
                X_train, X_val = images[train_index], images[val_index]
                y_train, y_val = labels[train_index], labels[val_index]
                bi_train, bi_val = bag_index[train_index], bag_index[val_index]
            
        else:
            n = labels.shape[0]
            s = int(n * ratio)
            X_train, X_val =  images[:s], images[s:]
            y_train, y_val =  labels[:s], labels[s:]
            bi_train, bi_val = bag_index[:s], bag_index[s:]
            
        return X_train, y_train, bi_train, X_val, y_val, bi_val
    
# -------------------------------------------------------------------------------------

# Function to pre-augment whole images (needed before patching)

    
def pre_augment_images(images, angles=[0, 90, 180, 270]):
    rotation_vector = [random.choice(angles) for _ in range(images.shape[0])]
    rotation_vector = np.asarray(rotation_vector) * np.pi / 180
    images = tf.contrib.image.rotate(images, angles = rotation_vector)
    return images



def rotate_images(images, angles = [0, 90, 180, 270]):
    X_rotate = []
    angles = np.array(angles) / 90
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (images.shape[1], images.shape[2], 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            rotated_img = sess.run(tf_img, feed_dict = {X: img, k: random.choice(angles)})
            X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate


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

        print("Patches  shape before failing: ", patches.shape)
        new_shape = (n, -1, size_w, size_h, 3)
        patches = tf.reshape(patches, shape=new_shape) 
        number_of_patches_per_image = patches.shape[1]
    return patches, number_of_patches_per_image


def extract_patches_from_tensor(images, size=(224, 224), overlap = 0):
    n = tf.shape(images)[0]
    
    size_w, size_h = size
    stride_w = (1 - overlap) * size_w
    stride_h = (1 - overlap) * size_h
    
    patches = tf.extract_image_patches(images,
                                       [1, size_h, size_w, 1],
                                       [1, stride_h, stride_w, 1],
                                       [1, 1, 1, 1],
                                       "SAME")
    
    new_shape = (n, -1, size_w, size_h, 3)
    patches = tf.reshape(patches, shape=new_shape) 
    number_of_patches_per_image = tf.shape(patches)[1]
    
    return patches, number_of_patches_per_image
