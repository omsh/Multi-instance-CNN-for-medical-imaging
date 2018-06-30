# For installing tflearn: https://stackoverflow.com/questions/48821174/how-to-install-tflearn-module-on-anaconda-distribution-in-windows-10

import tensorflow as tf
import numpy as np
from dataloaders import DatasetLoader, DatasetFileLoader
from models.BaseModel import BaseModel
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, class_num):
    return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')


class ResNeXt(BaseModel):
    def __init__(self, data_loader, config):
        super(ResNeXt, self).__init__(config)

        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.train_step = None
        self.num_classes = config.num_classes

        # ResNeXt specific hyperparameters
        # Can be added through config file
        self.cardinality = 8  # how many split ?
        self.blocks = 3  # res_block ! (split + transition)
        self.depth = 64 # out channel

        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        :return:
        """

        """
        Helper Variables
        """
        
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        print("Input to ResNext")
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """

        print("network arch ResNeXt")
        with tf.variable_scope('network'):

            input_x = self.first_layer(self.x, scope='first_layer')
            x = self.residual_layer(input_x, out_dim=64, layer_num='1', res_block=self.blocks)
            x = self.residual_layer(x, out_dim=128, layer_num='2', res_block=self.blocks)
            x = self.residual_layer(x, out_dim=256, layer_num='3', res_block=self.blocks)

            x = Global_Average_Pooling(x)
            x = flatten(x)
            self.logits = Linear(x, self.num_classes)

            print("network output ResNeXt")
            with tf.variable_scope('out'):
                # self.out = tf.squeeze(end_points['predictions'], axis=[1,2])
                self.out = tf.nn.softmax(self.logits, dim=-1)

            tf.add_to_collection('out', self.out)

            print("Logits shape: ", self.logits.shape)
            print("predictions out shape: ", self.out.shape)

            print("network output argmax ResNeXt")
            with tf.variable_scope('out_argmax'):
                self.out_argmax = tf.argmax(self.logits, axis=-1, output_type=tf.int64, name='out_argmax')
                # self.out_argmax = tf.squeeze(tf.argmax(self.out, 1), axis=[1])

                print("Arg Max Shape: ", self.out_argmax.shape)

        print("loss ResNeXt")
        with tf.variable_scope('loss-acc'):
            # one_hot_y = tf.one_hot(indices=self.y, depth=self.num_classes)

            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)

            # probabilities = end_points['Predictions']

            # accuracy, accuracy_update = tf.metrics.accuracy(labels = one_hot_y, predictions = self.out_argmax)
            #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))
            self.acc = self.evaluate_accuracy(self.y, self.out_argmax,
                                              self.is_training, self.config.patch_count)

        with tf.variable_scope('train_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch1')
            x = Relu(x)
            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=self.depth, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=self.depth, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch1')
            # x = Relu(x)
            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x