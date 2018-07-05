import tensorflow as tf
from models.BaseModel import BaseModel
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.layers import fully_connected
from utils.model_utils import acc_majority_class, combined_cost_function

import logging
import pprint


class ResNet50_MI(BaseModel):
    def __init__(self, data_loader, config):
        super(ResNet50_MI, self).__init__(config)
        
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.y_mi = None
        self.bi = None
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.train_step = None
        self.num_classes = config.num_classes

        self.build_model()
        self.init_saver()
    
    def build_model(self):
        """
        :return:
        """
        
        """
        Helper Variables
        """
        #self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        #self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)
        
        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y, self.y_mi, self.bi = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.y_mi)
        tf.add_to_collection('inputs', self.bi)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        
        with tf.variable_scope('network'):
            net, end_points = resnet_v2.resnet_v2_50(inputs = self.x, num_classes = None, global_pool = True)
    
            end_points['resnet_v2_50/pool5:0'] = net 
            print("Size after pool: ", net.shape)
    
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    
            end_points['resnet_v2_50/spatial_squeeze'] = net
            print("Size after squeeze: ", net.shape)
    
            if (self.config.mode == 'si_branch'):
                end_points['resnet_v2_50/output_si'] = fully_connected(net,
                                                                         self.num_classes, activation_fn=None,
                                                                         normalizer_fn=None, scope='logits_si')
                self.logits = end_points['resnet_v2_50/output_si']
                
                net = end_points['resnet_v2_50/output_si']
                
            if (self.config.mode == 'mi_branch'):
                net = self.mi_pool_layer(net, bag_indices = self.bi, pooling = self.config.pooling)
                end_points['resnet_v2_50/mi_pool1:0'] = net
                print("Size after MI: ", net.shape)
                
                end_points['resnet_v2_50/output_mi'] = fully_connected(end_points['resnet_v2_50/mi_pool1:0'],
                                                                         self.num_classes, activation_fn=None,
                                                                         normalizer_fn=None, scope='logits_mi')
                self.logits = end_points['resnet_v2_50/output_mi']
                
                net = end_points['resnet_v2_50/output_mi']
                
            if (self.config.mode == 'si_mi_branch'):
                end_points['resnet_v2_50/mi_pool1:0'] = self.mi_pool_layer(net,
                                                                           bag_indices = self.bi,
                                                                           pooling = self.config.pooling)
                
                end_points['resnet_v2_50/output_mi'] = fully_connected(end_points['resnet_v2_50/mi_pool1:0'],
                                                                         self.num_classes, activation_fn=None,
                                                                         normalizer_fn=None, scope='logits_mi')
                self.logits = end_points['resnet_v2_50/output_mi']
                
                end_points['resnet_v2_50/output_si'] = fully_connected(net,
                                                                         self.num_classes, activation_fn=None,
                                                                         normalizer_fn=None, scope='logits_si')
                self.logits_si = end_points['resnet_v2_50/output_si']
                
                net = end_points['resnet_v2_50/output_mi']
             
            
            end_points['predictions'] = tf.nn.softmax(net)
            
            with tf.variable_scope('out'):
                
                self.out = end_points['predictions']
            
            tf.add_to_collection('out', self.out)
            
            print("predictions out shape: ", self.out.shape)
            
            print("network output argmax resnet")
            with tf.variable_scope('out_argmax'):
                self.out_argmax = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='out_argmax')
                
                print("Arg Max Shape: ", self.out_argmax.shape)

        with tf.variable_scope('loss-acc'):
            
            if (self.config.mode == 'si_mi_branch'):
                self.update_beta_combined_cost()
                self.loss = combined_cost_function(self.y, self.logits_si, self.y_mi, self.logits,
                                                   beta = self.current_beta)
            else:    
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels = self.y, logits = self.logits)

            if (self.config.mode != 'si_branch'):
                self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_mi, self.out_argmax), tf.float32))
            else:
                self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))
            #self.acc = self.evaluate_accuracy(self.y, self.out_argmax,
            #                                  self.is_training, self.config.patch_count)

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