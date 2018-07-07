# Adapted from: https://github.com/dalgu90/resnet-18-tensorflow

import tensorflow as tf
import numpy as np
from models.BaseModel import BaseModel
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers import fully_connected
from utils.model_utils import acc_majority_class, combined_cost_function
import utils.resnet18_utils as utils
import logging
import pprint

# TODO:
# CLEAN PARAMETERS AND FUNCTIONS
# INCLUDE PRETRAINED WEIGHTS
# CHECK NECESSITY OF FLOP-UPDATE

class ResNet18_MI(BaseModel):
    def __init__(self, data_loader, config):
        super(ResNet18_MI, self).__init__(config)

        self.data_loader = data_loader

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
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

        self.build_model()
        self.init_saver()

    def build_tower(self):
        print('Building model')

        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        with tf.variable_scope('conv1'):
            x = self._conv(self.x, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
         #   x = self._fc(x, self.num_classes)

        logits = x

        end_points = {}

        return logits, end_points


    def build_model(self):
        """
        :return:
        """

        """
        Helper Variables
        """
        # self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        # self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
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
            net, end_points = self.build_tower()

            end_points['resnet_18/pool5:0'] = net
            print("Size after pool: ", net.shape)

            #net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

            end_points['resnet_18/spatial_squeeze'] = net
            print("Size after squeeze: ", net.shape)

            if (self.config.mode == 'si_branch'):
                end_points['resnet_18/output_si'] = fully_connected(net,
                                                                       self.num_classes, activation_fn=None,
                                                                       normalizer_fn=None, scope='logits_si')
                self.logits = end_points['resnet_18/output_si']

                net = end_points['resnet_18/output_si']

            if (self.config.mode == 'mi_branch'):
                net = self.mi_pool_layer(net, bag_indices=self.bi, pooling=self.config.pooling)
                end_points['resnet_18/mi_pool1:0'] = net
                print("Size after MI: ", net.shape)

                end_points['resnet_18/output_mi'] = fully_connected(end_points['resnet_18/mi_pool1:0'],
                                                                       self.num_classes, activation_fn=None,
                                                                       normalizer_fn=None, scope='logits_mi')
                self.logits = end_points['resnet_18/output_mi']

                net = end_points['resnet_18/output_mi']

            if (self.config.mode == 'si_mi_branch'):
                end_points['resnet_18/mi_pool1:0'] = self.mi_pool_layer(net,
                                                                           bag_indices=self.bi,
                                                                           pooling=self.config.pooling)

                end_points['resnet_18/output_mi'] = fully_connected(end_points['resnet_18/mi_pool1:0'],
                                                                       self.num_classes, activation_fn=None,
                                                                       normalizer_fn=None, scope='logits_mi')
                self.logits = end_points['resnet_18/output_mi']

                end_points['resnet_18/output_si'] = fully_connected(net,
                                                                       self.num_classes, activation_fn=None,
                                                                       normalizer_fn=None, scope='logits_si')
                self.logits_si = end_points['resnet_18/output_si']

                net = end_points['resnet_18/output_mi']

            end_points['predictions'] = tf.nn.softmax(net)

            with tf.variable_scope('out'):
#                self.out = tf.nn.softmax(self.logits, dim=-1)
                self.out = end_points['predictions']

            tf.add_to_collection('out', self.out)

            print("predictions out shape: ", self.out.shape)

            print("network output argmax resnet-18")
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

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_training, name=name)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        return x

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)