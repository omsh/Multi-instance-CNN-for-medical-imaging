import tensorflow as tf
from models.BaseModel import BaseModel
import tensorflow.contrib.slim as slim

class LeNet(BaseModel):
    def __init__(self, data_loader, config):
        super(LeNet, self).__init__(config)
        
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
            self.x, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        with tf.variable_scope('network'):
            net = slim.conv2d(self.x, 20, [5,5], scope='conv1')
            net = slim.max_pool2d(net, [2,2], scope='pool1')
            net = slim.conv2d(net, 50, [5,5], scope='conv2')
            net = slim.max_pool2d(net, [2,2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')
            net = slim.fully_connected(net, 500, scope='fc4')
            
            with tf.variable_scope('out'):
                self.out = slim.fully_connected(net, self.num_classes, activation_fn=None)
            
            tf.add_to_collection('out', self.out)
            
            with tf.variable_scope('out_argmax'):
                self.out_argmax = tf.argmax(self.out, axis=-1, output_type=tf.int64, name='out_argmax')

        with tf.variable_scope('loss-acc'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))

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