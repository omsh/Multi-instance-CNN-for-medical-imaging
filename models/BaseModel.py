import tensorflow as tf
from utils.model_utils import acc_majority_class
import numpy as np

class BaseModel:
    def __init__(self, config):
        self.config = config

        # Attributes needed for global_step and global_epoch
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_tensor = None
        self.global_step_tensor = None
        self.increment_global_step_tensor = None
        
        self.current_beta = self.config.beta

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        
        self.init_lr_scheduler()

        # init optimizer
        self.init_optimizer()
        
        # init lr scheduler

        # save attribute .. NOTE DON'T FORGET TO CONSTRUCT THE SAVER ON YOUR MODEL
        self.saver = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
    
    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            # this operator if you wanna increment the global_step_tensor by yourself instead of incrementing it
            # by .minimize function in the optimizers of tensorflow
            #self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)
    
    def init_lr_scheduler(self):
        if (self.config.lr_scheduler_type == 'exponential_decay'):
            self.config.optim_params['learning_rate'] = tf.train.exponential_decay(**self.config.lr_scheduler_params,
                                                  global_step = self.global_step_tensor)
            
        elif (self.config.lr_scheduler_type == 'natural_exp_decay'):
            self.config.optim_params['learning_rate'] = tf.train.natural_exp_decay(**self.config.lr_scheduler_params,
                                                  global_step = self.global_step_tensor)
            
        elif (self.config.lr_scheduler_type == 'inverse_time_decay'):
            self.config.optim_params['learning_rate'] = tf.train.inverse_time_decay(**self.config.lr_scheduler_params,
                                                  global_step = self.global_step_tensor)
        else:
            return

            
    def init_optimizer(self):
        
        if (self.config.optimizer_type == 'Adam'):
            self.optimizer = tf.train.AdamOptimizer(**self.config.optim_params)
        elif (self.config.optimizer_type == 'MomentumOptimizer'):
            self.optimizer = tf.train.MomentumOptimizer(**self.config.optim_params)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(**self.config.optim_params)
            
    def mi_pool_layer_x(self, input_vector, bag_indices, pooling = 'average'):
        print("Input to pool: ", input_vector.shape)
        
        unique_bag_ids, _ = tf.unique(bag_indices)
        indices = tf.map_fn(lambda z: tf.squeeze(tf.where(tf.equal(z, bag_indices))), unique_bag_ids)

        reshaped = tf.gather(input_vector, indices, axis=0)

        # pool, iterate over each bag in the batch and compute pooling
        if (pooling == 'max'):
            pooled = tf.map_fn(lambda z: tf.reduce_max(z, axis=[0]), reshaped)
        if (pooling == 'lse'):
            pooled = tf.map_fn(lambda z: tf.reduce_logsumexp(z, axis=[0]), reshaped)
        else:
            pooled = tf.map_fn(lambda z: tf.reduce_mean(z, axis=[0]), reshaped)
    
        print("Pooled shape: ", pooled.shape)
        pooled = tf.reshape(pooled, shape=(-1, 2048))
        print("Pooled reshaped: ", pooled.shape)
        return pooled
    
    def mi_pool_layer(self, input_vector, bag_indices, pooling = 'average'):
        _, idx = tf.unique(bag_indices)
        reshaped = tf.dynamic_partition(input_vector, idx, num_partitions=self.config.batch_size)

        if (pooling == 'max'):
            pooled = [tf.reduce_max(x, axis=[0]) for x in reshaped]
        elif (pooling == 'lse'):
            pooled = [tf.reduce_logsumexp(x, axis=[0]) for x in reshaped]
        else:
            pooled = [tf.reduce_mean(x, axis=[0]) for x in reshaped]
            
        return tf.stack(pooled)

    def evaluate_accuracy(self, y, preds, is_training, n_patches):
        return tf.cond(is_training,
                       lambda: tf.reduce_mean(tf.cast(tf.equal(y, preds), tf.float32)),
                       lambda: acc_majority_class(y, preds, n_patches))

    def update_beta_combined_cost(self):
        self.current_beta =  self.current_beta * (self.config.beta_decay ** (self.cur_epoch_tensor / self.config.num_epochs))
        self.current_beta = tf.cast(self.current_beta, dtype = tf.float32)
    
    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError