import tensorflow as tf


def acc_majority_class(labels, predictions, n_patches):
    
    def compute_majority_pred(x):
        values, _, counts = tf.unique_with_counts(x)
        return values[tf.argmax(counts)]

    maj_preds = tf.map_fn(compute_majority_pred, 
                         tf.reshape(predictions, shape=(-1, n_patches)))

    bags_labels = tf.map_fn(lambda x: x[0],
                            tf.reshape(labels, shape=(-1, n_patches)))

    return tf.reduce_mean(tf.cast(tf.equal(bags_labels, maj_preds), tf.float32))