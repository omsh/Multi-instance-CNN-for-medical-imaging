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


def combined_cost_function(y_si, logits_si,
                           y_mi, logits_mi,
                           initial_beta = 0.5, epoch_i=0, n_epochs = 200):
    
    print("y_si: ", y_si.shape)
    print("logits_si: ",logits_si.shape)
    print("y_mi: ", y_mi.shape)
    print("logits_mi: ", logits_mi.shape)
    si_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_si, logits = logits_si)
    
    mi_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_mi, logits = logits_mi)

    current_beta = initial_beta * (1 + epoch_i / n_epochs)
    
    current_beta = tf.cast(current_beta, dtype = tf.float32)
    
    return current_beta * mi_loss + (1 - current_beta) * si_loss