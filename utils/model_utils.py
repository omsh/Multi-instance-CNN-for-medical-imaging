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
                           beta = 0.5):
    
    print("y_si: ", y_si.shape)
    print("logits_si: ",logits_si.shape)
    print("y_mi: ", y_mi.shape)
    print("logits_mi: ", logits_mi.shape)
    
    si_loss = tf.losses.sparse_softmax_cross_entropy(labels = y_si, logits = logits_si)
    
    mi_loss = tf.losses.sparse_softmax_cross_entropy(labels = y_mi, logits = logits_mi)

    return (1-beta) * mi_loss + (beta) * si_loss