import tensorflow as tf

def mlp(hidden_neurons, activation='elu', regularization=None, use_bias=True):
    '''
    Creates a multilayer perzeptron 
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n, activation=activation, 
            activity_regularizer=regularization, use_bias=use_bias)
        for n in hidden_neurons ])
    return model

def repeat(x, repeats, axis):
    '''
    Behaves like numpy repeat. This function got 
    introduced in tf 2.1.0, but we have to use 2.0.0.
    '''
    shape = x.shape
    x = tf.expand_dims(x, axis+1)
    t = tf.eye(len(x.shape), dtype=tf.int64)
    t = t[axis+1] * (repeats - 1) + 1
    x = tf.tile(x, t)
    new_shape = shape * t[1:]
    return tf.reshape(x, new_shape)