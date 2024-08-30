import sys
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2

def custom_gather(params, indices_edges):
    """
    This computational graph module performs the gather_nd operation while taking into account
    the batch dimension.

    Inputs
        - params : tf tensor of shape [n_samples, n_nodes, d_out], and type tf.float32
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
    Output
        - tf tensor of shape [n_samples, n_edges, d_out] and type tf.float32
    """

    # Get all relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = tf.shape(params)[1]                                       # tf.int32, [1]
    n_edges = tf.shape(indices_edges)[1]                                # tf.int32, [1]
    d_out = tf.shape(params)[2]                                         # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Flatten the indices
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the node parameters
    params_flat = tf.reshape(params, [-1, d_out])                       # tf.float32, [n_samples * n_nodes, d_out]

    # Perform the gather operation
    gathered_flat = tf.gather_nd(params_flat, indices_flat)             # tf.float32, [n_samples * n_edges, d_out]

    # Un-flatten the result of the gather operation
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])   # [n_samples , n_edges, d_out]

    return gathered

def custom_scatter(indices_edges, params, shape):
    """
    This computational graph module performs the scatter_nd operation while taking into account
    the batch dimension. Note that here we can also have d instead of d_F

    Inputs
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - params : tf tensor of shape [n_samples, n_edges, d_F], and type tf.float32
        - shape : tf.tensor of shape [3]
    Output
        - tf tensor of shape [n_samples, n_nodes, n_nodes, d_F] and type tf.float32
    """

    # Get all the relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = shape[1]                                                  # tf.int32, [1]
    n_edges = tf.shape(params)[1]                                       # tf.int32, [1]
    d_F = tf.shape(params)[2]                                           # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Stack batch and edge dimensions
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the edge parameters
    params_flat = tf.reshape(params, [n_samples*n_edges, d_F])          # tf.float32, [n_samples * n_edges, d_F]

    # Perform the scatter operation
    scattered_flat = tf.scatter_nd(indices_flat, params_flat, shape=[n_samples*n_nodes, d_F])   
                                                                        # tf.float32, [n_samples * n_nodes, d_F]

    # Un-flatten the result of the scatter operation
    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, d_F])   # tf.float32, [n_samples, n_nodes, d_F]

    return scattered

class FullyConnected(keras.layers.Layer):
    """
    Simple fully connected block. Serves as an elementary learning block in our neural network architecture.

    Params
        - latent_dimension : integer, number of hidden neurons in every intermediate layer
        - hidden : integer, number of layers. If set to 1, there is no hidden layer
        - non_lin : string, chosen non linearity
        - input_dim : integer, dimension of the input; if not specified, set to latent_dimension
        - output_dim : integer, dimension of the output; if not specified, set to latent_dimension
        - name : string, name of the neural network block
    """
    
    def __init__(self, 
        latent_dimension=10,
        hidden_layers=3,
        non_lin='tanh', 
        input_dim=None,
        output_dim=None, 
        rate = 0.6,
        l2_reg = 0.04,
        name='encoder'):
        super(FullyConnected, self).__init__()
        
        # Get parameters
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.dense = {}
     #   self.bn = {}
        self.dropout = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rate = rate
        self.l2_reg = l2_reg
        # Convert str into an actual tensorflow operator
        

        if non_lin == 'tanh':
            self.non_lin = tf.tanh
            self.init = keras.initializers.GlorotNormal()
        elif non_lin == 'relu':
            self.non_lin = tf.keras.activations.relu
            self.init = keras.initializers.HeNormal()
        
        for i in range(hidden_layers-1):
            self.dense[str(i)] = keras.layers.Dense(units = self.latent_dimension, activation = self.non_lin, \
                                               kernel_initializer=self.init,bias_initializer='zeros',\
                                                   kernel_regularizer=l2(self.l2_reg))  
                
            self.dropout[str(i)] = keras.layers.Dropout(self.rate)
                       
        if self.output_dim == None:
            self.output_dim = self.latent_dimension
        self.dense[str(hidden_layers-1)] = keras.layers.Dense(units = self.output_dim, \
                                                              kernel_initializer=self.init,bias_initializer='zeros',\
                                                                 kernel_regularizer=l2(self.l2_reg))  

                            
    def call(self, h, training=None):
        """
        Builds the computational graph.
        """

        n_samples = tf.shape(h)[0]
        n_elem = tf.shape(h)[1]
        d = tf.shape(h)[2]

        h = tf.reshape(h, [-1, d])

        for layer in range(self.hidden_layers-1):
            
            h = self.dense[str(layer)](h)
            
            if training:
                h = self.dropout[str(layer)](h)
                
                
        h_out = self.dense[str(self.hidden_layers-1)](h)

        return tf.reshape(h_out, [n_samples, n_elem, -1])

