

# This file defines the optimization task

import tensorflow as tf
import numpy as np

def custom_gather(params, indices_edges):
    """
    Author: B. Donon
    
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
    Author: B. Donon
    
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



class Problem:
    
    """
    Author: B. Habib
    
    Custom loss function implementing the WLS algorithm as an optimization problem to minimize
    
    
    """


    def __init__(self, grid):

        self.name = 'DSSE'
        
        # Input dimensions
        self.d_in_A = 10        # 13 elements/[10 features]: from_bus, to_bus, [Re(Y), Im(Y), Re(Ys), Im(Ys), P, cov(P), Q, cov(Q), I, cov(I)], bool_closed, bool_edge_type, shift_rad
        self.d_in_B = 8         # 11 elements/[8 features]: port_bus, [V, cov(V), theta, cov(theta), P, cov(P), Q, cov(Q)], bool_slack, bool_zero_inj
        self.d_in = {}
        self.d_in[0] = self.d_in_A
        self.d_in[1] = self.d_in_B
        self.grid = grid
        # Output dimensions
        self.d_out = 2
        
        if self.grid=='cigre':
            self.num_buses = 15
        if self.grid=='ober':
            self.num_buses = 70
        
        # Case study
        self.grid = grid        # Fixing grid setup for other grid-related parameters
        
        # Initial values for latent output variables
        self.initial_U = np.array([0.,0.])
        
        # Setting normalization parameters
        if self.grid=='cigre':        
            # Normalization constants FOR CIGRE grid
            self.B_mean = np.array([ 1., 9926., -0.6, 2600., 1.25, 108376.4,  0.33,204485.98])
            self.B_std = np.array([0.03,241., 0.03,2600.   , 2.66, 277250.56, 0.62,390057.16])
          
            self.A_mean = np.array([0.84, -1.2, 0., 4.4e-5 ,0.6, 23323.5, 0.2,  126025.6, 0.03, 14166381.]) 
            self.A_std = np.array([0.77, 1.09, 1., 5.4e-5  ,0.35,20463.7, 0.1, 88732.7, 0.025, 23967946.])
            
            self.i_max = np.concatenate([np.ones(10) * 0.145, np.ones(2) * 0.195], axis=0)
            
        if self.grid=='ober':        
            # Normalization constants for Oberrhein grid
            self.B_mean = np.array([ 1.01,4258., -2.57, 4258. * 2.5, 0.23, 11560.,  0.07,   17857.])
            self.B_std = np.array([0.03, 4056., 0.066,  4056., 0.21, 28102., 0.04, 24661.])
           
            self.A_mean = np.array([10., -7.76, 0.    , 4.6e-5, 0.2, 5828., 0.    , 20791., 0.03, 14076696.])
            self.A_std = np.array([5.74, 4.3, 1.16e-07, 4.6e-5, 1.55, 13285., 0.32, 27126., 0.03, 24926946.])
           
            self.i_max = np.concatenate([np.ones(37) * 0.362, np.ones(2) * 0.645, np.ones(1) * 0.421,\
                                         np.ones(1) * 0.645, np.ones(2) * 0.362, np.ones(1) * 0.645, np.ones(5) * 0.362,\
                                            np.ones(1) * 0.421, np.ones(1) * 0.362, np.ones(1) * 0.421, np.ones(5) * 0.362,\
                                                np.ones(1) * 0.421, np.ones(1) * 0.421, np.ones(9) * 0.362], axis=0)
    

    def cost_function(self, A_flat, B_flat, y, lamda, A0, B0):
        
        
        """ Processing input features, parameters and output variables """
        
        V_n = 20.       # Nominal voltage of the grid
        V_h = 110.      # Nominal voltage at the high-voltage side
        n = V_h/V_n     # Transformer ratio
        
        if self.grid=='cigre':
            n_trafo = 2
        if self.grid=='ober':
            n_trafo = 1 # Number of trafos to separate from lines later
        
                
        # Reshape the iterator
        minibatch_size_ = tf.shape(A_flat)[0]
        
        # Getting and reshaping normalization constants
        A_mean_tf = tf.ones([minibatch_size_, tf.shape(A0)[1], 1]) * \
        tf.reshape(tf.constant(self.A_mean, dtype=tf.float32), [1, 1, self.d_in[0]])
        A_std_tf = tf.ones([minibatch_size_, tf.shape(A0)[1], 1]) * \
        tf.reshape(tf.constant(self.A_std, dtype=tf.float32), [1, 1,self.d_in[0]])
        B_mean_tf = tf.ones([minibatch_size_, tf.shape(B0)[1], 1]) * \
        tf.reshape(tf.constant(self.B_mean, dtype=tf.float32), [1, 1, self.d_in[1]])
        B_std_tf = tf.ones([minibatch_size_, tf.shape(B0)[1], 1]) * \
        tf.reshape(tf.constant(self.B_std, dtype=tf.float32), [1, 1, self.d_in[1]])
      

        # Reshaping and un-normalizing input features
        a_meas = tf.reshape(A_flat, [minibatch_size_, -1, self.d_in[0]])
        A_meas = (a_meas  * A_std_tf +A_mean_tf) * tf.cast(a_meas!=0., tf.float32)              # tf.float32, [n_samples, n_edges, 10]
        b_meas = tf.reshape(B_flat, [minibatch_size_, -1, self.d_in[1]])
        B_meas = (b_meas  * B_std_tf +B_mean_tf) * tf.cast(b_meas!=0., tf.float32)              # tf.float32, [n_samples, n_edges, 8]
        
        
        
        # Getting and un-normalizing output values (voltage amplitude and angles)
        U1 = self.B_std[0] * y[:,:,0:1] + self.B_mean[0]                        # tf.float32, [n_samples, n_buses,1]
        U2 = (self.B_std[2] * y[:,:,1:] + self.B_mean[2])  * (1.-B0[:,:,-1:])   # tf.float32, [n_samples, n_buses,1], fixing slack angle to 0
        
        
        # Gather instances dimensions (samples, nodes)
        n_samples = tf.shape(U1)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(U1)[1]                                    # tf.int32, [1]
        
        
        
        # Rebuild of A and B matrice
        meas_theta = tf.expand_dims(B_meas[:,:,2]  * (1.-B0[:,:,-1]),axis=2)  # Enforcing a 0 value in measurement of theta at the slack  
        
        A = tf.concat([A0[:,:,0:2], A_meas, A0[:,:,6:]], axis=2)                                            # tf.float32, [n_samples, n_edges, 13]
        B = tf.concat([B0[:,:,0:1], B_meas[:,:,0:2],  meas_theta, B_meas[:,:,3:], B0[:,:,1:]], axis=2)      # tf.float32, [n_samples, n_buses, 11]


        # Getting topology from matrice A
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]

        # Extact edge characteristics from A matrix
        Y1_ij = A[:,:,2:3]                                          # tf.float32, [n_samples, n_edges, 1] = Re(Y)
        Y2_ij = A[:,:,3:4]                                          # tf.float32, [n_samples, n_edges, 1] = Im(Y)    

        Ys1_ij =  A[:,:,4:5]                                        # tf.float32, [n_samples, n_edges, 1] = Re(Ys)                                   
        Ys2_ij =  A[:,:,5:6]                                        # tf.float32, [n_samples, n_edges, 1] = Im(Ys)      



        # Get (inverse of) covariance matrix per measurements type      
        cov_v = B[:,:,2:3]                                          # tf.float32, [n_samples, n_buses, 1]      
        cov_theta = B[:,:,4:5]        
        cov_P = B[:,:,6:7]        
        cov_Q = B[:,:,8:9] 
       
        cov_PL = A[:,:,7:8]                                         # tf.float32, [n_samples, n_edges, 1]     
        cov_QL = A[:,:,9:10]      
        cov_IL = A[:,:,11:12]
       
      
        # Gather V and theta on both sides of each edge
        U1_i = custom_gather(U1, indices_from)                      # tf.float32, [n_samples , n_edges, 1], in p.u.
        U2_i = custom_gather(U2, indices_from)                      # tf.float32, [n_samples , n_edges, 1], in p.u.
        U1_j = custom_gather(U1, indices_to)                        # tf.float32, [n_samples , n_edges, 1], in rad
        U2_j = custom_gather(U2, indices_to)                        # tf.float32, [n_samples , n_edges, 1], in rad


        """ Compute h(U) = V_i, theta_i, P_i, Q_i, P_ij, Q_ij, I_ij """
        
        # Flow from bus       
        P_ij_from = - U1_i * U1_j * ( Y1_ij * tf.math.cos(U2_i - U2_j - A[:,:,14:]) + Y2_ij * tf.math.sin(U2_i - U2_j - A[:,:,14:])) \
                    + (Y1_ij + Ys1_ij/2) * U1_i**2                                                                                      # tf.float32, [n_samples , n_edges, 1], in p.u.
                   
        Q_ij_from =  U1_i * U1_j * ( - Y1_ij * tf.math.sin(U2_i - U2_j - A[:,:,14:]) + Y2_ij * tf.math.cos(U2_i - U2_j - A[:,:,14:])) \
                    - (Y2_ij + Ys2_ij/2) * U1_i**2                                                                                      # tf.float32, [n_samples , n_edges, 1], in p.u.

        
        # Casting to complex value for correct handling with tensorflow
        P_ij_from_comp = tf.cast(P_ij_from, dtype=tf.complex64)
        Q_ij_from_comp = tf.cast(Q_ij_from, dtype=tf.complex64)
        U1_i_comp = tf.cast(U1_i, dtype=tf.complex64)
        
        imag = tf.complex(tf.constant(0.),tf.constant(1.))
        I_ij_from = tf.math.abs((P_ij_from_comp-imag*Q_ij_from_comp)/(U1_i_comp*np.sqrt(3)))          # tf.complex64, [n_samples , n_edges, 1], in p.u.
                                
       
        # Casting I to tf.float32, and dividing by transformer ratio for the transformers
        I_ij_from = tf.concat([tf.cast(I_ij_from[:,:-n_trafo], dtype=tf.float32), tf.cast(I_ij_from[:,-n_trafo:] / n, dtype=tf.float32)],axis=1)
        
        
        
        
        # Flow to bus
        P_ij_to = - U1_i * U1_j * ( Y1_ij * tf.math.cos(U2_i - U2_j - A[:,:,14:]) - Y2_ij * tf.math.sin(U2_i - U2_j - A[:,:,14:])) \
                  + (Y1_ij + Ys1_ij/2) * U1_j**2                                                                                        # tf.float32, [n_samples , n_edges, 1], in p.u.
               
        Q_ij_to =  U1_i * U1_j * ( Y1_ij * tf.math.sin(U2_i - U2_j - A[:,:,14:]) + Y2_ij * tf.math.cos(U2_i - U2_j - A[:,:,14:])) \
            - (Y2_ij + Ys2_ij/2) * U1_j**2                                                                                              # tf.float32, [n_samples , n_edges, 1], in p.u.
 
     

        # Casting to complex value for correct handling with tensorflow
        P_ij_to_comp = tf.cast(P_ij_to, dtype=tf.complex64)
        Q_ij_to_comp = tf.cast(Q_ij_to, dtype=tf.complex64)
        U1_j_comp = tf.cast(U1_j, dtype=tf.complex64)
        
        imag = tf.complex(tf.constant(0.),tf.constant(1.))
        I_ij_to = tf.math.abs((P_ij_to_comp-imag*Q_ij_to_comp)/(U1_j_comp*np.sqrt(3)))                # tf.complex64, [n_samples , n_edges, 1], in p.u.
        
        #Casting back to tf.float32
        I_ij_to= tf.cast(I_ij_to, dtype=tf.float32)
        
        
        #Summing flow to balance in buses, negative signs in sum to follow conventions from PandaPower
        P_i = -custom_scatter(indices_to, P_ij_to, [n_samples, n_nodes, 1]) - custom_scatter(indices_from, P_ij_from, [n_samples, n_nodes, 1])  # tf.float32, [n_samples, n_nodes, 1], in p.u.
        Q_i = -custom_scatter(indices_to, Q_ij_to, [n_samples, n_nodes, 1]) - custom_scatter(indices_from, Q_ij_from, [n_samples, n_nodes, 1])  # tf.float32, [n_samples, n_nodes, 1], in p.u.
        

        """ WLS calculation: Weighted MSE for each single measurement and summation """
        

        # Compute weighted errors
        delta_v = (B[:,:,1:2] - U1)**2 * cov_v
        delta_theta = (B[:,:,3:4] - U2)**2 * cov_theta
        delta_PL = (A[:,:,6:7]/V_n**2 - P_ij_from)**2 * cov_PL * V_n**2
        delta_QL = (A[:,:,8:9]/V_n**2 - Q_ij_from)**2 * cov_QL * V_n**2
        delta_IL = (A[:,:,10:11]/V_n  - I_ij_from)**2 * cov_IL
        delta_P = ((1-B[:,:,10:11]) * (B[:,:,5:6]/V_n**2 - P_i)**2 + B[:,:,10:11] * (0. - P_i)**2) * cov_P * V_n**2   
        delta_Q = (((1-B[:,:,10:11]) * (B[:,:,7:8]/V_n**2 - Q_i)**2 + B[:,:,10:11] * (0. - Q_i)**2) * cov_Q) * V_n**2
        

        # Compute valid number of measurement entries
        non_zero_v = tf.cast(delta_v != 0., tf.float32)
        non_zero_theta = tf.cast(delta_theta != 0., tf.float32)
        non_zero_P = tf.cast(delta_P != 0., tf.float32)
        non_zero_Q = tf.cast(delta_Q != 0., tf.float32)
        non_zero_IL = tf.cast(delta_IL != 0., tf.float32)
        non_zero_PL = tf.cast(delta_PL != 0., tf.float32)
        non_zero_QL = tf.cast(delta_QL != 0., tf.float32)
        

        # Compute cost per sample   
        cost_P = tf.reduce_sum(delta_P, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_P, axis=[1,2]))
        cost_v = tf.reduce_sum(delta_v, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_v, axis=[1,2]))
        cost_theta = tf.reduce_sum(delta_theta, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_theta, axis=[1,2]))
        cost_Q = tf.reduce_sum(delta_Q, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_Q, axis=[1,2]))
        cost_PL = tf.reduce_sum(delta_PL, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_PL, axis=[1,2]))
        cost_QL = tf.reduce_sum(delta_QL, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_QL, axis=[1,2])) 
        cost_IL = tf.reduce_sum(delta_IL, axis=[1,2]) / tf.math.maximum(tf.constant([1.]),tf.reduce_sum(non_zero_IL, axis=[1,2]) )

        
        # Add constraint: U1 in [0.95,1.05] p.u.
        regularizer = tf.reduce_sum(((tf.nn.relu(0.95 - U1) + tf.nn.relu(U1-1.05)) * tf.math.reduce_max(cov_v)), axis=[1,2]) #+ \

        # Compute loading and add constraint: loading < 100%
        i_ka = tf.maximum(I_ij_from, I_ij_to)[:,:-2,:] * V_n
        i_kat = tf.maximum(I_ij_from[:,-2:,:] * 110/25, I_ij_to[:,-2:,:] *20/25) *V_n 
           
        i_max = tf.ones([tf.shape(A0)[0], 1,1]) * tf.reshape(tf.constant(self.i_max, dtype=tf.float32), [1, tf.shape(A0)[1]-2,1])
            
        loading = tf.concat([i_ka /i_max, i_kat],axis=1)
        
        regularizer2 = tf.reduce_sum(tf.nn.relu(loading - 1.)**2  * tf.math.reduce_max(cov_IL), axis=[1,2])  
        
        # Compute difference of voltage angle across the lines and add constraint: \Delta U2 < 15 deg. = 0.25 rad
        U2_e = U2_i - U2_j - A[:,:,14:]
        regularizer3 = tf.reduce_sum((tf.nn.relu(-0.25 - U2_e) + tf.nn.relu(U2_e-0.25)) * tf.math.reduce_max(cov_IL), axis=[1,2]) 
        
        # Summing all terms
        cost_per_sample =  cost_P + cost_Q + cost_PL + cost_QL + cost_theta + cost_IL + cost_v
        
        # Adding constraints, with lamda as hyper-parameter
        cost_per_sample += lamda * (regularizer + regularizer2 + regularizer3)
        
        
        return cost_per_sample  

















