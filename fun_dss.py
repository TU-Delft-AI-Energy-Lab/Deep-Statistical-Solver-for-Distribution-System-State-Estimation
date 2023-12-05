import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from layers_tf2_improved import custom_gather, custom_scatter
from layers_tf2_improved import FullyConnected as fc_improved

import pandas as pd


def get_models(num_models, saved_model, problem, hparam, data_dir, case, grid):
    models = {}

    for m in range(num_models):

        try:

            models[m] = tf.keras.models.load_model(saved_model)

        except OSError:

            print("no model here!")

            models[m] = DeepStatisticalSolver2(name="test",
                                               problem=problem,
                                               latent_dimension=hparam['latent dimension'],
                                               hidden_layers=hparam['hidden layers'],
                                               time_step_size=1. / hparam['steps'],
                                               rate=hparam['dropout rate'],
                                               l2_reg=hparam['l2 regularizer'],
                                               non_lin=hparam['non linearity'])

            optimizer = tf.keras.optimizers.Adamax(hparam['learning rate'])
            models[m] = train_model(models[m], problem, hparam['learning rate'], hparam['penalty coefficient'],
                                    hparam['clipping norm'], hparam['epochs'], hparam['minibatch size'],
                                    data_dir, case, optimizer, grid)
            test_model(models[m], problem, hparam['minibatch size'], data_dir, case, grid)

    return models


def DSSData_from_PPNet(net, pm_bus, meas_bus, meas_line, meas_error, labels):
    """
    - Function taking a PandaPower's network and convert it to a DSS data sample. Pseudomeasurements
     and measurements location are provided as well for labelling and input
    
    - Input:
        - net: PandaPower Network, with detailed topology, buses, trafos and lines parameters, and load and generators.
        
        - pm_bus = {"p_mw" : pm_p, "q_mvar": pm_q} Dataframe of pseudomeasurements of bus p and q injection. 
            - df: [index = net.bus.index, column = p_mw, q_mvar]
            
        - meas_bus = {"vm_pu": meas_v, "va_rad": meas_phi, "p_mw": meas_p, "q_mvar": meas_q}: DataFrame
         of measurements v, phi, p and q on buses.
            - df: [index = net.bus.index, column = vm_pu, va_rad, p_mw, q_mvar]
            
        - meas_line = {"p_from_mw": meas_pf, "q_from_mvar": meas_qf, "i_from_ka": meas_if}:
            measurements p, q, i (from (and to)) on lines
        
        - meas_error = [error_v, error_phi, error_p, error_pl, error_i, error_pm, error_zero_inj],
         with error_p = error_q = error_pflow = error_qflow
        
        - labels = {"vm_pu": node_vm, "va_angle": node_va}
     
    - Output: 
        - A: Matrice of line parameters including both topology parameters and measurement features
        - B: Matrice of buses parameters including both topology parameters and measurement features     
        - U: Labels of buses voltage amplitude and angle
    
    """

    net.bus["name"] = np.arange(net.bus.index.size)
    new_node_index = np.arange(net.bus.index.size)

    slack_bus = net.ext_grid['bus']
    load_bus = net.load.sort_values('bus')['bus']
    sgen_bus = net.sgen.sort_values('bus')['bus']

    pm_p = pm_bus["p_mw"].values.astype(float)
    pm_q = pm_bus["q_mvar"].values.astype(float)

    meas_v = meas_bus["vm_pu"].values.astype(float)
    error_v = np.absolute(meas_error[0] * np.nan_to_num(meas_v))

    meas_phi = meas_bus["va_rad"].values.astype(float)
    error_phi = np.absolute(meas_error[1] * np.nan_to_num(meas_phi))

    meas_p = meas_bus["p_mw"].values.astype(float)
    error_p = np.absolute(meas_error[2] * np.nan_to_num(meas_p))

    meas_q = meas_bus["q_mvar"].values.astype(float)
    error_q = np.absolute(meas_error[2] * np.nan_to_num(meas_q))

    aggr_p = np.nan_to_num(meas_p) + pm_p * (np.isnan(meas_p))
    error_p += np.absolute(meas_error[5] * (np.isnan(meas_p)) * pm_p)

    aggr_q = np.nan_to_num(meas_q) + pm_q * (np.isnan(meas_q))
    error_q += np.absolute(meas_error[5] * (np.isnan(meas_q)) * pm_q)

    error_zero_inj = np.absolute(meas_error[6] * np.isnan(meas_p))

    bool_zero_inj = pd.DataFrame(data={"type": 0}, index=net.bus.index)

    trafo_bus = np.concatenate([net.trafo["hv_bus"].to_numpy(), net.trafo["lv_bus"].to_numpy()])

    bool_slack = pd.DataFrame(data={"type": 0}, index=net.bus.index)
    bool_trafo = pd.DataFrame(data={"type": 0}, index=net.bus.index)

    for i in net.bus.index:

        if any(trafo_bus == i):
            bool_trafo.loc[i] = 1

        if not (any(slack_bus == i) or (any(load_bus == i) or any(sgen_bus == i))):
            bool_zero_inj.loc[i] = 1

        if any(slack_bus == i):
            bool_slack.loc[i] = 1

    bool_trafo = bool_trafo.values[:, 0]
    bool_slack = bool_slack.values[:, 0]
    bool_zero_inj = bool_zero_inj.values[:, 0]

    error_p = (1 - bool_zero_inj) * error_p + bool_zero_inj * error_zero_inj
    error_q = (1 - bool_zero_inj) * error_q + bool_zero_inj * error_zero_inj

    node_features = np.nan_to_num(np.array([new_node_index, meas_v, error_v, meas_phi, error_phi, aggr_p, \
                                            error_p, aggr_q, error_q, bool_trafo, bool_zero_inj, bool_slack]).T)

    p_line_from = meas_line["p_from_mw"].values.astype(float)
    error_pl = np.concatenate([meas_error[3] * (np.nan_to_num(p_line_from)[:net.line.index.size]), \
                               np.ones(net.trafo.index.size)])

    q_line_from = meas_line["q_from_mvar"].values.astype(float)
    error_ql = np.concatenate([meas_error[3] * (np.nan_to_num(q_line_from)[:net.line.index.size]), \
                               np.ones(net.trafo.index.size)])

    i_line_from = meas_line["i_from_ka"].values.astype(float)
    error_il = np.concatenate([meas_error[4] * (np.nan_to_num(i_line_from)[:net.line.index.size]), \
                               np.ones(net.trafo.index.size)])

    edge_length = net.line["length_km"]
    edge_r = net.line["r_ohm_per_km"] * edge_length
    edge_x = net.line["x_ohm_per_km"] * edge_length

    edge_c = net.line["c_nf_per_km"] * edge_length
    edge_b = -2 * np.pi * net.f_hz * edge_c * 1e-9
    edge_g = net.line["g_us_per_km"] * edge_length * 1e-6
    # edge_max_i = net.line["max_i_ka"]
    edge_df = net.line["df"]
    edge_parallel = net.line["parallel"]

    net.trafo.index += net.line.index.size

    t_r = (net.trafo["vkr_percent"] / 100) * (net.sn_mva / net.trafo["sn_mva"])
    t_z = (net.trafo["vk_percent"] / 100) * (net.sn_mva / net.trafo["sn_mva"])
    t_x_square = t_z.pow(2) - t_r.pow(2)
    t_x = t_x_square.pow(0.5)

    t_g = (net.trafo["pfe_kw"] / 1000) * (net.sn_mva / net.trafo["sn_mva"] ** 2)
    t_y = (net.trafo["i0_percent"] / 100)
    t_b_square = t_y ** 2 - t_g ** 2
    t_b = t_b_square.pow(0.5)

    Z_trafo = (net.trafo["vn_lv_kv"] ** 2 * net.sn_mva)

    t_R = t_r * Z_trafo
    t_X = t_x * Z_trafo
    t_G = t_g / Z_trafo
    t_B = t_b / Z_trafo

    edge_r = pd.concat([edge_r, t_R])
    edge_x = pd.concat([edge_x, t_X])
    edge_b = pd.concat([edge_b, t_B])
    edge_g = pd.concat([edge_g, t_G])

    t_parallel = net.trafo["parallel"]
    t_df = net.trafo["df"]
    t_phase_shift = net.trafo["shift_degree"] * np.pi / 180

    edge_phase_shift = np.concatenate((np.zeros(net.line.index.size), t_phase_shift.values))

    edge_parallel = pd.concat([edge_parallel, t_parallel])
    edge_df = pd.concat([edge_df, t_df])

    edge_type = pd.DataFrame({"type": np.append(np.zeros(net.line.index.size), np.ones(net.trafo.index.size))})

    edge_switch = pd.DataFrame(data={"closed": True}, index=np.concatenate([net.line.index, net.trafo.index]))
    edge_ind = -1
    for i in net.switch.index:
        old_ind = edge_ind
        edge_ind = net.switch["element"][i]

        if edge_ind == old_ind:
            edge_switch.loc[edge_ind] = (net.switch["closed"][i] and net.switch["closed"][i - 1])
        else:
            edge_switch.loc[edge_ind] = net.switch["closed"][i]

    bool_closed_line = edge_switch["closed"].values.astype("float64")

    edge_source = pd.concat([net.line["from_bus"], net.trafo["hv_bus"]])
    edge_target = pd.concat([net.line["to_bus"], net.trafo["lv_bus"]])

    edge_source_index = edge_source.values
    edge_target_index = edge_target.values

    new_edge_source = net.bus['name'][edge_source_index].values.astype(float)
    new_edge_target = net.bus['name'][edge_target_index].values.astype(float)

    edge_Z = edge_r.values + 1j * edge_x.values
    edge_Y = np.reciprocal(edge_Z)
    edge_Ys = edge_g - 1j * edge_b

    edge_features = np.array(
        [new_edge_source, new_edge_target, np.real(edge_Y), np.imag(edge_Y), np.nan_to_num(np.real(edge_Ys)),
         np.nan_to_num(np.imag(edge_Ys)), np.nan_to_num(p_line_from), \
         error_pl, np.nan_to_num(q_line_from), error_ql, np.nan_to_num(i_line_from), error_il, \
         bool_closed_line, edge_type["type"].values, edge_phase_shift]).T

    net.trafo.index -= net.line.index.size

    B = node_features
    A = edge_features

    U = labels.values
    return A, B, U


def preprocess_data(A_flat, B_flat, problem, grid):
    """
    
    Function to pre-process the data:
        - Reshaping the data matrices
        - Removing open lines
        - Calculating weights of measurements from standard deviation
        - Separating topology parameters from input features
        - Normalizing features
        
    
    Inputs
        - A_flat: tf tensor of edge parameters and measurements of shape [n_samples, n_edges * n_edge_input],
         carrying topology parameters and input features
        - B_flat: tf tensor of bus parameters and measurements of shape [n_samples, n_buses * n_bus_input],
         carrying topology parameters and input features
        - problem: Instance of problem at hands, to get specific characteristics
        - grid: specifying grid for specific characteristics
        
        
    Output
        - a_ij_flat: tf tensor of normalized edge features of shape [n_samples, n_edges * n_edge_features] 
        - b_i_flat: tf tensor of normalized bus features of shape [n_samples, n_buses * n_bus_features] 
        - A0: tf tensor of edge parameters, reshaped to [n_samples, n_edges, n_edge_parameters]
        - B0: tf tensor of bus parameters, reshaped to [n_samples, n_edges, n_bus_parameters]
    """

    # Reshaping input tensors
    n_samples = tf.shape(A_flat)[0]
    A = tf.reshape(A_flat, [n_samples, -1, problem.d_in[0] + 2 + 3])
    B = tf.reshape(B_flat, [n_samples, -1, problem.d_in[1] + 1 + 3])

    # Removing open lines
    mask = A[:, :, 12].numpy().astype(bool)
    A = tf.boolean_mask(A, mask[0], axis=1)

    # Get relevant tensor dimensions
    num_buses = tf.shape(B)[1]
    num_lines = tf.shape(A)[1]

    # Specifying boundaries for weights value to avoid high impact of outliers
    if grid == 'cigre':
        lim = [1e6, 1e6, 3e6, 3e6, 1e5, 1e6, 1e8]

    if grid == 'ober':
        lim = [1e6, 1e7, 1e5, 1e5, 1e5, 1.4e5, 1e8]

    else:
        lim = [1e6, 1e7, 1e75, 1e7, 9e5, 1e6, 1e8]

    # Calculation of weights from standard deviation and removing outliers
    cov_v_tf = tf.math.minimum((1. / B[:, :, 2:3] ** 2), lim[0])  # tf.float32, [n_samples, n_nodes, 1]
    cov_v = cov_v_tf * tf.cast(cov_v_tf < lim[0], dtype=tf.float32)

    cov_theta_tf = tf.math.minimum((1. / B[:, :, 4:5] ** 2), lim[1])  # tf.float32, [n_samples, n_nodes, 1]
    cov_theta = cov_theta_tf * tf.cast(cov_theta_tf < lim[1], dtype=tf.float32)

    cov_P_tf = tf.math.minimum((1. / B[:, :, 6:7] ** 2), lim[2])  # tf.float32, [n_samples, n_nodes, 1]
    cov_P = cov_P_tf * tf.cast(cov_P_tf < lim[2], dtype=tf.float32)

    cov_Q_tf = tf.math.minimum((1. / B[:, :, 8:9] ** 2), lim[3])  # tf.float32, [n_samples, n_nodes, 1]
    cov_Q = cov_Q_tf * tf.cast(cov_Q_tf < lim[3], dtype=tf.float32)

    cov_PL_tf = tf.math.minimum((1. / A[:, :, 7:8] ** 2), lim[4])  # tf.float32, [n_samples, n_nodes, 1]
    cov_PL = cov_PL_tf * tf.cast(cov_PL_tf < lim[4], dtype=tf.float32) * tf.cast(cov_PL_tf > 1e0, dtype=tf.float32)

    cov_QL_tf = tf.math.minimum((1. / A[:, :, 9:10] ** 2), lim[5])  # tf.float32, [n_samples, n_nodes, 1]
    cov_QL = cov_QL_tf * tf.cast(cov_QL_tf < lim[5], dtype=tf.float32) * tf.cast(cov_QL_tf > 1e0, dtype=tf.float32)

    cov_IL_tf = tf.math.minimum((1. / A[:, :, 11:12] ** 2), lim[6])
    cov_IL = cov_IL_tf * tf.cast(cov_IL_tf < lim[6], dtype=tf.float32) * tf.cast(cov_IL_tf > 1e0, dtype=tf.float32)

    # Get feature matrices and topology parameters. Lines admittance is both a parameter and a feature
    A0 = tf.concat([A[:, :, :2], A[:, :, 12:]], axis=2)
    B0 = tf.concat([B[:, :, :1], B[:, :, 9:]], axis=2)

    A_ij = tf.concat([A[:, :, 2:7], cov_PL, A[:, :, 8:9], cov_QL, A[:, :, 10:11], cov_IL], axis=2)
    B_i = tf.concat([B[:, :, 1:2], cov_v, B[:, :, 3:4], cov_theta, B[:, :, 5:6], cov_P, B[:, :, 7:8], cov_Q], axis=2)

    # Getting normalization constants
    A_mean_tf = tf.ones([n_samples, num_lines, 1]) * \
                tf.reshape(tf.constant(problem.A_mean, dtype=tf.float32), [1, 1, problem.d_in[0]])
    A_std_tf = tf.ones([n_samples, num_lines, 1]) * \
               tf.reshape(tf.constant(problem.A_std, dtype=tf.float32), [1, 1, problem.d_in[0]])
    B_mean_tf = tf.ones([n_samples, num_buses, 1]) * \
                tf.reshape(tf.constant(problem.B_mean, dtype=tf.float32), [1, 1, problem.d_in[1]])
    B_std_tf = tf.ones([n_samples, num_buses, 1]) * \
               tf.reshape(tf.constant(problem.B_std, dtype=tf.float32), [1, 1, problem.d_in[1]])

    # Normalizing inputs A and B
    a_ij = ((A_ij - A_mean_tf) / A_std_tf) * tf.cast(A_ij != 0., tf.float32)
    b_i = ((B_i - B_mean_tf) / B_std_tf) * tf.cast(B_i != 0., tf.float32)

    a_ij_flat = tf.reshape(a_ij, [n_samples, -1])
    b_i_flat = tf.reshape(b_i, [n_samples, -1])

    return a_ij_flat, b_i_flat, A0, B0


class DeepStatisticalSolver2(keras.Model):
    """
    
    Author: B. Habib on the basis of B. Donon work
    
    Defines the latest Deep Statistical Solver architecture, built for the DSSE problem
   
    """

    def __init__(self,
                 latent_dimension=10,
                 hidden_layers=3,
                 time_step_size=0.2,
                 hyper_edge_classes=2,
                 hyper_edge_port_number=[2, 1],
                 alpha=1e-3,
                 non_lin='tanh',
                 minibatch_size=10,
                 rate=0.6,
                 l2_reg=0.04,
                 name='dss',
                 directory='./',
                 model_to_restore=None,
                 default_data_directory='datasets/dsse_cigre14',
                 proxy=False,
                 problem=None):

        super(DeepStatisticalSolver2, self).__init__()

        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.time_step_size = time_step_size
        self.hyper_edge_classes = hyper_edge_classes
        self.hyper_edge_port_number = hyper_edge_port_number
        self.alpha = alpha
        self.non_lin = non_lin
        self.minibatch_size = minibatch_size
        self.directory = directory
        self.current_train_iter = 0
        self.default_data_directory = default_data_directory
        self.proxy = proxy

        self.rate = rate
        self.l2_reg = l2_reg

        self.problem = problem

        self.d_in = {}
        self.d_out = {}
        self.d_in[0] = self.problem.d_in_A
        self.d_in[1] = self.problem.d_in_B
        self.d_out[1] = self.problem.d_out
        self.d_out[0] = 0
        self.initial_U = self.problem.initial_U

        # Normalization constants
        self.B_mean = self.problem.B_mean
        self.B_std = self.problem.B_std
        self.A_mean = self.problem.A_mean
        self.A_std = self.problem.A_std

        # Build weight tensors
        self.build_weights()

    def build_weights(self):

        """
        Builds all the trainable variables
        """

        # Build weights of each trainable mapping block, and store them

        self.phi_vertice = {}
        self.phi_edge = {}
        self.phi_out = {}

        for classe in range(self.hyper_edge_classes):

            for port in range(self.hyper_edge_port_number[classe]):
                self.phi_vertice[str(classe) + str(port)] = fc_improved(
                    non_lin=self.non_lin,
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    name=self.name + '_phi_vertice_class_{}'.format(classe) + '_port_{}'.format(port),
                    rate=self.rate,
                    l2_reg=self.l2_reg,
                    input_dim=1 + (1 + self.hyper_edge_port_number[classe]) * (self.latent_dimension) + self.d_out[
                        classe] + self.d_in[classe]
                )
            self.phi_edge[str(classe)] = fc_improved(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name + '_phi_edge_class_{}'.format(classe),
                rate=self.rate,
                l2_reg=self.l2_reg,
                input_dim=1 + (1 + self.hyper_edge_port_number[classe]) * (self.latent_dimension) + self.d_out[classe] +
                          self.d_in[classe]
            )
        self.phi_out['1'] = fc_improved(
            non_lin=self.non_lin,
            latent_dimension=self.latent_dimension,
            hidden_layers=self.hidden_layers,
            name=self.name + '_phi_out_class_{}'.format(1),
            rate=self.rate,
            l2_reg=self.l2_reg,
            input_dim=1 + (1 + self.hyper_edge_port_number[1]) * (self.latent_dimension) + self.d_out[1] + self.d_in[1],
            output_dim=2
        )

    def call(self, A_flat, B_flat, A0):
        """
        Build model
        """

        self.A0 = A0

        # Reshape the inputs
        self.minibatch_size_ = tf.shape(A_flat)[0]
        self.a_ij = tf.reshape(A_flat, [self.minibatch_size_, -1, self.d_in[0]])
        self.b_i = tf.reshape(B_flat, [self.minibatch_size_, -1, self.d_in[1]])

        # Get relevant tensor dimensions
        self.minibatch_size_tf = tf.shape(self.a_ij)[0]
        self.num_buses = tf.shape(self.b_i)[1]
        self.num_lines = tf.shape(self.a_ij)[1]
        self.A_dim = tf.shape(self.a_ij)[2]
        self.B_dim = tf.shape(self.b_i)[2]

        # Extract indices from matrix A 
        self.indices_from = tf.cast(tf.ones([self.minibatch_size_tf, self.num_lines]) * self.A0[:, :, 0], tf.int32)
        self.indices_to = tf.cast(tf.ones([self.minibatch_size_tf, self.num_lines]) * self.A0[:, :, 1], tf.int32)

        # Initialize messages, predictions and losses dict
        self.H_v = {}
        self.H_e = {}
        self.U = {}

        # Get the natural offset that will be added to every output U at every node
        self.initial_U_tf = tf.ones([self.minibatch_size_tf, self.num_buses, 1]) * \
                            tf.reshape(tf.constant(self.initial_U, dtype=tf.float32), [1, 1, self.d_out[1]])

        # Initialize latent message and prediction to 0
        self.H_v = tf.zeros([self.minibatch_size_tf, self.num_buses, self.latent_dimension])
        self.H_e['0'] = tf.zeros([self.minibatch_size_tf, self.num_lines, self.latent_dimension])
        self.H_e['1'] = tf.zeros([self.minibatch_size_tf, self.num_buses, self.latent_dimension])

        # Decode the first message. Although this step useless, it is still there for compatibility issues
        self.U = self.initial_U_tf

        # Initialize time t       
        self.time = 0.

        # Iterate over given time      
        while self.time < 1.:

            self.time_edge_input = tf.ones([self.minibatch_size_tf, self.num_lines, 1]) * tf.constant(self.time,
                                                                                                      dtype=tf.float32)
            self.time_bus_input = tf.ones([self.minibatch_size_tf, self.num_buses, 1]) * tf.constant(self.time,
                                                                                                     dtype=tf.float32)

            # Gather messages from both extremities of each edges
            self.H_v_from = custom_gather(self.H_v, self.indices_from)
            self.H_v_to = custom_gather(self.H_v, self.indices_to)

            # Concatenate all the inputs of the phi neural network
            self.Phi_edge_input = tf.concat(
                [self.time_edge_input, self.H_v_from, self.H_v_to, self.H_e['0'], self.a_ij], axis=2)
            self.Phi_bus_input = tf.concat([self.time_bus_input, self.H_v, self.H_e['1'], self.U, self.b_i], axis=2)

            # Compute the phi using the dedicated neural network blocks
            self.Phi_port1 = self.phi_vertice['00'](self.Phi_edge_input)
            self.Phi_port2 = self.phi_vertice['01'](self.Phi_edge_input)

            # Get the sum of each transformed messages at each node
            self.Phi_port1_sum = custom_scatter(
                self.indices_from,
                self.Phi_port1,
                [self.minibatch_size_tf, self.num_buses, self.latent_dimension])
            self.Phi_port2_sum = custom_scatter(
                self.indices_to,
                self.Phi_port2,
                [self.minibatch_size_tf, self.num_buses, self.latent_dimension])

            # Update latent variables H_v and normalize         
            self.H_v += self.time_step_size * self.phi_vertice['10'](
                self.Phi_bus_input) + self.Phi_port1_sum + self.Phi_port2_sum
            self.H_v = tf.math.divide(self.H_v, tf.norm(self.H_v, axis=2, keepdims=True) + 1.)

            # Concatenate all the inputs of the phi neural network : Update of H_v
            self.Phi_edge_input = tf.concat(
                [self.time_edge_input, self.H_v_from, self.H_v_to, self.H_e['0'], self.a_ij], axis=2)
            self.Phi_bus_input = tf.concat([self.time_bus_input, self.H_v, self.H_e['1'], self.U, self.b_i], axis=2)

            # Update latent variables H_e and normalize           
            self.H_e['1'] += self.time_step_size * self.phi_edge['1'](self.Phi_bus_input)
            self.H_e['0'] += self.time_step_size * self.phi_edge['0'](self.Phi_edge_input)

            self.H_e['1'] = tf.math.divide(self.H_e['1'], tf.norm(self.H_e['1'], axis=2, keepdims=True) + 1.)
            self.H_e['0'] = tf.math.divide(self.H_e['0'], tf.norm(self.H_e['0'], axis=2, keepdims=True) + 1.)

            # Concatenate all the inputs of the phi neural network : Update of H_e
            self.Phi_edge_input = tf.concat(
                [self.time_edge_input, self.H_v_from, self.H_v_to, self.H_e['0'], self.a_ij], axis=2)
            self.Phi_bus_input = tf.concat([self.time_bus_input, self.H_v, self.H_e['1'], self.U, self.b_i], axis=2)

            # Update of prediction U
            self.U += self.time_step_size * self.phi_out['1'](self.Phi_bus_input)

            if self.time + self.time_step_size < 1.:
                self.time += self.time_step_size

            else:
                self.time += self.time_step_size

        # Get the final prediction      
        self.U_final = self.U

        return self.U_final


def extract_fn(tfrecord):
    """
    Helper function to map data from tfrecords file to A,B and U flat matrices and use it as iterator
    
    Input: tfrecords file containing A, B and U flat matrices
    
    Output: A,B and U flat matrices for each of the samples
       
    """

    # Extract features using the keys set during creation
    features = {
        'A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'U': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)

    return [sample['A'], sample['B'], sample['U']]


def loss(model, A_flat, B_flat, lamda, A0, B0, problem):
    """
    Calculate the loss during training
    
    """

    # get predictions from model
    y = model(A_flat, B_flat, A0, training=True)

    # get loss value per sample in batch
    cost_per_sample = problem.cost_function(A_flat, B_flat, y, lamda, A0, B0)

    # get mean loss value for the batch
    loss_object = tf.reduce_mean(cost_per_sample)

    return loss_object


def loss_sup(model, A_flat, B_flat, U_flat, A0, B0, problem):
    """
    Calculate the training loss in a supervised fashion
    
    """

    # Reshape labels
    size_mb = tf.shape(U_flat)[0]
    y_true = tf.reshape(U_flat, [size_mb, -1, 2])

    # Get prediction from model and post-process it
    pred = model(A_flat, B_flat, A0, training=True)
    y_pred = pred * problem.B_std[0:3:2] + problem.B_mean[0:3:2]
    y = tf.concat([y_pred[:, :, 0:1], y_pred[:, :, 1:] * (1. - B0[:, :, -1:])], axis=2)

    # Calculate MSE loss
    mse = tf.keras.losses.MeanSquaredError()
    loss_object = mse(y_true, y)

    return loss_object


def grad(model, A_flat, B_flat, lamda, A0, B0, problem):
    """
    Create gradient tape for gradient descent learning
    
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model, A_flat, B_flat, lamda, A0, B0, problem)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def grad_sup(model, A_flat, B_flat, U_flat, lamda, A0, B0, problem):
    """
    Create gradient tape for gradient descent learning, for supervised learning
    
    """

    with tf.GradientTape() as tape:
        loss_value = loss_sup(model, A_flat, B_flat, U_flat, A0, B0, problem)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_model(model, problem, lr, lamda, norm, num_epochs, minibatch_size, data_directory, case, optimizer, grid):
    """
    Training function
    
    """

    # Building data iterator
    train_dataset = tf.data.TFRecordDataset([os.path.join(data_directory, 'train_' + case + '.tfrecords')])
    train_dataset = train_dataset.map(extract_fn).shuffle(100).batch(minibatch_size)

    valid_dataset = tf.data.TFRecordDataset([os.path.join(data_directory, 'val_' + case + '.tfrecords')])
    valid_dataset = valid_dataset.map(extract_fn).shuffle(100).batch(minibatch_size)

    # Keep results for plotting
    train_loss_results = []

    valload = []
    valv = []
    start = time.time()

    for epoch in range(num_epochs):

        start_ep = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        step = 0

        # Training loop
        for A_flat, B_flat, U_flat in train_dataset:
            step += 1

            a_flat, b_flat, A0, B0 = preprocess_data(A_flat, B_flat, problem, grid)

            # Optimize the model
            loss_value, grads = grad(model, a_flat, b_flat, lamda, A0, B0, problem)

            # Clipping gradient
            if not norm == 0:
                grads = [tf.clip_by_norm(g, norm) for g in grads]

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

        # End epoch
        end_ep = time.time()
        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
        print("Time epoch: " + str(end_ep - start_ep))

        # Validation of the epoch

        val_lossv_results = []
        val_load_results = []
        val_loadline_results = []
        val_lossp_results = []

        step_val = 0
        std_U = 0
        std_pred = 0

        for A_flat, B_flat, U_flat in valid_dataset:
            step_val += 1
            a_flat, b_flat, A0, B0 = preprocess_data(A_flat, B_flat, problem, grid)

            pred = model(a_flat, b_flat, A0, training=False)
            y_pred = pred * problem.B_std[0:3:2] + problem.B_mean[0:3:2]
            y = tf.concat([y_pred[:, :, 0:1], y_pred[:, :, 1:] * (1. - B0[:, :, -1:])], axis=2)

            size_mb = tf.shape(U_flat)[0]
            y_true = tf.reshape(U_flat, [size_mb, -1, 2])

            # Compute other variables to assess accuracy

            pf_pred, qf_pred, pt_pred, qt_pred, if_pred, it_pred, loading_pred = get_pflow(y, A0, problem, grid)
            pf_true, qf_true, pt_true, qt_true, if_true, it_true, loading_true = get_pflow(y_true, A0, problem, grid)

            # Compute standard deviation to detect high bias
            std_U += y_true[:, :, 0].numpy().std()
            std_pred += y[:, :, 0].numpy().std()

            # Compute RMSE on loading and voltage
            ll = tf.reduce_mean((loading_pred - loading_true) ** 2) ** (1 / 2)
            lpl = tf.reduce_mean((loading_pred - loading_true) ** 2, axis=[0, 2]) ** (1 / 2)
            lv = tf.reduce_mean((y_true[:, :, 0] - y[:, :, 0]) ** 2) ** (1 / 2)
            lp = tf.reduce_mean((pf_pred - pf_true) ** 2) ** (1 / 2)

            val_load_results.append(ll)
            val_lossp_results.append(lp)
            val_loadline_results.append(lpl)
            val_lossv_results.append(lv)

        print("std U: " + str(std_U / step_val))
        print("std pred: " + str(std_pred / step_val))

        print("Val loss V: " + str(tf.math.reduce_mean(val_lossv_results).numpy()))
        print(" val RMSE loading: " + str(tf.math.reduce_mean(val_load_results).numpy()))
        print(" val RMSE loading per line: " + str(tf.math.reduce_mean(val_loadline_results, axis=0).numpy()))
        print("val RMSE pflow: " + str(tf.math.reduce_mean(val_lossp_results).numpy()))
        print("")

        valv.append(tf.math.reduce_mean(val_lossv_results))
        valload.append(tf.math.reduce_mean(val_load_results))

    print("training time :" + str(time.time() - start))

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(valload)
    ax.set_title("RMSE of loading in validation per epoch")
    ax.set_ylabel("[%]")
    ax.set_xlabel("# epoch")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(valv)
    ax.set_title("RMSE of V in validation per epoch")
    ax.set_ylabel("[-]")
    ax.set_xlabel("# epoch")

    return model


def train_model_sup(model, problem, lr, lamda, norm, num_epochs, minibatch_size, data_directory, case, optimizer, grid):
    """
   Training function for supervised learning
   
   """

    # Building data iterator
    train_dataset = tf.data.TFRecordDataset([os.path.join(data_directory, 'train_' + case + '.tfrecords')])
    train_dataset = train_dataset.map(extract_fn).shuffle(100).batch(minibatch_size)

    valid_dataset = tf.data.TFRecordDataset([os.path.join(data_directory, 'val_' + case + '.tfrecords')])
    valid_dataset = valid_dataset.map(extract_fn).shuffle(100).batch(minibatch_size)

    # Keep results for plotting
    train_loss_results = []

    valload = []
    valv = []
    start = time.time()

    for epoch in range(num_epochs):

        start_ep = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        step = 0

        # Training loop
        for A_flat, B_flat, U_flat in train_dataset:
            step += 1

            a_flat, b_flat, A0, B0 = preprocess_data(A_flat, B_flat, problem, grid)

            # Optimize the model
            loss_value, grads = grad_sup(model, a_flat, b_flat, U_flat, lamda, A0, B0, problem)

            # Clipping gradient
            if not norm == 0:
                grads = [tf.clip_by_norm(g, norm) for g in grads]

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

        # End epoch
        end_ep = time.time()
        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
        print("Time epoch: " + str(end_ep - start_ep))

        # Validation of the epoch

        val_lossv_results = []
        val_load_results = []
        val_loadline_results = []
        val_lossp_results = []

        step_val = 0
        std_U = 0
        std_pred = 0

        for A_flat, B_flat, U_flat in valid_dataset:
            step_val += 1
            a_flat, b_flat, A0, B0 = preprocess_data(A_flat, B_flat, problem, grid)

            pred = model(a_flat, b_flat, A0, training=False)
            y_pred = pred * problem.B_std[0:3:2] + problem.B_mean[0:3:2]
            y = tf.concat([y_pred[:, :, 0:1], y_pred[:, :, 1:] * (1. - B0[:, :, -1:])], axis=2)

            size_mb = tf.shape(U_flat)[0]
            y_true = tf.reshape(U_flat, [size_mb, -1, 2])

            # Compute other variables to assess accuracy

            pf_pred, qf_pred, pt_pred, qt_pred, if_pred, it_pred, loading_pred = get_pflow(y, A0, problem, grid)
            pf_true, qf_true, pt_true, qt_true, if_true, it_true, loading_true = get_pflow(y_true, A0, problem, grid)

            # Comput standard deviation to detect high bias
            std_U += y_true[:, :, 0].numpy().std()
            std_pred += y[:, :, 0].numpy().std()

            # Compute RMSE on loading and voltage
            ll = tf.reduce_mean((loading_pred - loading_true) ** 2) ** (1 / 2)
            lpl = tf.reduce_mean((loading_pred - loading_true) ** 2, axis=[0, 2]) ** (1 / 2)
            lv = tf.reduce_mean((y_true[:, :, 0] - y[:, :, 0]) ** 2) ** (1 / 2)
            lp = tf.reduce_mean((pf_pred - pf_true) ** 2) ** (1 / 2)

            val_load_results.append(ll)
            val_lossp_results.append(lp)
            val_loadline_results.append(lpl)
            val_lossv_results.append(lv)

        print("std U: " + str(std_U / step_val))
        print("std pred: " + str(std_pred / step_val))

        print("Val loss V: " + str(tf.math.reduce_mean(val_lossv_results).numpy()))
        print(" val RMSE loading: " + str(tf.math.reduce_mean(val_load_results).numpy()))
        print(" val RMSE loading per line: " + str(tf.math.reduce_mean(val_loadline_results, axis=0).numpy()))
        print("val RMSE pflow: " + str(tf.math.reduce_mean(val_lossp_results).numpy()))
        print("")

        valv.append(tf.math.reduce_mean(val_lossv_results))
        valload.append(tf.math.reduce_mean(val_load_results))

    print("training time :" + str(time.time() - start))

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(valload)
    ax.set_title("RMSE of loading in validation per epoch")
    ax.set_ylabel("[-]")
    ax.set_xlabel("# epoch")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(valv)
    ax.set_title("RMSE of V in validation per epoch")
    ax.set_ylabel("[-]")
    ax.set_xlabel("# epoch")

    return model


def test_model(model, problem, minibatch_size, data_directory, case, grid):
    # Building iterator for test data
    test_dataset = tf.data.TFRecordDataset([os.path.join(data_directory, 'test_' + case + '.tfrecords')])
    test_dataset = test_dataset.map(extract_fn).shuffle(100).batch(minibatch_size)

    test_loss_results = []
    test_lossv_results = []

    for A_flat, B_flat, U_flat in test_dataset:
        a_flat, b_flat, A0, B0 = preprocess_data(A_flat, B_flat, problem, grid)

        pred = model(a_flat, b_flat, A0, training=False)
        y_pred = pred * problem.B_std[0:3:2] + problem.B_mean[0:3:2]
        y = tf.concat([y_pred[:, :, 0:1], y_pred[:, :, 1:] * (1. - B0[:, :, -1:])], axis=2)

        size_mb = tf.shape(U_flat)[0]
        y_true = tf.reshape(U_flat, [size_mb, -1, 2])

        pf_pred, qf_pred, pt_pred, qt_pred, if_pred, it_pred, loading_pred = get_pflow(y, A0, problem, grid)
        pf_true, qf_true, pt_true, qt_true, if_true, it_true, loading_true = get_pflow(y_true, A0, problem, grid)

        ll = tf.reduce_mean((loading_pred - loading_true) ** 2) ** (1 / 2)
        lv = tf.reduce_mean((y_true[:, :, 0] - y[:, :, 0]) ** 2) ** (1 / 2)

        test_loss_results.append(ll)
        test_lossv_results.append(lv)

    print("RMSE loading for test set: " + str(tf.math.reduce_mean(test_loss_results)))
    print("RMSE V for test set: " + str(tf.math.reduce_mean(test_lossv_results)))


def get_pflow(y, A0, problem, grid):
    """
    
   Power flow equations to compute other estimated variables from the outputs of the model

   """

    # Store trafo numbers for separation later and set grid parameters
    if grid == 'cigre':
        n_trafo = 2
        ratio = 5.5
        V_n = 20.
        V_hv = 110.
    if (grid == 'ober' or grid == 'ober2'):
        n_trafo = 1
        ratio = 5.5
        V_n = 20.
        V_hv = 110.
    if grid == 'lv':
        n_trafo = 1
        V_n = 0.416
        V_hv = 11.
        ratio = V_hv/V_n



    # Store output values separately
    U1 = y[:, :, 0:1]
    U2 = y[:, :, 1:]

    # Get topology fro A0
    indices_from = tf.cast(A0[:, :, 0], tf.int32)  # tf.int32, [n_samples, n_edges, 1]
    indices_to = tf.cast(A0[:, :, 1], tf.int32)

    # Extact edge characteristics from A matrix
    Y1_ij = A0[:, :, 2:3]  # tf.float32, [n_samples, n_edges, 1] = Re(Y)
    Y2_ij = A0[:, :, 3:4]  # tf.float32, [n_samples, n_edges, 1] = Im(Y)

    Ys1_ij = A0[:, :, 4:5]  # tf.float32, [n_samples, n_edges, 1] = Re(Ys)
    Ys2_ij = A0[:, :, 5:6]  # tf.float32, [n_samples, n_edges, 1] = Im(Ys)

    # Gather V and theta on both sides of each edge
    U1_i = custom_gather(U1, indices_from)  # tf.float32, [n_samples , n_edges, 1], in p.u.
    U2_i = custom_gather(U2, indices_from)  # tf.float32, [n_samples , n_edges, 1], in p.u.
    U1_j = custom_gather(U1, indices_to)  # tf.float32, [n_samples , n_edges, 1], in rad
    U2_j = custom_gather(U2, indices_to)  # tf.float32, [n_samples , n_edges, 1], in ra

    # Compute h(U) = V_i, theta_i, P_i, Q_i, P_ij, Q_ij, I_ij
    P_ij_from = - U1_i * U1_j * (
            Y1_ij * tf.math.cos(U2_i - U2_j - A0[:, :, -1:]) + Y2_ij * tf.math.sin(U2_i - U2_j - A0[:, :, -1:])) \
                + (Y1_ij + Ys1_ij / 2) * U1_i ** 2  # per unit

    Q_ij_from = U1_i * U1_j * (
            - Y1_ij * tf.math.sin(U2_i - U2_j - A0[:, :, -1:]) + Y2_ij * tf.math.cos(U2_i - U2_j - A0[:, :, -1:])) \
                - (Y2_ij + Ys2_ij / 2) * U1_i ** 2  # per unit

    P_ij_to = - U1_i * U1_j * (
            Y1_ij * tf.math.cos(U2_i - U2_j - A0[:, :, -1:]) - Y2_ij * tf.math.sin(U2_i - U2_j - A0[:, :, -1:])) \
              + (Y1_ij + Ys1_ij / 2) * U1_j ** 2  # per unit

    Q_ij_to = U1_i * U1_j * (
            Y1_ij * tf.math.sin(U2_i - U2_j - A0[:, :, -1:]) + Y2_ij * tf.math.cos(U2_i - U2_j - A0[:, :, -1:])) \
              - (Y2_ij + Ys2_ij / 2) * U1_j ** 2  # per unit

    P_ij_from_comp = tf.cast(P_ij_from, dtype=tf.complex64)
    Q_ij_from_comp = tf.cast(Q_ij_from, dtype=tf.complex64)
    U1_i_comp = tf.cast(U1_i, dtype=tf.complex64)

    imag = tf.complex(tf.constant(0.), tf.constant(1.))
    I_ij_from = tf.math.abs((P_ij_from_comp - imag * Q_ij_from_comp) / (U1_i_comp * np.sqrt(3)))

    if n_trafo>0:
        I_ij_from = tf.concat([tf.cast(I_ij_from[:, :-n_trafo], dtype=tf.float32),
                               tf.cast(I_ij_from[:, -n_trafo:] / ratio, dtype=tf.float32)], axis=1)

    else:
        I_ij_from = tf.cast(I_ij_from, dtype=tf.float32)

    P_ij_to_comp = tf.cast(P_ij_to, dtype=tf.complex64)
    Q_ij_to_comp = tf.cast(Q_ij_to, dtype=tf.complex64)
    U1_j_comp = tf.cast(U1_j, dtype=tf.complex64)

    imag = tf.complex(tf.constant(0.), tf.constant(1.))
    I_ij_to = tf.math.abs((P_ij_to_comp - imag * Q_ij_to_comp) / (U1_j_comp * np.sqrt(3)))
    I_ij_to = tf.cast(I_ij_to, dtype=tf.float32)

    # Calculating line and trafo loading

    if n_trafo>0:
        i_ka = tf.maximum(I_ij_from, I_ij_to)[:, :-n_trafo, :] * V_n
    else:
        i_ka = tf.maximum(I_ij_from, I_ij_to) * V_n

    if n_trafo>0:
        i_kat = tf.maximum(I_ij_from[:, -n_trafo:, :] * V_hv / 25, I_ij_to[:, -n_trafo:, :] * V_n / 25) * V_n * 100

        i_max = tf.ones([tf.shape(A0)[0], 1, 1]) * \
                tf.reshape(tf.constant(problem.i_max, dtype=tf.float32), [1, tf.shape(A0)[1] - n_trafo, 1])

        loading = tf.concat([i_ka * 100 / i_max, i_kat], axis=1)

    else:
        i_max = tf.ones([tf.shape(A0)[0], 1, 1]) * \
                tf.reshape(tf.constant(problem.i_max, dtype=tf.float32), [1, tf.shape(A0)[1], 1])

        loading = i_ka * 100 / i_max

    return P_ij_from * V_n ** 2, Q_ij_from * V_n ** 2, P_ij_to * V_n ** 2, Q_ij_to * V_n ** 2, I_ij_from * V_n, I_ij_to * V_n, loading
