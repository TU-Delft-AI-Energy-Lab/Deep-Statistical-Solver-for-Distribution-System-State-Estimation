import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
import pickle
from torch_geometric.utils import scatter, get_laplacian

def get_bus_param(net):
    df_bus_param = pd.DataFrame(index=net.bus.index)
    df_bus_param["vn_kv"] = net.bus["vn_kv"]
    df_bus_param["bool_slack"] = (df_bus_param["vn_kv"]==df_bus_param["vn_kv"].max()).astype(float)
    df_bus_param["bool_zero_inj"] = 0.
    for i in net.bus.index:
        if (not i in net.load["bus"].values):
            if df_bus_param["bool_slack"][i]==0.:
                df_bus_param["bool_zero_inj"][i]=1.
                

    return df_bus_param

def get_edge_param(net):
    
    net.bus["name"] = np.arange(net.bus.index.size)
    
    edge_length = net.line["length_km"]
    edge_r = net.line["r_ohm_per_km"] * edge_length
    edge_x = net.line["x_ohm_per_km"] * edge_length
    
    edge_c = net.line["c_nf_per_km"] * edge_length
    edge_b =  -2 * np.pi * net.f_hz * edge_c * 1e-9
    edge_g = net.line["g_us_per_km"] * edge_length * 1e-6
    
    edge_imax = net.line["max_i_ka"]
    t_sn = net.trafo["sn_mva"]
    
    t_r = (net.trafo["vkr_percent"]/100) * (net.sn_mva/t_sn)
    t_z = (net.trafo["vk_percent"]/100) * (net.sn_mva/t_sn)
    t_x_square =  t_z.pow(2) - t_r.pow(2) 
    t_x = t_x_square.pow(0.5)
    
    t_g = (net.trafo["pfe_kw"]/1000) * (net.sn_mva/t_sn**2)
    t_y = (net.trafo["i0_percent"]/100)
    t_b_square = t_y**2 - t_g**2
    t_b = t_b_square.pow(0.5)
    
    Z_trafo = (net.trafo["vn_lv_kv"]**2*net.sn_mva)
    
    
    
    t_R = t_r * Z_trafo
    t_X = t_x * Z_trafo
    t_G = t_g/Z_trafo
    t_B = t_b/Z_trafo
    
    edge_r = pd.concat([edge_r,t_R])
    edge_x = pd.concat([edge_x,t_X])
    edge_b = pd.concat([edge_b,t_B])
    edge_g = pd.concat([edge_g,t_G])
    
    t_phase_shift = net.trafo["shift_degree"]*np.pi/180
    
    edge_phase_shift = np.concatenate((np.zeros(net.line.index.size), t_phase_shift.values))
    
    edge_switch =  pd.DataFrame(data ={"closed": True}, index = np.concatenate([net.line.index,net.trafo.index]))
    edge_ind = -1
    for i in net.switch.index:
        old_ind = edge_ind
        edge_ind = net.switch["element"][i]

        if edge_ind == old_ind:
            edge_switch.loc[edge_ind] = (net.switch["closed"][i] and net.switch["closed"][i-1])
        else:
            edge_switch.loc[edge_ind] = net.switch["closed"][i]
            
    bool_closed_line = edge_switch["closed"].values.astype("float64")

    edge_source = pd.concat([net.line["from_bus"],net.trafo["hv_bus"]])
    edge_target = pd.concat([net.line["to_bus"],net.trafo["lv_bus"]])
    edge_source_index =  edge_source.values
    edge_target_index =  edge_target.values
    new_edge_source = net.bus['name'][edge_source_index].values.astype(float)
    new_edge_target = net.bus['name'][edge_target_index].values.astype(float)
    
    
    edge_Z = edge_r.values + 1j * edge_x.values
    edge_Y = np.reciprocal(edge_Z)
    edge_Ys = edge_g - 1j* edge_b
    
    edge_imax_or_sn = pd.concat([edge_imax, t_sn])
    
    
    edge_param = pd.DataFrame({'from_bus': new_edge_source, 'to_bus': new_edge_target, 'G': np.real(edge_Y), 'B': np.imag(edge_Y), 'Gs': np.nan_to_num(np.real(edge_Ys)), 'Bs': np.nan_to_num(np.imag(edge_Ys)), 'closed line': bool_closed_line, 'phase shift': edge_phase_shift, 'imax or sn': edge_imax_or_sn})
    
    return edge_param

def data_from_pickles(folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pflow):
    
    
    with open(folder+'nodes', 'rb') as file:
        df_nodes_list = pickle.load(file)
    with open(folder+'edges', 'rb') as file:
        df_edges_list = pickle.load(file)
    with open(folder+'labels', 'rb') as file:
        df_labels_list = pickle.load(file)
    with open(folder+'noise_param', 'rb') as file:
        df_noise_list = pickle.load(file)
        
    data_list = []
    nodes_noises = df_noise_list[['v_noise', 'v_noise', 'pm_noise', 'pm_noise']].values
    zero_inj_noises = df_noise_list[['zero_inj_coef', 'zero_inj_coef']].values
    slack_noise = df_noise_list[['v_noise', 'zero_inj_coef', 'p_noise', 'p_noise']].values
    pflow_noises = df_noise_list[['p_noise','p_noise']].values
    
    x_tensor = torch.tensor([])
    y_tensor = torch.tensor([])
    edge_attr_tensor = torch.tensor([])
    edge_index_list = []
    
    for i in range(len(df_nodes_list)):
        
        num_nodes = df_nodes_list[i].shape[0]
        meas_bus_mask = np.ones([num_nodes,num_nmeas]) * [0,0,1,1]
            
        for j in meas_v:
            meas_bus_mask[j][0] = 1.

        # Extract node features from the DataFrame
        x_mean = np.multiply(df_nodes_list[i][['vm_pu','va_rad','p_mw', 'q_mvar']].values, meas_bus_mask)
        x_std = x_mean * (slack_noise * np.expand_dims(df_nodes_list[i]['bool_slack'].values,axis=1) +  nodes_noises * (1 - np.expand_dims(df_nodes_list[i]['bool_slack'].values,axis=1)))
        
        x = torch.tensor(x_mean + np.random.normal(loc=0., scale = np.abs(x_std)), dtype=torch.float32)
        
        x_std[:,2:] += zero_inj_noises * np.expand_dims(df_nodes_list[i]['bool_zero_inj'].values, axis=1)
        
        x_std[:,1:2] += slack_noise[:,1:2] * np.expand_dims(df_nodes_list[i]['bool_slack'].values, axis=1)
        
        x_cov = torch.tensor(1, dtype=torch.float32)/torch.maximum((torch.abs(torch.tensor(x_std, dtype=torch.float32))),torch.tensor(1e-6, dtype=torch.float32))**2
        x_cov *= (x_cov < 1e12).type(torch.float32)


        x = torch.concat([x[:,0:1], x_cov[:,0:1], x[:,1:2], x_cov[:,1:2], x[:,2:3],x_cov[:,2:3], x[:,3:], x_cov[:,3:]], axis=1)

        # Extract edge index from the DataFrame
        df_closed_edges = df_edges_list[i][df_edges_list[i]['closed line']==1.]
        
        num_lines = df_closed_edges.shape[0]
        
        meas_pflow_mask = np.zeros([num_lines,num_emeas])
        
        for j in meas_pflow:
            meas_pflow_mask[j] = np.ones([1, num_emeas])
        
        edge_index = torch.tensor(df_closed_edges[['from_bus', 'to_bus']].values.astype(int), dtype=torch.long).t().contiguous()
        
        # Extract edge features from the DataFrame
        edge_attr_mean = np.multiply(df_closed_edges[['p_from_mw', 'q_from_mvar']].values, meas_pflow_mask)
        edge_attr_std = edge_attr_mean * pflow_noises

        edge_attr = torch.tensor(edge_attr_mean+ np.random.normal(loc=0., scale = np.abs(edge_attr_std)), dtype=torch.float32)
        
        edge_attr_cov = torch.tensor(1, dtype=torch.float32)/torch.maximum((torch.abs(torch.tensor(edge_attr_std, dtype=torch.float32))),torch.tensor(1e-5, dtype=torch.float32))**2
        edge_attr_cov *= (edge_attr_cov < 1e10).type(torch.float32)
        
        edge_attr_imp=torch.tensor(df_closed_edges[['G', 'B']].values, dtype=torch.float32)


        edge_attr = torch.concat([edge_attr[:,0:1], edge_attr_cov[:,0:1], edge_attr[:,1:], edge_attr_cov[:,1:],edge_attr_imp], axis=1)

        y = torch.tensor(df_labels_list[i].values, dtype=torch.float32)
        
        nodes_param = torch.tensor(df_nodes_list[i][['vn_kv', 'bool_slack', 'bool_zero_inj']].values, dtype=torch.float32)
        edges_param = torch.tensor(df_closed_edges[['G','B','Gs', 'Bs', 'closed line', 'phase shift', 'imax or sn']].values, dtype=torch.float32)
        
        x_tensor = torch.cat((x_tensor,(torch.concat([x,nodes_param],axis=1))), dim=0)
        y_tensor = torch.cat((y_tensor,y), dim=0)
        edge_attr_tensor = torch.cat((edge_attr_tensor,torch.concat([edge_attr, edges_param], axis=1)), dim=0)
        edge_index_list.append(edge_index)
        
    x_mask = x_tensor !=0.
    x_set_mean = torch.nan_to_num((x_tensor*x_mask).sum(dim=[0])/x_mask.sum(dim=[0]))
    x_set_std = torch.nan_to_num(torch.sqrt(((x_tensor-x_set_mean)**2*x_mask).sum(dim=[0])/x_mask.sum(dim=[0])))
    x_set = torch.nan_to_num((x_tensor-x_set_mean)*x_mask/x_set_std)
    x_set[:,num_nfeat:] = x_tensor[:,num_nfeat:]
    
    edge_attr_mask = edge_attr_tensor != 0.
    edge_attr_set_mean = torch.nan_to_num((edge_attr_tensor*edge_attr_mask).sum(dim=[0])/edge_attr_mask.sum(dim=[0]))
    edge_attr_set_std = torch.nan_to_num(torch.sqrt(((edge_attr_tensor-edge_attr_set_mean)**2*edge_attr_mask).sum(dim=[0])/edge_attr_mask.sum(dim=[0])))
    edge_attr_set = torch.nan_to_num((edge_attr_tensor-edge_attr_set_mean)*edge_attr_mask/edge_attr_set_std)
    
    edge_attr_set[:,num_efeat:] = edge_attr_tensor[:,num_efeat:]
    
    for i in range(len(df_nodes_list)):
        num_nodes = df_nodes_list[i].shape[0]
        df_closed_edges = df_edges_list[i][df_edges_list[i]['closed line']==1.]
        
        num_lines = df_closed_edges.shape[0]
        # Create a PyTorch Geometric Data object
        data = Data(x=x_set[:num_nodes], edge_index=edge_index_list[i], edge_attr=edge_attr_set[:num_lines], y=y_tensor[:num_nodes])
        data.validate(raise_on_error=True)
        data_list.append(data)
        
        x_set = x_set[num_nodes:]
        y_tensor = y_tensor[num_nodes:]
        edge_attr_set = edge_attr_set[num_lines:]
        
    return data_list, x_set_mean[:num_nfeat], x_set_std[:num_nfeat], edge_attr_set_mean[:num_efeat], edge_attr_set_std[:num_efeat]

    
    
    with open(folder+'nodes', 'rb') as file:
        df_nodes_list = pickle.load(file)
    with open(folder+'edges', 'rb') as file:
        df_edges_list = pickle.load(file)
    with open(folder+'labels', 'rb') as file:
        df_labels_list = pickle.load(file)
    with open(folder+'noise_param', 'rb') as file:
        df_noise_list = pickle.load(file)
        
    data_list = []
    nodes_noises = df_noise_list[['v_noise', 'v_noise', 'pm_noise', 'pm_noise']].values
    zero_inj_noises = df_noise_list[['zero_inj_coef', 'zero_inj_coef']].values
    slack_noise = df_noise_list[['v_noise', 'zero_inj_coef', 'p_noise', 'p_noise']].values
    pflow_noises = df_noise_list[['p_noise','p_noise']].values
    
    x_tensor = torch.tensor([])
    y_tensor = torch.tensor([])
    edge_attr_tensor = torch.tensor([])
    edge_index_list = []
    
    for i in range(len(df_nodes_list)):
        
        num_nodes = df_nodes_list[i].shape[0]
        meas_bus_mask = np.ones([num_nodes,num_nmeas]) * [0,0,1,1]
    

        meas_v = df_nodes_list[i].index
        
        
        print(meas_v)
            
        for j in meas_v:
            meas_bus_mask[j][0] = 1.

        # Extract node features from the DataFrame
        x_mean = np.multiply(df_nodes_list[i][['vm_pu','va_rad','p_mw', 'q_mvar']].values, meas_bus_mask)
        x_std = x_mean * (slack_noise * np.expand_dims(df_nodes_list[i]['bool_slack'].values,axis=1) +  nodes_noises * (1 - np.expand_dims(df_nodes_list[i]['bool_slack'].values,axis=1)))
        
        x = torch.tensor(x_mean + np.random.normal(loc=0., scale = np.abs(x_std)), dtype=torch.float32)
        
        x_std[:,2:] += zero_inj_noises * np.expand_dims(df_nodes_list[i]['bool_zero_inj'].values, axis=1)
        
        x_std[:,1:2] += slack_noise[:,1:2] * np.expand_dims(df_nodes_list[i]['bool_slack'].values, axis=1)
        
        x_cov = torch.tensor(1, dtype=torch.float32)/torch.maximum((torch.abs(torch.tensor(x_std, dtype=torch.float32))),torch.tensor(1e-6, dtype=torch.float32))**2
        x_cov *= (x_cov < 1e12).type(torch.float32)


        x = torch.concat([x[:,0:1], x_cov[:,0:1], x[:,1:2], x_cov[:,1:2], x[:,2:3],x_cov[:,2:3], x[:,3:], x_cov[:,3:]], axis=1)

        # Extract edge index from the DataFrame
        df_closed_edges = df_edges_list[i][df_edges_list[i]['closed line']==1.]
        
        num_lines = df_closed_edges.shape[0]
        meas_pflow = df_closed_edges.index
        
        print(meas_pflow)
        
        meas_pflow_mask = np.zeros([num_lines,num_emeas])
        
        for j in range(meas_pflow.shape[0]):
            meas_pflow_mask[j] = np.ones([1, num_emeas])
        
        edge_index = torch.tensor(df_closed_edges[['from_bus', 'to_bus']].values.astype(int), dtype=torch.long).t().contiguous()
        
        # Extract edge features from the DataFrame
        edge_attr_mean = np.multiply(df_closed_edges[['p_from_mw', 'q_from_mvar']].values, meas_pflow_mask)
        edge_attr_std = edge_attr_mean * pflow_noises

        edge_attr = torch.tensor(edge_attr_mean+ np.random.normal(loc=0., scale = np.abs(edge_attr_std)), dtype=torch.float32)
        
        edge_attr_cov = torch.tensor(1, dtype=torch.float32)/torch.maximum((torch.abs(torch.tensor(edge_attr_std, dtype=torch.float32))),torch.tensor(1e-5, dtype=torch.float32))**2
        edge_attr_cov *= (edge_attr_cov < 1e10).type(torch.float32)
        
        edge_attr_imp=torch.tensor(df_closed_edges[['G', 'B']].values, dtype=torch.float32)


        edge_attr = torch.concat([edge_attr[:,0:1], edge_attr_cov[:,0:1], edge_attr[:,1:], edge_attr_cov[:,1:],edge_attr_imp], axis=1)

        y = torch.tensor(df_labels_list[i].values, dtype=torch.float32)
        
        nodes_param = torch.tensor(df_nodes_list[i][['vn_kv', 'bool_slack', 'bool_zero_inj']].values, dtype=torch.float32)
        edges_param = torch.tensor(df_closed_edges[['G','B','Gs', 'Bs', 'closed line', 'phase shift', 'imax or sn']].values, dtype=torch.float32)
        
        x_tensor = torch.cat((x_tensor,(torch.concat([x,nodes_param],axis=1))), dim=0)
        y_tensor = torch.cat((y_tensor,y), dim=0)
        edge_attr_tensor = torch.cat((edge_attr_tensor,torch.concat([edge_attr, edges_param], axis=1)), dim=0)
        edge_index_list.append(edge_index)
        
    x_mask = x_tensor !=0.
    x_set_mean = torch.nan_to_num((x_tensor*x_mask).sum(dim=[0])/x_mask.sum(dim=[0]))
    x_set_std = torch.nan_to_num(torch.sqrt(((x_tensor-x_set_mean)**2*x_mask).sum(dim=[0])/x_mask.sum(dim=[0])))
    x_set = torch.nan_to_num((x_tensor-x_set_mean)*x_mask/x_set_std)
    x_set[:,num_nfeat:] = x_tensor[:,num_nfeat:]
    
    edge_attr_mask = edge_attr_tensor != 0.
    edge_attr_set_mean = torch.nan_to_num((edge_attr_tensor*edge_attr_mask).sum(dim=[0])/edge_attr_mask.sum(dim=[0]))
    edge_attr_set_std = torch.nan_to_num(torch.sqrt(((edge_attr_tensor-edge_attr_set_mean)**2*edge_attr_mask).sum(dim=[0])/edge_attr_mask.sum(dim=[0])))
    edge_attr_set = torch.nan_to_num((edge_attr_tensor-edge_attr_set_mean)*edge_attr_mask/edge_attr_set_std)
    
    edge_attr_set[:,num_efeat:] = edge_attr_tensor[:,num_efeat:]
    
    for i in range(len(df_nodes_list)):
        num_nodes = df_nodes_list[i].shape[0]
        df_closed_edges = df_edges_list[i][df_edges_list[i]['closed line']==1.]
        
        num_lines = df_closed_edges.shape[0]
        # Create a PyTorch Geometric Data object
        data = Data(x=x_set[:num_nodes], edge_index=edge_index_list[i], edge_attr=edge_attr_set[:num_lines], y=y_tensor[:num_nodes])
        data.validate(raise_on_error=True)
        data_list.append(data)
        
        x_set = x_set[num_nodes:]
        y_tensor = y_tensor[num_nodes:]
        edge_attr_set = edge_attr_set[num_lines:]
        
    return data_list, x_set_mean[:num_nfeat], x_set_std[:num_nfeat], edge_attr_set_mean[:num_efeat], edge_attr_set_std[:num_efeat]

def get_pflow(y, edge_index, node_param, edge_param, phase_shift=True):
    """
    
    Power flow equations to compute other estimated variables from the outputs of the model

    """
    V_n = node_param[:,0]
    V_hv = V_n.max()
    V_lv = V_n.min()  # Considering only two V level
    
    ratio = (V_hv / V_lv).clone().detach()

    # Store output values separately
    V = y[:,0] # in pu
    Theta = y[:,1] # in rad

    indices_from = edge_index[0]
    indices_to = edge_index[1]

    # Extact edge characteristics from A matrix
    Y1_ij = edge_param[:,0]
    Y2_ij = edge_param[:,1]

    Ys1_ij = edge_param[:,2]
    Ys2_ij = edge_param[:,3]

    # Gather V and theta on both sides of each edge
    V_i = torch.gather(V,0, indices_from)  # torch.float32, [n_samples*n_edges, 1], in p.u.
    Th_i = torch.gather(Theta,0, indices_from)  # torch.float32, [n_samples*n_edges, 1], in p.u.
    V_j = torch.gather(V,0, indices_to)  # torch.float32, [n_samples*n_edges, 1], in rad
    Th_j = torch.gather(Theta,0, indices_to)  # torch.float32, [n_samples*n_edges, 1], in ra

    # Compute h(U) = V_i, theta_i, P_i, Q_i, P_ij, Q_ij, I_ij
    
    if phase_shift:
        shift = 0
    else:  
        shift = torch.tensor(edge_param[:,5])

    trafo_pos = torch.ceil(edge_param[:, 5].clone().detach())
    imax_or_sn = edge_param[:, 6].clone().detach()

    P_ij_from = (- V_i * V_j * (Y1_ij * torch.cos(Th_i - Th_j - shift) + Y2_ij * torch.sin(Th_i - Th_j - shift)) + (Y1_ij + Ys1_ij / 2) * V_i ** 2) * V_lv**2  

    Q_ij_from = (V_i * V_j * (- Y1_ij * torch.sin(Th_i - Th_j - shift) + Y2_ij * torch.cos(Th_i - Th_j - shift)) - (Y2_ij + Ys2_ij / 2) * V_i ** 2) * V_lv**2  

    P_ij_to = (- V_i * V_j * ( Y1_ij * torch.cos(Th_i - Th_j - shift) - Y2_ij * torch.sin(Th_i - Th_j - shift)) + (Y1_ij + Ys1_ij / 2) * V_j ** 2) * V_lv**2  

    Q_ij_to = (V_i * V_j * (Y1_ij * torch.sin(Th_i - Th_j - shift) + Y2_ij * torch.cos(Th_i - Th_j - shift)) - (Y2_ij + Ys2_ij / 2) * V_j ** 2) * V_lv**2 

    I_ij_from = (torch.complex(P_ij_from, -Q_ij_from).abs() / (V_i * V_lv * torch.sqrt(torch.tensor(3))))

    I_ij_from = I_ij_from/(1.- (trafo_pos*(1.-ratio)))


    I_ij_to = torch.complex(P_ij_to, -Q_ij_to).abs() / (V_j * V_lv * torch.sqrt(torch.tensor(3)))

    # Calculating line and trafo loading

    loading_lines = ((1.- trafo_pos) * torch.maximum(I_ij_from, I_ij_to)) / imax_or_sn
    loading_trafo = (trafo_pos * torch.maximum(I_ij_from * V_hv, I_ij_to * V_lv))/ imax_or_sn

    return loading_lines, loading_trafo, P_ij_from, Q_ij_from, P_ij_to, Q_ij_to, I_ij_from, I_ij_to  # loading in %, P in MW, Q in MVAr, I in kA


def gsp_wls_edge(input, edge_input, output, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, num_samples, node_param, edge_param):
    
    total_nodes =input.shape[0]
    
    z = input[:,::2] # V, theta, P, Q (buses) [batch*num_nodes ,4]
    edge_z = edge_input[:,:4:2] # Pflow, Qflow (lines) [batch*num_lines, 2]
    z_mask = z != 0.
    edge_z_mask = edge_z != 0.
    edge_Z = (edge_z*edge_std[:4:2] + edge_mean[:4:2]) * edge_z_mask
    Z = (z*x_std[::2] + x_mean[::2]) * z_mask
    r_inv = input[:,1::2]  # Cov(V)^-1, Cov(theta)^-1, Cov(P)^-1, Cov(Q)^-1 (buses) [batch*num_nodes,4]
    r_mask = r_inv!=0.
    r_edge_inv = edge_input[:,1:4:2]
    r_edge_mask = r_edge_inv !=0.
    
    R_inv = (r_inv*x_std[1::2] + x_mean[1::2]) * r_mask
    R_edge_inv = (r_edge_inv*edge_std[1:4:2] + edge_mean[1:4:2]) * r_edge_mask
    
    v_i = output[:,0:1]*x_std[:1] + x_mean[:1] # [batch*num_nodes, 1]
    theta_i = output[:,1:] # [batch*num_nodes, 1]
    theta_i *= (1.- node_param[:,1:2]) # Enforce theta_slack = 0.
    
    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param)
    
    loading = loading_lines + loading_trafos
    
    indices_from = edge_index[0] # [batch*num_edges,1]
    indices_to = edge_index[1] # [batch*num_edges,1]
    
    L = get_laplacian(edge_index=edge_index) # [batch*num_nodes, batch*num_nodes]
    Ld = torch.sparse_coo_tensor(L[0],L[1]).to_dense()
    
    
    #Summing flow to balance in buses, negative signs in sum to follow conventions from PandaPower

    p_i = -scatter(p_to,indices_to,dim_size=total_nodes) - scatter(p_from, indices_from, dim_size=total_nodes) # [batch*num_nodes, 1]
    q_i = -scatter(q_to,indices_to, dim_size=total_nodes) - scatter(q_from, indices_from, dim_size=total_nodes) # [batch*num_nodes, 1]
    
    theta_ij = torch.abs(torch.gather(theta_i[:,0],0, indices_from) - torch.gather(theta_i[:,0],0,indices_to))

    h = torch.concatenate([v_i, theta_i, torch.unsqueeze(p_i,1), torch.unsqueeze(q_i,1)], dim = 1) # [batch*num_nodes, 4]
    
    h_edge = torch.concatenate([ torch.unsqueeze(p_from,1),  torch.unsqueeze(q_from,1)], dim =1)

    delta = Z - h  # [batch*num_nodes, 4]
    
    delta_edge = edge_Z - h_edge
    
    
    
    meas_node_weights = torch.tensor([reg_coefs['lam_v'], reg_coefs['lam_v'], reg_coefs['lam_p'], reg_coefs['lam_p']])
    meas_edge_weights = torch.tensor([reg_coefs['lam_pf'], reg_coefs['lam_pf']])
    
    J_sample = torch.sum(torch.mul(delta**2 * R_inv, meas_node_weights), axis=1)
    J_sample_edge = torch.sum(torch.mul(delta_edge**2 * R_edge_inv, meas_edge_weights), axis=1)
    
    
    J = torch.mean(J_sample) + torch.mean(J_sample_edge) # [1,1]
    

    J_v = reg_coefs['lam_reg']*torch.mean(torch.relu(v_i - 1.1) + torch.relu(0.9 - v_i))**2
    J_theta = reg_coefs['lam_reg'] * torch.mean(torch.relu(theta_ij - 0.5))**2
    J_loading = reg_coefs['lam_reg'] *torch.mean(torch.relu(loading - 1.5))**2
    
    J_reg = J +  J_v +  J_theta +  J_loading # [1,1]
    
    return J_reg
    
    
def gsp_wls(input, output, x_mean, x_std, edge_index, reg_coefs, grid, num_samples):
    
    if grid == 'cigre14':
        V_n = 20.
        V_hv = 110.
    if (grid == 'ober' or grid == 'ober2'):
        V_n = 20.
        V_hv = 110.
    if grid == 'lv':
        V_n = 0.416
        V_hv = 11.
    
    z = input[:,::2] # V, theta, P, Q (buses) [batch*num_nodes ,4]
    Z = z*x_std[::2] + x_mean[::2]
    r_inv = input[:,1::2]  # Cov(V)^-1, Cov(theta)^-1, Cov(P)^-1, Cov(Q)^-1 (buses) [batch*num_nodes,4]
    R_inv = r_inv*x_std[1::2] + x_mean[1::2]
    
    num_nodes = (z.shape[0]/num_samples).type(torch.int32)
    
    v_i = output[:,0:1]*x_std[:1] + x_mean[:1] # [batch*num_nodes, 1]
    theta_i = output[:,1:] # [batch*num_nodes, 1]
    
    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(torch.concat([v_i, theta_i], axis=1), grid,edge_index, num_samples)
    
    loading = loading_lines + loading_trafos
    
    indices_from = edge_index[0] # [batch*num_edges,1]
    indices_to = edge_index[1] # [batch*num_edges,1]
    
    L = get_laplacian(edge_index=edge_index) # [batch*num_nodes, batch*num_nodes]
    Ld = torch.sparse_coo_tensor(L[0],L[1]).to_dense()
    
    
    #Summing flow to balance in buses, negative signs in sum to follow conventions from PandaPower
    
    p_i = -scatter(p_to,indices_to) - scatter(p_from, indices_from) # [batch*num_nodes, 1]
    q_i = -scatter(q_to,indices_to) - scatter(q_from, indices_from) # [batch*num_nodes, 1]
    
    theta_ij = torch.abs(torch.gather(theta_i[:,0],0, indices_from) - torch.gather(theta_i[:,0],0,indices_to))
    

    h = torch.concatenate([v_i, theta_i, torch.unsqueeze(p_i,1), torch.unsqueeze(q_i,1)], dim = 1) # [batch*num_nodes, 4]

    delta = Z - h  # [batch*num_nodes, 4]
    
    
    
    J_sample = torch.sum(delta**2 * R_inv, axis=1)
    
    J = torch.mean(J_sample) # [1,1]


    
    
    J_v = torch.mean(torch.relu(v_i/V_n - 1.1) + torch.relu(0.9 - v_i/V_n))**2
    J_theta = torch.mean(torch.relu(theta_ij - 0.5))**2
    J_loading = torch.mean(torch.relu(loading - 1.5))**2
    
    J_reg = J + reg_coefs['lam_reg']* (J_v + J_theta + J_loading) # [1,1]
    
    return J_reg
    
    