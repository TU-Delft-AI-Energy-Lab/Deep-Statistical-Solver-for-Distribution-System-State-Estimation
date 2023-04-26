# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:18:10 2022

@author: Benjamin Habib
"""


import numpy as np
import pandas as pd




def DSSData_from_PPNet(net,pm_bus,meas_bus, meas_line, meas_error, labels): #removed meas_trafo
    
    """
    - Function taking a PandaPower's network and convert it to a DSS data sample. Pseudomeasurements and measurements location are provided as well for labelling and input
    
    - Input:
        - net: PandaPower Network, with detailed topology, busses, trafos and lines parameters, and load an generators.
        
        - pm_bus = {"p_mw" : pm_p, "q_mvar": pm_q} Dataframe of pseudomeasurements of bus p and q injection. 
            - df: [index = net.bus.index, column = p_mw, q_mvar]
            
        - meas_bus = {"vm_pu": meas_v, "va_rad": meas_phi, "p_mw": meas_p, "q_mvar": meas_q}: DataFrame of measurements v, phi, p and q on busses.
            - df: [index = net.bus.index, column = vm_pu, va_rad, p_mw, q_mvar]
            
        - meas_line = {"p_from_mw": meas_pf, "q_from_mvar": meas_qf, "i_from_ka": meas_if} REMOVED:{, "p_to_mw": meas_pt, "q_to_mvar": meas_qt, "i_to_ka": meas_it}:
            measurements p, q, i (from (and to)) on lines AND TRAFOS
        
        - meas_error = [error_v, error_phi, error_p, error_pl, error_i, error_pm, error_zero_inj], with error_p = error_q = error_pflow = error_qflow
        
        - labels = {"vm_pu": node_vm, "va_angle": node_va}
        
        
        
    - Output: 
        - data: DSS data instance = {A,B,U}
    
    
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
    error_phi = np.absolute(meas_error[1]  * np.nan_to_num(meas_phi))
    
    meas_p = meas_bus["p_mw"].values.astype(float)
    error_p = np.absolute(meas_error[2]  * np.nan_to_num(meas_p))
    
    meas_q = meas_bus["q_mvar"].values.astype(float)
    error_q = np.absolute(meas_error[2]  * np.nan_to_num(meas_q))
     
    aggr_p = np.nan_to_num(meas_p) + pm_p * (np.isnan(meas_p))
    error_p += np.absolute(meas_error[5] * (np.isnan(meas_p)) * pm_p)
    
    aggr_q = np.nan_to_num(meas_q) + pm_q * (np.isnan(meas_q))
    error_q += np.absolute(meas_error[5] * (np.isnan(meas_q)) * pm_q)
    
    error_zero_inj =  np.absolute(meas_error[6] * np.isnan(meas_p)) 
    bool_zero_inj = pd.DataFrame(data ={"type": 0}, index = net.bus.index)


    trafo_bus = np.concatenate([net.trafo["hv_bus"].to_numpy(),net.trafo["lv_bus"].to_numpy()])
    
    bool_slack = pd.DataFrame(data ={"type": 0}, index = net.bus.index)
    bool_trafo = pd.DataFrame(data ={"type": 0}, index = net.bus.index)
    
    for i in net.bus.index:  # 0: aux_bus, 1: load_bus, 2: sgen_bus, 3: load+sgen_bus, 4: slack_bus
        
        if any(trafo_bus==i):
            bool_trafo.loc[i] = 1

            
        if not (any(slack_bus==i) or (any(load_bus==i) or any(sgen_bus==i))):
            bool_zero_inj.loc[i] = 1
            
        if any(slack_bus==i) :
             bool_slack.loc[i] = 1

    bool_trafo = bool_trafo.values[:,0]
    bool_slack = bool_slack.values[:,0]
    bool_zero_inj = bool_zero_inj.values[:,0]

    
    error_p = (1 - bool_zero_inj) * error_p + bool_zero_inj * error_zero_inj
    error_q = (1 - bool_zero_inj) * error_q + bool_zero_inj * error_zero_inj
    
    node_features = np.nan_to_num(np.array([new_node_index,meas_v,error_v,meas_phi,error_phi,aggr_p,\
                                            error_p,aggr_q,error_q,bool_trafo,bool_zero_inj,bool_slack]).T) #shape = (num nodes, num features)
    

    
    p_line_from = meas_line["p_from_mw"].values.astype(float)
    error_pl = np.concatenate([meas_error[3]* (np.nan_to_num(p_line_from)[:net.line.index.size]),\
                               np.ones(net.trafo.index.size)]) 
    
    q_line_from = meas_line["q_from_mvar"].values.astype(float)
    error_ql = np.concatenate([meas_error[3]* (np.nan_to_num(q_line_from)[:net.line.index.size]),\
                               np.ones(net.trafo.index.size)])
    
    i_line_from = meas_line["i_from_ka"].values.astype(float)
    error_il = np.concatenate([meas_error[4]* (np.nan_to_num(i_line_from)[:net.line.index.size]),\
                               np.ones(net.trafo.index.size)]) 
    
    edge_length = net.line["length_km"]
    edge_r = net.line["r_ohm_per_km"] * edge_length
    edge_x = net.line["x_ohm_per_km"] * edge_length
    
    edge_c = net.line["c_nf_per_km"] * edge_length
    edge_b =  -2 * np.pi * net.f_hz * edge_c * 1e-9
    edge_g = net.line["g_us_per_km"] * edge_length * 1e-6

    edge_df = net.line["df"]
    edge_parallel = net.line["parallel"]

    
    net.trafo.index += max(net.line.index)+1
    
    t_r = (net.trafo["vkr_percent"]/100) * (net.sn_mva/net.trafo["sn_mva"])
    t_z = (net.trafo["vk_percent"]/100) * (net.sn_mva/net.trafo["sn_mva"])
    t_x_square =  t_z.pow(2) - t_r.pow(2) 
    t_x = t_x_square.pow(0.5)
    
    t_g = (net.trafo["pfe_kw"]/1000) * (net.sn_mva/net.trafo["sn_mva"]**2)
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
    
 
    t_parallel = net.trafo["parallel"]
    t_df = net.trafo["df"]
    t_phase_shift = net.trafo["shift_degree"]*np.pi/180
    
    edge_phase_shift = np.concatenate((np.zeros(net.line.index.size), t_phase_shift.values))

    
    
    edge_parallel = pd.concat([edge_parallel, t_parallel])
    edge_df = pd.concat([edge_df, t_df])
    
    edge_type = pd.DataFrame({"type": np.append(np.zeros(net.line.index.size), np.ones(net.trafo.index.size))})


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
    
    
    edge_features = np.array([new_edge_source, new_edge_target, np.real(edge_Y),np.imag(edge_Y),np.nan_to_num(np.real(edge_Ys)),np.nan_to_num(np.imag(edge_Ys)),np.nan_to_num(p_line_from),\
                                  error_pl, np.nan_to_num(q_line_from), error_ql, np.nan_to_num(i_line_from), error_il, \
                                  bool_closed_line, edge_type["type"].values, edge_phase_shift]).T
    
    net.trafo.index -= net.line.index.size
    
    B = node_features 
    A = edge_features
    
    U = labels.values
    return A,B,U