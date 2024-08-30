import numpy as np 
import pandas as pd 
import pandapower as pp 
import pandapower.plotting.plotly as pplotly
import torch
import time
import numba
from data import get_edge_param, get_bus_param
import pickle

from loadsampling import samplermontecarlo as smc
from loadsampling import samplermontecarlo_normal as smc_norm
from loadsampling import kumaraswamymontecarlo as smc_ks
from loadsampling import progressBar

# Constants
NETWORKS: int = 1  # Number of networks to simulate
ITERATIONS: int = 365  # Number of ITERATIONS (days)
P_NOISE: float = 0.02  # Coefficient of power measurement noise
V_NOISE: float = 0.01  # Coefficient of voltage sensors noise
I_NOISE: float = 0.01
ZERO_INJ_COEF: float = 0.001
PM_ERROR: float = 0.3  # Coefficient of load error
SGEN_ERROR: float = 0.25  # Coefficient of sgen error
PM_NOISE: float = PM_ERROR / 2  # 95% confidence
SGEN_NOISE: float = SGEN_ERROR / 2
POWER_COEF: float = 0.9  # Power coefficient
LOAD_DIST: str = 'normal'

# Lists to store results
df_nodes_list_mul = []
df_edges_list_mul = []
df_labels_list_mul = []
df_pm_list_mul = []
df_edges_param_mul = []
df_nodes_param_mul = []

# Loop over each network
for i in range(NETWORKS):
    # Create different network configurations
    if i < 2:
        grid = 'cigre14'
        net = pp.networks.create_cigre_network_mv(with_der="pv_wind")
        if i == 1:
            grid = 'cigre14_reswitched'
            net.switch['closed'] = True
            net.switch['closed'][0] = False
            net.switch['closed'][3] = False
    else:
        grid = 'ober_sub'
        net, net1 = pp.networks.mv_oberrhein(scenario='generation', cosphi_load=0.98, cosphi_pv=1.0, include_substations=False, separation_by_sub=True)

    # Time series for 24 hours over the number of ITERATIONS
    ts = np.arange(24 * ITERATIONS) 

    # Flags for plotting and generation
    plotting = True
    sgen_prod = True

    # Load distribution type and power coefficient
    

    # Plot network with Plotly
    if plotting:
        net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
        net.line_geodata.drop(net.line_geodata.index, inplace=True)
        pp.plotting.create_generic_coordinates(net, respect_switches=True)

        hv_node = net.bus[net.bus['vn_kv'] == 110.0].index
        hvb = pplotly.create_bus_trace(net, hv_node, size=10, color="peachpuff", trace_name='HV bus', legendgroup="gridelem", infofunc=pd.Series(index=net.bus.index, data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + 'kV'))
        bc = pplotly.create_bus_trace(net, net.bus.index, size=10, color="orange", legendgroup="gridelem", infofunc=pd.Series(index=net.bus.index, data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))
        lc = pp.plotting.create_line_trace(net, net.line.index, color="grey", legendgroup="gridelem", infofunc=pd.Series(index=net.line.index, data=net.line.index.astype(str) + '<br>' + net.line.length_km.astype(str) + ' km'))
        tc = pplotly.create_trafo_trace(net, net.trafo.index, trace_name='trafos', color="slateblue", infofunc=pd.Series(index=net.trafo.index, data=net.trafo.index.astype(str) + '<br>' + net.trafo.shift_degree.astype(str) + ' degree'))
        closed_lines = set(net.line.index) - set(net.switch[(net.switch.et == "l") & (net.switch.closed == False)].element.values)
        cl = pplotly.create_line_trace(net, closed_lines, color="grey", legendgroup="gridelem", infofunc=pd.Series(index=net.line.index, data=net.line.index.astype(str) + '<br>' + net.line.length_km.astype(str) + ' km'))
        fig = pplotly.draw_traces(tc + cl + bc + hvb, figsize=1, showlegend=False)

    # Set DER generation to zero if sgen_prod is False
    if not sgen_prod:
        net.sgen['p_mw'] = 0.

    # Load profiles for household and industry
    household_load_profile = np.array([0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.4, 0.65, 0.65, 0.65, 0.7, 0.6, 0.7, 0.65, 0.55, 0.5, 0.45, 0.6, 0.8, 0.9, 0.8, 0.7, 0.55, 0.3])
    industry_load_profile = np.array([0.35, 0.35, 0.3, 0.3, 0.4, 0.5, 0.6, 0.9, 1., 1., 1., 0.9, 0.85, 0.85, 0.85, 0.85, 0.8, 0.55, 0.5, 0.45, 0.4, 0.4, 0.35, 0.35])

    # Generation profiles for sun and wind
    profile_day_sun = np.array([0., 0., 0., 0., 0., 0., 0.1, 0.25, 0.4, 0.7, 0.9, 1., 1., 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.1, 0., 0.])
    profile_day_wind = np.array([0.6, 0.6, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.8, 0.7, 0.5, 0.5, 0.4, 0.5, 0.4, 0.5, 0.6, 0.6, 0.3, 0.4, 0.7, 0.6, 0.4, 0.5])


    df_noise = pd.DataFrame({'p_noise': P_NOISE, 'v_noise': V_NOISE, 'i_noise': I_NOISE, 'PM_NOISE': PM_NOISE, 'SGEN_NOISE': SGEN_NOISE, 'zero_inj_coef': ZERO_INJ_COEF}, index=['def_value'])

    # Load sampling for scenario generation
    load_p = pd.DataFrame()
    load_sgen = pd.DataFrame()

    # Masks for different types of loads and generation
    load_r_mask = net.load['name'].str.contains('R').astype(float)
    load_ind_mask = net.load['name'].str.contains('CI').astype(float)
    load_lv_mask = net.load['name'].str.contains('LV').astype(float)
    load_mv_mask = net.load['name'].str.contains('MV').astype(float)
    sgen_pv_mask = net.sgen['name'].str.contains('PV').astype(float)
    sgen_wind_mask = net.sgen['name'].str.contains('WKA').astype(float)
    sgen_static_mask = net.sgen['name'].str.contains('Static').astype(float)

    # Generate load profiles for each hour of the day
    for i in range(household_load_profile.size):  
        load_p[i] = (load_r_mask + load_lv_mask) * net.load['p_mw'].mul(household_load_profile[i]) + (load_ind_mask + load_mv_mask) * net.load['p_mw'].mul(industry_load_profile[i])
        load_sgen[i] = (sgen_pv_mask + sgen_static_mask) * net.sgen['p_mw'].mul(profile_day_sun[i]) + sgen_wind_mask * net.sgen['p_mw'].mul(profile_day_wind[i]) 

    # Convert load profiles to numpy arrays and unroll them
    numpy_load_p = load_p.to_numpy()
    unroll_load_p = np.reshape(numpy_load_p, load_p.size)
    numpy_load_sgen = load_sgen.to_numpy()
    unroll_load_sgen = np.reshape(numpy_load_sgen, load_sgen.size)

    # Perform Monte Carlo sampling based on load distribution type
    if LOAD_DIST == "uniform":
        mc_unroll_load_p = smc(unroll_load_p * (1 - PM_ERROR), unroll_load_p * (1 + PM_ERROR), ITERATIONS)
        mc_unroll_load_sgen = smc(unroll_load_sgen * (1 - SGEN_ERROR), unroll_load_sgen * (1 + SGEN_ERROR), ITERATIONS)
    elif LOAD_DIST == "normal":
        mc_unroll_load_p = smc_norm(unroll_load_p, unroll_load_p * PM_NOISE, ITERATIONS)
        mc_unroll_load_sgen = smc_norm(unroll_load_sgen, unroll_load_sgen * SGEN_NOISE, ITERATIONS)   
    elif LOAD_DIST == "kumaraswamy":
        mc_unroll_load_p = smc_ks(unroll_load_p * (1 - PM_ERROR), unroll_load_p * (1 + PM_ERROR), ITERATIONS)
        mc_unroll_load_sgen = smc_ks(unroll_load_sgen * (1 - SGEN_ERROR), unroll_load_sgen * (1 + SGEN_ERROR), ITERATIONS) 

    # Reshape Monte Carlo samples back to original shape
    mc_load_p = np.reshape(mc_unroll_load_p, [load_p.shape[0], load_p.shape[1] * ITERATIONS])
    pd_mc_load_p = pd.DataFrame(mc_load_p, index=net.load.index)
    mc_load_sgen = np.reshape(mc_unroll_load_sgen, [load_sgen.shape[0], load_sgen.shape[1] * ITERATIONS])
    pd_mc_load_sgen = pd.DataFrame(mc_load_sgen, index=net.sgen.index)

    # Calculate reactive power load
    pd_mc_load_q = pd_mc_load_p.mul(POWER_COEF, axis=0)

    # DataFrames to store power flow results
    pf_vm = pd.DataFrame(columns=ts)
    pf_va = pd.DataFrame(columns=ts)
    pf_p = pd.DataFrame(columns=ts)
    pf_q=pd.DataFrame(columns = ts)

    pf_pl=pd.DataFrame(columns = ts)
    pf_ql=pd.DataFrame(columns = ts)
    pf_plt=pd.DataFrame(columns = ts)
    pf_qlt=pd.DataFrame(columns = ts)

    pf_il=pd.DataFrame(columns = ts)
    pf_ilt=pd.DataFrame(columns = ts)
    pf_ilm=pd.DataFrame(columns = ts)
    pf_loading=pd.DataFrame(columns = ts)

    pf_pt=pd.DataFrame(columns = ts)
    pf_qt=pd.DataFrame(columns = ts)
    pf_it=pd.DataFrame(columns = ts)

    pf_ptl=pd.DataFrame(columns = ts)
    pf_qtl=pd.DataFrame(columns = ts)
    pf_itl=pd.DataFrame(columns = ts)

    
# Start the power flow calculation timer
start_pf = time.time()

# Initialize lists to store data for nodes, edges, labels, and power measurements
df_nodes_list = []
df_edges_list = []
df_labels_list = []
df_pm_list = []

# Iterate over each time step with a progress bar
for t in progressBar(ts, prefix='Progress:', suffix='Complete', length=50):
    
    # Update load and sgen values in the network for the current time step
    net.load['p_mw'] = pd_mc_load_p[t]
    net.load['q_mvar'] = pd_mc_load_q[t]
    net.sgen['p_mw'] = pd_mc_load_sgen[t]
    
    # Run power flow calculation
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    
    # Get bus parameters
    df_bus_param = get_bus_param(net)
    
    # Store results of the power flow calculation
    pf_vm[t] = net.res_bus["vm_pu"]
    net.res_bus["va_degree"][df_bus_param['bool_slack'] == 0.] += net.trafo["shift_degree"].values[0]
    net.res_bus["va_rad"] = net.res_bus["va_degree"] * np.pi / 180
    
    pf_va[t] = net.res_bus["va_rad"]
    pf_p[t] = net.res_bus["p_mw"]
    pf_q[t] = net.res_bus["q_mvar"]
    
    pf_pl[t] = net.res_line["p_from_mw"]
    pf_ql[t] = net.res_line["q_from_mvar"]
    pf_plt[t] = net.res_line["p_to_mw"]
    pf_qlt[t] = net.res_line["q_to_mvar"]
    
    pf_il[t] = net.res_line["i_from_ka"]
    pf_ilt[t] = net.res_line["i_to_ka"]
    pf_ilm[t] = net.res_line["i_ka"]
    pf_loading[t] = net.res_line["loading_percent"]
    
    pf_pt[t] = net.res_trafo["p_hv_mw"]
    pf_qt[t] = net.res_trafo["q_hv_mvar"]
    pf_it[t] = net.res_trafo["i_hv_ka"]
    pf_ptl[t] = net.res_trafo["p_lv_mw"]
    pf_qtl[t] = net.res_trafo["q_lv_mvar"]
    pf_itl[t] = net.res_trafo["i_lv_ka"]
    
    # Get edge parameters
    df_edges_param = get_edge_param(net)
    
    # Create a DataFrame for nodes by concatenating bus parameters and results
    df_nodes = pd.concat([df_bus_param, net.res_bus[['vm_pu', 'va_rad', 'p_mw', 'q_mvar']]], axis=1)
    df_nodes.index = np.arange(net.bus.index.size)
    
    # Create a DataFrame for edges by concatenating line and transformer results
    res_trafo = net.res_trafo[['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar', 'pl_mw', 'ql_mvar', 'i_hv_ka', 'i_lv_ka', 'loading_percent']]
    res_trafo.columns = ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'ql_mvar', 'i_from_ka', 'i_to_ka', 'loading_percent']
    res_edges = pd.concat([net.res_line[['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'ql_mvar', 'i_from_ka', 'i_to_ka', 'loading_percent']], res_trafo], axis=0)
    
    df_edges = pd.concat([df_edges_param, res_edges], axis=1)
    df_edges.index = np.arange(df_edges.index.size)
    
    # Append the DataFrames to the respective lists
    df_nodes_list.append(df_nodes)
    df_edges_list.append(df_edges)
    df_labels_list.append(df_nodes[['vm_pu', 'va_rad']])
    
# Save the data to files
with open('data/' + grid + '/nodes', 'wb') as file:
    pickle.dump(df_nodes_list, file)
with open('data/' + grid + '/edges', 'wb') as file:
    pickle.dump(df_edges_list, file)
with open('data/' + grid + '/labels', 'wb') as file:
    pickle.dump(df_labels_list, file)
with open('data/' + grid + '/edge_param', 'wb') as file:
    pickle.dump(df_edges_param, file)
with open('data/' + grid + '/bus_param', 'wb') as file:
    pickle.dump(df_bus_param, file)
with open('data/' + grid + '/noise_param', 'wb') as file:
    pickle.dump(df_noise, file)




