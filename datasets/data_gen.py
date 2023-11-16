# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:51:16 2022

@author: Benjamin Habib
"""

import numpy as np
import pandapower as pp
import pandapower.networks as nw
try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]

import pandapower.plotting.plotly as pplotly
from pandas import Series
import pandas as pd

from loadsampling import samplermontecarlo as smc
from loadsampling import samplermontecarlo_normal as smc_norm
from loadsampling import kumaraswamymontecarlo as smc_ks
from loadsampling import progressBar
from pp_to_dss_data import DSSData_from_PPNet as dssdata



import pandapower.plotting as plot

import time

s = time.time()


grid= "cigre"


# building the grid in pandapower
if grid=='ober':
    net, net1 = nw.mv_oberrhein(separation_by_sub=True, include_substations = False, scenario = "generation")
    closed_lines = set(net.line.index) - set(net.switch[(net.switch.et=="l") & (net.switch.closed==False)].element.values)
    
if grid=='cigre':
    net = nw.create_cigre_network_mv(with_der="pv_wind")
    
if grid=='ober2':
    net = nw.mv_oberrhein(separation_by_sub=False, include_substations = False, scenario = "generation")
    net.switch['closed']=True
    closed_lines = set(net.line.index) - set(net.switch[(net.switch.et=="l") & (net.switch.closed==False)].element.values)


slack_bus = net.ext_grid['bus'][0]
load_bus = net.load.sort_values('bus')['bus']
sgen_bus = net.sgen.sort_values('bus')['bus']


iteration = 1
timesteps = 24
start = 0

ts = np.arange(start,start + timesteps*iteration) # timesteps profile

# Tweak load and gen profiles
if grid=='ober':
    sgen_inc = 1.5 # parameter to tweak overall DER generation
    load_inc = 1.5 # parameter to tweak overall load
    
if grid=='cigre':
    sgen_inc = 1.4 # parameter to tweak overall DER generation
    load_inc = 1.2 # parameter to tweak overall load

    
if grid=='ober2':
    sgen_inc = 2. # parameter to tweak overall DER generation
    load_inc = .5 # parameter to tweak overall load


# Set the PandaPowrr SE algortihm's  parameters
algorithm = "wls"
init='flat'
estimator='wls_estimator with zero constraints'
zero_injection = "auto"



load_dist = "normal" # Set the load distribution type for the sampling
se_case = "bad"      # Set the level of noise in the measurements

max_loading = 90    # Threshols for line overloading

"""  Set the measurement case """
meas_case = 1      

if grid=='cigre':
    if meas_case == 1:
    
        v_bus = np.array([0,1,12,8])
        phi_bus = np.array([])
        p_bus = np.array([0])          
        p_line = np.array([0,10])
        i_line = np.array([])   
        
    if meas_case == 2:
    
        v_bus = np.array([0,1,4,12,8])
        phi_bus = np.array([])
        p_bus = np.array([0])
        p_line = np.array([0,2,6,10])
        i_line = np.array([])
      
    if meas_case == 3:
    
        v_bus = net.bus.index
        phi_bus = net.bus.index
        p_bus = np.array([])
        i_line = net.line.index 
        p_line = net.line.index 

        
if grid=='ober':
    if meas_case == 1:

        v_bus = np.array([58,39,80,86,146,81,34,142,100,50, 82, 161])
        phi_bus = np.array([])
        p_bus = np.array([])    
        p_line = np.array([162,165,185,81,60,171,70,64]) 
        i_line = np.array([162,165,185,81])
        
    if meas_case == 2:

        v_bus = np.array([58,39,80,86,146,81,34,142])
        phi_bus = np.array([])
        p_bus = np.array([])    
        p_line = np.array([162,165,185,81]) 
        i_line = np.array([])
        
    if meas_case == 3:

        v_bus = net.bus.index
        phi_bus = net.bus.index
        p_bus = np.array([])      
        i_line = net.line.index 
        p_line = net.line.index 

if grid=='ober2':
    if meas_case == 1:

        v_bus = np.array([58,39,80,86,146,81,34,142,100,50, 82, 161,318,319,6,126,245,171,273,54,167,74,33,213,237,316])
        phi_bus = np.array([])
        p_bus = slack_bus.values
        p_line = np.array([162,165,185,81,60,171,70,64,62,193,18,46,5,11,122,28,45,94]) 
        i_line = np.array([162,165,185,171,62,193,46,28])
    if meas_case == 2:

        v_bus = np.array([58,39,80,86,146,81,100,50, 318,319,6,126,245,171,273,167,33,237])
        phi_bus = np.array([])
        p_bus = slack_bus.values     
        p_line = np.array([162,165,81,60,70,62,193,18,5,122,28,94])
        i_line = np.array([162,165,62,193])
        
    if meas_case == 3:

        v_bus = net.bus.index
        phi_bus = net.bus.index
        p_bus = np.array([])    
        i_line = net.line.index 
        p_line = net.line.index 
        
        
"""Plot network with Plotly"""     
   
net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
net.line_geodata.drop(net.line_geodata.index, inplace=True)
plot.create_generic_coordinates(net, respect_switches=True) #create artificial coordinates with the igraph package

closed_lines = set(net.line.index) - set(net.switch[(net.switch.et=="l") & (net.switch.closed==False)].element.values)

hv_node = net.bus[net.bus['vn_kv']==110.0].index
hvb = pplotly.create_bus_trace(net, hv_node, size=10, color="peachpuff",  trace_name='HV bus', legendgroup= "gridelem",
                               
                                infofunc=Series(index=net.bus.index, data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))
bc = pplotly.create_bus_trace(net, net.bus.index, size=10, color="orange",  legendgroup= "gridelem",
                              infofunc=Series(index=net.bus.index,
                                              data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))

bc_v = pplotly.create_bus_trace(net, v_bus, size=15, color="indianred",  trace_name='voltage measurement',  legendgroup= "gridmeas",
                               
                              infofunc=Series(index=net.bus.index,
                                              data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))

bc_phi = pplotly.create_bus_trace(net, phi_bus, size=20, color="green", trace_name='phasor measurement',  legendgroup= "gridmeas",
                              infofunc=Series(index=net.bus.index,
                                              data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))

lc = plot.create_line_trace(net, net.line.index, color="grey", legendgroup= "gridelem",
                                infofunc=Series(index=net.line.index, 
                                                data=net.line.index.astype(str)+ '<br>' + net.line.length_km.astype(str) + ' km'))
long_lines = net.line[net.line.length_km > 2.].index
lcl = pplotly.create_line_trace(net, long_lines, color="green", width =2.5,
                                infofunc=Series(index=net.line.index, 
                                                data=net.line.index.astype(str)+ '<br>' + net.line.length_km.astype(str) + ' km'))
lcd = pplotly.create_line_trace(net, closed_lines, color="grey", legendgroup= "gridelem",
                                infofunc=Series(index=net.line.index, 
                                                data=net.line.index.astype(str)+ '<br>' + net.line.length_km.astype(str) + ' km'))
lc_p = pplotly.create_line_trace(net, p_line, color="teal", width =3, trace_name='power flow measurement',  legendgroup= "gridmeas",
                                infofunc=Series(index=net.line.index, 
                                                data=net.line.index.astype(str)+ '<br>' + net.line.length_km.astype(str) + ' km'))

tc = pplotly.create_trafo_trace(net, net.trafo.index, trace_name='trafos', color="slateblue",
                                infofunc=Series(index=net.trafo.index, 
                                                data=net.trafo.index.astype(str)+ '<br>' + net.trafo.shift_degree.astype(str) + ' degree'))

meas_bus = np.array([1,5])

m_bus = pplotly.create_bus_trace(net, meas_bus, size=25, color="green", patch_type="circle-open", trace_name='chosen bus',legendgroup= "gridmeas",
                              infofunc=Series(index=net.bus.index,
                                              data=net.bus.index.astype(str) + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))

meas_line = np.array([0,2])
m_line = pplotly.create_line_trace(net, meas_line, color="green", width =8,  trace_name='chosen line meas.',legendgroup= "gridmeas",
                                infofunc=Series(index=net.line.index, 
                                                data=net.line.index.astype(str)+ '<br>' + net.line.length_km.astype(str) + ' km'))
fig=pplotly.draw_traces(m_line+tc+lc_p + lcd+ bc + bc_phi + bc_v + hvb + m_bus, figsize=1, showlegend=False)#, aspectratio=(3,1));


  

""" Measurement Noise specification """


if se_case == "perso":
    
    meas_noise = 0.02 # coefficient of power measurement noise
    v_noise = 0.01 # coefficient of voltage sensors noise
    i_noise = 0.01
    zero_inj_coef = 0.001
    
    load_error = 0.2 # coef load error
    sgen_error = 0.15 # coef sgen error

    pm_noise = load_error/2  # 95% confidence
    sgen_noise = sgen_error/2

    
if se_case == "bad":
    
    meas_noise = 0.05 # coefficient of power measurement noise
    v_noise = 0.03 # coefficient of voltage sensors noise
    i_noise = 0.03
    zero_inj_coef = 0.001
    
    load_error = 0.3 # coef load error
    sgen_error = 0.2 # coef sgen error

    pm_noise = load_error/2  # 95% confidence
    sgen_noise = sgen_error/2

    
if se_case == "good":
    
    meas_noise = 0.01 # coefficient of power measurement noise
    v_noise = 0.005 # coefficient of voltage sensors noise
    i_noise = 0.005
    zero_inj_coef = 0.001

    load_error = 0.15 # coef load error
    sgen_error = 0.1 # coef sgen error
    
    pm_noise = load_error/2  # 95% confidence
    sgen_noise = sgen_error/2



""" Profiles """

profile_day_pq = np.array([0.2, 0.2, 0.2,0.2,0.3,0.3,
                           0.4,0.4,0.5,0.5,0.7,0.8,
                           0.9,1.0,1.0,1.0,0.9,0.8,
                           0.8,0.7,0.7,0.6,0.5,0.3])

profile_day_sun = np.array([0., 0., 0.,0.,0.,0.,
                            0.1,0.25,0.4,0.7,0.9,1.,
                            1.,1.0,1.0,1.0,0.9,0.8,
                            0.6,0.4,0.3,0.1,0.,0.])

profile_day_wind = np.array([0.6, 0.6, 0.7,0.5,0.4,0.4,
                             0.5,0.7,0.8,0.7,0.5,0.5,
                             0.4,0.5,0.4,0.5,0.6,0.6,
                             0.3,0.4,0.7,0.6,0.4,0.5])

industry_load_profile = np.array([0.35, 0.35, 0.3,0.3,0.4,0.5,
                                  0.6,0.9,1.,1.,1.,0.9,
                                  0.85,0.85,0.85,0.85,0.8,0.55,
                                  0.5,0.45,0.4,0.4,0.35,0.35])

household_load_profile = np.array([0.25, 0.2, 0.2,0.2,0.2,0.25,
                                  0.4,0.65,0.65,0.65,0.7,0.6,
                                  0.7,0.65,0.55,0.5,0.45,0.6,
                                  0.8,0.9,0.8,0.7,0.55,0.3])

"""Load sampling"""

load_p = pd.DataFrame()
load_q = pd.DataFrame()
load_sgen = pd.DataFrame()

for i in range(household_load_profile.size): 
    
    
    load_p[i]= pd.concat([net.load['p_mw'][:10].mul(household_load_profile[i]),net.load['p_mw'][10:].mul(industry_load_profile[i])]) * load_inc
    load_q[i]= pd.concat([net.load['q_mvar'][:10].mul(household_load_profile[i]),net.load['q_mvar'][10:].mul(industry_load_profile[i])]) * load_inc
    load_sgen[i]= net.sgen['p_mw'].mul(profile_day_sun[i]) * sgen_inc
    load_sgen.loc[8] = net.sgen['p_mw'][8] * profile_day_wind[i] *sgen_inc
    
    


numpy_load_p = load_p.to_numpy()
unroll_load_p = np.reshape(numpy_load_p,load_p.size)

numpy_load_sgen = load_sgen.to_numpy()
unroll_load_sgen = np.reshape(numpy_load_sgen,load_sgen.size)

if load_dist == "uniform":
    mc_unroll_load_p = smc(unroll_load_p * (1- load_error), unroll_load_p * (1 + load_error), iteration)
    mc_unroll_load_sgen = smc(unroll_load_sgen * (1- sgen_error), unroll_load_sgen * (1 + sgen_error), iteration)
    
if load_dist == "normal":
    mc_unroll_load_p = smc_norm(unroll_load_p, unroll_load_p * pm_noise, iteration)
    mc_unroll_load_sgen = smc_norm(unroll_load_sgen, unroll_load_sgen * sgen_noise, iteration)   
    
if  load_dist == "kumaraswamy":
    
    mc_unroll_load_p = smc_ks(unroll_load_p * (1- load_error), unroll_load_p * (1 + load_error), iteration)
    mc_unroll_load_sgen = smc_ks(unroll_load_sgen * (1- sgen_error), unroll_load_sgen * (1 + sgen_error), iteration) 
    
mc_load_p = np.reshape(mc_unroll_load_p,[load_p.shape[0],load_p.shape[1]*iteration])
pd_mc_load_p = pd.DataFrame(mc_load_p)

mc_load_sgen = np.reshape(mc_unroll_load_sgen,[load_sgen.shape[0],load_sgen.shape[1]*iteration])
pd_mc_load_sgen = pd.DataFrame(mc_load_sgen)


pd_mc_load_q = pd_mc_load_p.mul((net.load["q_mvar"]/net.load["p_mw"]),axis=0)



"""Deviation load actual vs profile: RMSE""";

mape_load = pd.DataFrame(columns = np.arange(iteration))
mape_sgen = pd.DataFrame(columns = np.arange(iteration))

load_scen= pd.DataFrame(data = 0,columns = range(24), index= net.bus.index)
load_pro= pd.DataFrame(data = 0,columns = range(24), index= net.bus.index)
for t in range(24):
    

    for i in net.bus.index:
        
        load_index = net.load[net.load['bus']==i].index
        sgen_index = net.sgen[net.sgen['bus']==i].index
        
        
        for j in sgen_index:
           load_scen[t][i] -= pd_mc_load_sgen[t][j] 
           load_pro[t][i] -= load_sgen[t][j]
            
        for j in load_index:
           load_scen[t][i] += pd_mc_load_p[t][j]
           load_pro[t][i] -= load_p[t][j]
           



for i in range(iteration):

    dev_load = np.abs(np.divide(mc_load_p[:,i::iteration] - numpy_load_p, mc_load_p[:,i::iteration], \
                                out=np.zeros_like(numpy_load_p), where=mc_load_p[:,i::iteration]!=0))
        
    mape_load[i] = np.mean(dev_load,axis=1)
    
    dev_sgen = np.abs(np.divide(mc_load_sgen[:,i::iteration] - numpy_load_sgen,mc_load_sgen[:,i::iteration], \
                                out=np.zeros_like(numpy_load_sgen), where=mc_load_sgen[:,i::iteration]!=0))
        
    mape_sgen[i] = np.mean(dev_sgen,axis=1)


bus_mape_load = mape_load.mean(axis=1) *100
bus_mape_sgen = mape_sgen.mean(axis=1) *100



""" Power Flow """

pf_vm=pd.DataFrame(columns = ts)
pf_va=pd.DataFrame(columns = ts)
pf_p=pd.DataFrame(columns = ts)
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

start_pf = time.time()

start_pf = time.time()

for t in ts:
    
    
    net.load['p_mw'] = pd_mc_load_p[t]
    net.load['q_mvar'] = pd_mc_load_q[t]
    net.sgen['p_mw'] = pd_mc_load_sgen[t]
    
    pp.runpp(net,calculate_voltage_angles=True)
    
    pf_vm[t] = net.res_bus["vm_pu"]
    pf_va[t] = net.res_bus["va_degree"]
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
    
duration_pf = time.time() - start_pf

A_dss = np.array([])
B_dss = np.array([])
U_dss = np.array([])


pf_va *= (np.pi/180.)

"""Build Dataset""" 


for t in progressBar(ts, prefix = 'Progress:', suffix = 'Complete', length = 50):
    
 
    """ Load per bus """

    pm_bus= pd.DataFrame(data = {"p_mw": 0, "q_mvar": 0}, index= net.bus.index)

    for i in net.bus.index:
        
        load_index = net.load[net.load['bus']==i].index
        sgen_index = net.sgen[net.sgen['bus']==i].index
        
        
        for j in sgen_index:
            pm_bus["p_mw"][i] -= load_sgen[t%load_p.shape[1]][j] 
          
            
        for j in load_index:
            pm_bus["p_mw"][i] += load_p[t%load_p.shape[1]][j]
            pm_bus["q_mvar"][i] += load_q[t%load_p.shape[1]][j]
   
    nan = np.nan
    
    
    meas_bus = pd.DataFrame({"vm_pu": pf_vm[t][v_bus] + np.random.normal(loc=0., scale = np.abs(pf_vm[t][v_bus] * v_noise)),\
                             "va_rad": pf_va[t][phi_bus] + np.random.normal(loc=0.,  scale = np.abs(pf_va[t][phi_bus] * v_noise)), \
                                "p_mw": pf_p[t][p_bus] + np.random.normal(loc= 0. , scale = np.abs(pf_p[t][p_bus] * meas_noise)), \
                             "q_mvar": pf_q[t][p_bus] + np.random.normal(loc=0. , scale = np.abs(pf_q[t][p_bus] * meas_noise))}, index = net.bus.index)
        
    edge_index = np.concatenate((net.line.index, net.trafo.index + net.line.index.size))
    
    meas_edge = pd.DataFrame({"p_from_mw": pf_pl[t][p_line] + np.random.normal(loc=0., scale = np.abs(pf_pl[t][p_line] * meas_noise)), \
                              "q_from_mvar":pf_ql[t][p_line] + np.random.normal(loc=0., scale = np.abs(pf_ql[t][p_line] * meas_noise)), \
                              "i_from_ka": pf_il[t][i_line] + np.random.normal(loc=0., scale = np.abs(pf_il[t][i_line] * i_noise))}, index = edge_index) 
                                                                             
        
        

    labels = pd.DataFrame({"vm_pu": pf_vm[t], "va_rad": pf_va[t]}, index = net.bus.index)
    
    


    meas_error = np.array([v_noise, v_noise,meas_noise , meas_noise, i_noise ,pm_noise, \
                                               zero_inj_coef], dtype=float)
    
    A,B,U = dssdata(net, pm_bus, meas_bus, meas_edge, meas_error, labels)
    
    if A_dss.size == 0:
        
        A_dss = np.atleast_3d(A)
        B_dss = np.atleast_3d(B)
        U_dss = np.atleast_3d(U)

        
    else:
    
        A_dss = np.dstack((A_dss,A))
        B_dss = np.dstack((B_dss,B))
        U_dss = np.dstack((U_dss,U))

        
A_dss = np.moveaxis(A_dss, 2,0)
B_dss = np.moveaxis(B_dss, 2,0)
U_dss = np.moveaxis(U_dss, 2,0)

np.save("data_"+grid+"/A_"+se_case+"_meas"+str(meas_case), A_dss)    
np.save("data_"+grid+"/B_"+se_case+"_meas"+str(meas_case), B_dss)
np.save("data_"+grid+"/U_"+se_case+"_meas"+str(meas_case), U_dss)


    

    
