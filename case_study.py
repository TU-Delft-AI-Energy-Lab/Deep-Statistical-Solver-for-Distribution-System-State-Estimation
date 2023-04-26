
import numpy as np
import pandapower as pp
import pandapower.estimation as est
import pandapower.networks as ppn
import matplotlib.pyplot as plt
import pandas as pd


from loadsampling import samplermontecarlo as smc
from loadsampling import samplermontecarlo_normal as smc_norm
from loadsampling import kumaraswamymontecarlo as smc_ks
from loadsampling import progressBar



import tensorflow as tf
from fun_dss import DeepStatisticalSolver2 as dss
from fun_dss import preprocess_data, train_model, test_model, get_pflow, extract_fn, train_model_sup
from fun_dss import DSSData_from_PPNet as dssdata


import os
import sys
import time



"""
 Choosing the case study:
     
     - grid: (Used for training and testing)
         - 'cigre': 14-bus MV grid
         - 'ober': 70-bus MV/LV grid
         - 'ober2': 179-bus MV/LV grid
         
     
    - data_directory: Should be linked to grid, change directory if needed
    
    - case: (Only for training purpose)
        - Choose a measurement set to train on. Default is 'meas1' that uses the data from measurement set 1

"""

grid= "cigre"

if grid=='ober':
    data_directory = "datasets/data_oberrhein"
    
if grid=='cigre':
    data_directory = "datasets/data_cigre14"
    
if grid=='ober2':
    data_directory = "datasets/data_ober2"

case = "meas1"

try:
    

    from problem_dss import Problem

    problem = Problem(grid)

except ImportError:
    print('You should provide a compatible "problem.py" file in your data folder!')
    
    
    
# Setting up hyper-parameters

lamda = 0.8
latent_dim=40
hidden_layers=3
steps = 20
step_size = 1./steps
rate = 0.4
l2_reg = 0.002
lr=0.006
non_lin = 'tanh'
norm = 4000

num_epochs = 200
minibatch_size = 320

# Choosing number of models to train
num_rep = 1

models = {}

# Can specifcy a model to load, otherwise will generate a new model
saved_model = "saved_models/"+grid+"_"+case+"_dss"

# Set boolean if wanting to save the model
saving = False

for m in range(num_rep):
    

    try:
        
        models[m] = tf.keras.models.load_model(saved_model)
               
    except OSError:
        
        print("no model here!")
               
        models[m] = dss(name="test",
                    problem=problem,
                    latent_dimension=latent_dim,
                    hidden_layers=hidden_layers,
                    time_step_size = step_size,
                    rate = rate,
                    l2_reg = l2_reg,
                    non_lin=non_lin)
        
        optimizer = tf.keras.optimizers.Adamax(lr)
        models[m] = train_model(models[m], problem, lr, lamda, norm, num_epochs, minibatch_size,data_directory, case,optimizer,grid)
        test_model(models[m], problem, minibatch_size, data_directory, case,grid)
                
        if saving:
            models[m].save("datasets\saved_models\_"+grid+"_"+str(case)+"_dss_model_"+str(m))


# Loading grid for PandaPower
if grid=='cigre':
    net = ppn.create_cigre_network_mv(with_der="pv_wind")
if grid=='ober':
    net, net1 = ppn.mv_oberrhein(separation_by_sub=True, include_substations = False, scenario = "generation")
if grid=='ober2':
    net = ppn.mv_oberrhein(separation_by_sub=False, include_substations = False, scenario = "generation")
    net.switch['closed']=True
    net.trafo.index = np.arange(1,net.trafo.index.size+1) + net.line.index[-1]

# Setting the number of days (iteration) and hours through a profile day (timesteps), and a starting point (start)
iteration = 5
timesteps = 24
start = 0

ts = np.arange(start,start + timesteps*iteration) # total samples


# Parameters for WLS algorithm
algorithm = "wls"
init='flat'
estimator='wls_estimator with zero constraints'
zero_injection = "auto"


# Parameters for load sampling
load_dist = "normal"



# Setting numbers of case studies
sets = np.arange(10)

# Initializations of DataFrames  and dicts to save performences
t_df = pd.DataFrame(index = sets, columns = ts)
duration_se =pd.DataFrame(index = sets, columns = ts)

res_load = pd.DataFrame(index = sets, columns = ts)
res_v = pd.DataFrame(index = sets, columns = ts)

res_load_perc = pd.DataFrame(index = sets, columns = ts)
res_v_perc = pd.DataFrame(index = sets, columns = ts)

mae_v = pd.DataFrame(index = sets, columns = ts)
mae_load = pd.DataFrame(index = sets, columns = ts)

bus_vrmse = {}
line_loadrmse = {}

bus_vmae = {}
line_loadmae = {}

bus_vrmsep = {}
line_loadrmsep = {}

dss_metrics = pd.DataFrame({"RMSE V": 0., "RMSE load": 0., "RMSE load line only": 0., "RMSE% V": 0., "RMSE% load": 0., \
                            "MAE V": 0., "MAE load": 0., "Mean duration": 0., "Convergence rate": 1.}, index=sets)

wls_metrics = pd.DataFrame({"RMSE V": 0., "RMSE load": 0., "RMSE load line only": 0., "RMSE% V": 0., "RMSE% load": 0., \
                            "MAE V": 0., "MAE load": 0., "Mean duration": 0., "Convergence rate": 0.}, index=sets)

# Defining default measurement set and a special one to try the model on   
mc_def = 1
mc_spe = 3


""" Performing the case studies """

for s in sets:
    
    
    """  Defining the case studies """
    if grid=='cigre':

    
        if s==0:
           se_case = "perso"    # Defines the measurement errors srt case
           meas_case = mc_def   # Defines the measurement set case
           
           v_wrong = np.array([])  # Set some buses to provide wrong values of V
           p_wrong = np.array([])  # Set some buses to provide wrong values of P
           v_miss = np.array([])   # Set some buses to show missing values of V
           
        if s==1:
           se_case = "bad" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           v_miss = np.array([])
           
        if s==2:
           se_case = "good" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           v_miss = np.array([])
           
        if s==3:
           se_case = "perso" 
           meas_case = mc_spe
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           v_miss = np.array([])
           
        if s==4:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([10])
           v_miss = np.array([])
           
        if s==5:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([4,8])
           p_wrong = np.array([])
           v_miss = np.array([])
           
        if s==6:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           
           v_miss = np.array([4,8])
           
        if s==7:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([8])
           p_wrong = np.array([])
           
           v_miss = np.array([4])
           
        if s==8:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           
           v_miss = np.array([])
           
           sgen_inc = 1.7 # coefficient to increase/decrease installed generation capacity in the grid
           load_inc = 1.4   # coefficient to increase/decrease load capacity in the grid
           
        if s==9:
           se_case = "perso" 
           meas_case = mc_def
           
           v_wrong = np.array([])
           p_wrong = np.array([])
           
           v_miss = np.array([])
           
           sgen_inc = 0.8 
           load_inc = 1.2
       
        if meas_case == 1:
       
            v_bus = np.array([0,1,12,8])  # Set of buses with a V meter
            phi_bus = np.array([])        # Set of buses with a phasor meter
            p_bus = np.array([0])         # Set of buses with a P and Q injection meter
       
           
            p_line = np.array([0,10])     # Set of lines with a P and Q flow meter
            i_line = np.array([])         # Set of buses with an I flow meter

                
        if meas_case == 2:

            v_bus = np.array([0,1,12,4,8])
            phi_bus = np.array([])
            p_bus = np.array([0])
            
            i_line = np.array([0,10,6]) 
            p_line = np.array([0,10,2,6])  
           
        if meas_case == 3:

            v_bus = np.array([])
            phi_bus = np.array([])
            p_bus = np.array([0,1,12])

            
            p_line = np.array([0,10])  
            i_line = np.array([0,10])
            
        if meas_case == 4:

            v_bus = net.bus.index
            phi_bus = net.bus.index
            p_bus = np.array([])
            
            i_line = net.line.index 
            p_line = net.line.index  
           
           
        if meas_case == 5:
           
           v_bus = np.array([0,6,7,8,10,11])
           phi_bus = np.array([4,14])
           p_bus = np.array([])
           
           i_line = np.array([0,10])
           p_line = np.array([0,6,7,8,9,10]) 
           
        if meas_case == 6:
       
           v_bus = net.bus.index
           phi_bus = np.array([])
           p_bus = np.array([])
           
           i_line = net.line.index 
           p_line = net.line.index 
           
        if meas_case == 7:
       
           v_bus = net.bus.index
           phi_bus = np.array([3])
           p_bus = np.array([])
           
           i_line = net.line.index 
           p_line = net.line.index 
            
    if grid=='ober':
        
        if s==0:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==1:
            se_case = "bad" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==2:
            se_case = "good" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==3:
            se_case = "good" 
            meas_case = 2
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==4:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([39])
            
        if s==5:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([58,39,80])
            p_wrong = np.array([162,165])
            v_miss = np.array([])
            
        if s==6:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([58])
            p_wrong = np.array([])
            v_miss = np.array([34,39,80])
            
        if s==7:
           
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = 1.4
            load_inc = 0.65
            
        if s==8:
           
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = 2.5 
            load_inc = 1.
        
        if s==9:
           
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = .5 
            load_inc = 0.8
            
            
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

                v_bus = np.array([58,39])
                phi_bus = np.array([])
                p_bus = np.array([])

                
                p_line = np.array([])  
                i_line = np.array([])
                
    if grid=='ober2':
        
        if s==0:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==1:
            se_case = "bad" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==2:
            se_case = "good" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==3:
            se_case = "perso" 
            meas_case = 2
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([])
            
        if s==4:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            v_miss = np.array([39])
            
        if s==5:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([58,39,80])
            p_wrong = np.array([162,165])
            v_miss = np.array([])
            
        if s==6:
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([58])
            p_wrong = np.array([])
            v_miss = np.array([39,80])
            
        if s==7:           
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = 1.05 # coefficient of increased capacity in future
            load_inc = 1.95
            
        if s==8:          
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = 1.875 # coefficient of increased capacity in future
            load_inc = 3.
        
        if s==9:          
            se_case = "perso" 
            meas_case = 1
            
            v_wrong = np.array([])
            p_wrong = np.array([])
            
            v_miss = np.array([])
            
            sgen_inc = .375 # coefficient of increased capacity in future
            load_inc = 2.4
        

        if meas_case == 1:      
            v_bus = np.array([58,39,80,86,146,81,34,142,100,50, 82, 161,318,319,6,126,245,171,273,54,167,74,33,213,237,316])
            phi_bus = np.array([])
            p_bus = np.array([])
        
            
            p_line = np.array([162,165,185,81,60,171,70,64,62,193,18,46,5,11,122,28,45,94])  # line 12,13,14 have switches
            i_line = np.array([162,165,185,171,62,193,46,28])
        if meas_case == 2:

            v_bus = np.array([58,39,80,86,146,81,100,50, 318,319,6,126,245,171,273,167,33,237])
            phi_bus = np.array([])
            p_bus = np.array([])

            
            p_line = np.array([162,165,81,60,70,62,193,18,5,122,28,94])  # line 12,13,14 have switches
            i_line = np.array([162,165,62,193])


    
    """ Measurement Noise specification """

    if se_case == "perso":
    
        meas_noise = 0.02       # coefficient of power measurement noise
        v_noise = 0.01          # coefficient of voltage sensors noise
        i_noise = 0.01          # coefficient of current sensors noise
        zero_inj_coef = 0.001   # added weights for zero injection buses
        
        load_error = 0.2        # coef load error
        sgen_error = 0.15       # coef sgen error
    
        pm_noise = load_error/2 # 95% confidence
        sgen_noise = sgen_error/2
    
        
    if se_case == "bad":
        
        meas_noise = 0.05 
        v_noise = 0.03 
        i_noise = 0.03
        zero_inj_coef = 0.001
        
        load_error = 0.3 
        sgen_error = 0.2 
    
        pm_noise = load_error/2  
        sgen_noise = sgen_error/2
    
        
    if se_case == "good":
        
        meas_noise = 0.01
        v_noise = 0.005 
        i_noise = 0.005
        zero_inj_coef = 0.001
    
        load_error = 0.15 
        sgen_error = 0.1 
        
        pm_noise = load_error/2  
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
    
    
    
    """ Sampling load scenarios """
    
    if grid=='cigre':
        
        load_p = pd.DataFrame()
        load_q = pd.DataFrame()
        load_sgen = pd.DataFrame()
        
        if s<8:
            sgen_inc = 1.4 
            load_inc = 1.2
        
        for i in range(household_load_profile.size): 
            
            
            load_p[i]= pd.concat([net.load['p_mw'][:10].mul(household_load_profile[i]),net.load['p_mw'][10:].mul(industry_load_profile[i])]) * load_inc
            load_q[i]= pd.concat([net.load['q_mvar'][:10].mul(household_load_profile[i]),net.load['q_mvar'][10:].mul(industry_load_profile[i])]) * load_inc
            load_sgen[i]= net.sgen['p_mw'].mul(profile_day_sun[i]) * sgen_inc
            load_sgen.loc[8] = net.sgen['p_mw'][8] * profile_day_wind[i] *sgen_inc
        
    if grid=='ober':
        
        load_p = pd.DataFrame(index = net.load.index)
        load_q = pd.DataFrame(index = net.load.index)
        load_sgen = pd.DataFrame(index = net.sgen.index)
        
        if s<7:
            sgen_inc = 2. 
            load_inc = .5
        
        for i in range(household_load_profile.size): 
            
            
            load_p[i]=net.load['p_mw'].mul(household_load_profile[i]) * load_inc
            load_p[i][net.load[net.load['type']=="MV Load"].index]=\
                net.load['p_mw'][net.load[net.load['type']=="MV Load"].index].mul(industry_load_profile[i]) * load_inc
            load_q[i]=net.load['q_mvar'].mul(household_load_profile[i]) * load_inc
            load_q[i][net.load[net.load['type']=="MV Load"].index]=\
                net.load['q_mvar'][net.load[net.load['type']=="MV Load"].index].mul(industry_load_profile[i]) * load_inc
                
            load_sgen[i]= net.sgen['p_mw'].mul((4*profile_day_sun[i] + 6*profile_day_wind[i])/10) * sgen_inc
            
    if grid=='ober2':
        
        load_p = pd.DataFrame(index = net.load.index)
        load_q = pd.DataFrame(index = net.load.index)
        load_sgen = pd.DataFrame(index = net.sgen.index)
        
        if s<7:
            sgen_inc = 1.5 # coefficient of increased capacity in future
            load_inc = 1.5
        
        for i in range(household_load_profile.size): 
            
            
            load_p[i]=net.load['p_mw'].mul(household_load_profile[i]) * load_inc
            load_p[i][net.load[net.load['type']=="MV Load"].index]=\
                net.load['p_mw'][net.load[net.load['type']=="MV Load"].index].mul(industry_load_profile[i]) * load_inc
            load_q[i]=net.load['q_mvar'].mul(household_load_profile[i]) * load_inc
            load_q[i][net.load[net.load['type']=="MV Load"].index]=\
                net.load['q_mvar'][net.load[net.load['type']=="MV Load"].index].mul(industry_load_profile[i]) * load_inc
                
            load_sgen[i]= net.sgen['p_mw'].mul((4*profile_day_sun[i] + 6*profile_day_wind[i])/10) * sgen_inc
                 
    
    
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
    pd_mc_load_p = pd.DataFrame(mc_load_p, index = net.load.index)
    
    mc_load_sgen = np.reshape(mc_unroll_load_sgen,[load_sgen.shape[0],load_sgen.shape[1]*iteration])
    pd_mc_load_sgen = pd.DataFrame(mc_load_sgen, index = net.sgen.index)
    
    
    pd_mc_load_q = pd_mc_load_p.mul((net.load["q_mvar"]/net.load["p_mw"]),axis=0)




    """ Power Flow """
    
    # Initialization of dataframes to store pf results
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
    
    if grid=='cigre':
        pf_loading_traf=pd.DataFrame(index = net.trafo.index + net.line.index.size - 3,columns = ts)
    if grid=='ober':
        pf_loading_traf=pd.DataFrame(index = net.trafo.index,columns = ts)
    if grid=='ober2':
        pf_loading_traf=pd.DataFrame(index =  net.trafo.index,columns = ts)
        
        
        
    pf_pt=pd.DataFrame(columns = ts)
    pf_qt=pd.DataFrame(columns = ts)
    pf_it=pd.DataFrame(columns = ts)
    
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
        pf_loading_traf[t] = net.res_trafo["loading_percent"].values
        
        pf_pt[t] = net.res_trafo["p_hv_mw"]
        pf_qt[t] = net.res_trafo["q_hv_mvar"]
        pf_it[t] = net.res_trafo["i_hv_ka"]
        
    duration_pf = time.time() - start_pf
    
    
    # Initialization of input measurements arrays and  dataframes for State Estimation
    A_dss = np.array([])
    B_dss = np.array([])
    U_dss = np.array([])
    
    meas_vm = pd.DataFrame(index = net.bus.index, columns = ts)
    meas_va = pd.DataFrame(index = net.bus.index, columns = ts)
    meas_p = pd.DataFrame(index = net.bus.index, columns = ts)
    meas_q = pd.DataFrame(index = net.bus.index, columns = ts)
    pseudo_p = pd.DataFrame(data= 0,index = net.bus.index, columns = ts)
    pseudo_q = pd.DataFrame(data= 0,index = net.bus.index, columns = ts)
    
    meas_pl = pd.DataFrame(index = net.line.index, columns = ts)
    meas_ql = pd.DataFrame(index = net.line.index, columns = ts)
    meas_il = pd.DataFrame(index = net.line.index, columns = ts)
    
    

     # Setting angles to rad  
    pf_va_angle = pf_va
    pf_va *= (np.pi/180.)
     
    
    """ State Estimation """
 
    print("--- State Estimation ---")
           
    count_div = 0
    
    # Initialization of dataframes to store WLS algorithm and DSS model results
    succ = pd.DataFrame(columns = ts)
    se_vm=pd.DataFrame(index=net.bus.index,columns = ts)
    se_va=pd.DataFrame(index=net.bus.index,columns = ts)
    se_pl=pd.DataFrame(index=net.line.index,columns = ts)
    se_ql=pd.DataFrame(index=net.line.index,columns = ts)
    se_loading=pd.DataFrame(index=net.line.index,columns = ts)
    
    
    if grid=='cigre':
        nl = 14
        se_loading_traf=pd.DataFrame(index=net.trafo.index + net.line.index.size - 3,columns = ts)
        loading_pred=pd.DataFrame(index=np.arange(14),columns = ts)
    if grid=='ober':
        nl=68
        se_loading_traf=pd.DataFrame(index=net.trafo.index, columns = ts)
        loading_pred=pd.DataFrame(index=np.concatenate([net.line.index[:-5], net.line.index[-4:], net.trafo.index]),columns = ts)
    if grid=='ober2':
            nl=181
            se_loading_traf=pd.DataFrame(index=net.trafo.index, columns = ts)
            loading_pred=pd.DataFrame(index=np.concatenate([net.line.index, net.trafo.index]),columns = ts)
        
    pred_v=pd.DataFrame(index=net.bus.index,columns = ts)
    
    
    se_fail = []
    
    
    # Testing loop
    for t in progressBar(ts, prefix = 'Progress:', suffix = 'Complete', length = 50):

            
        """ Building input data for both WLS and DSS """
        
        # Line measurements
        
        # Power flow measurements
        for l in p_line:
            
            # Adding wrong measurements
            if any(p_wrong==l):
                p = pf_pl[t][l] + np.random.normal(0,3*meas_noise * np.abs(pf_pl[t][l]))
                q = pf_ql[t][l] + np.random.normal(0,3*meas_noise * np.abs(pf_ql[t][l]))
                
            else:
                p = pf_pl[t][l] + np.random.normal(0,meas_noise * np.abs(pf_pl[t][l]))
                q = pf_ql[t][l] + np.random.normal(0,meas_noise * np.abs(pf_ql[t][l]))
            
            meas_pl[t][l] = p
            meas_ql[t][l] = q
            
            # Flow measurements only for CIGRE otherwise very poor convergence rate
            if not (grid=='ober' or grid=='ober2'):
                pp.create_measurement(net, "q", "line", q , q * meas_noise, element=l, side="from")
                pp.create_measurement(net, "p", "line", p , p * meas_noise, element=l, side="from")
                
                
        # Current flow measurements
        for l in i_line:
            i = pf_il[t][l] + np.random.normal(0,i_noise * np.abs(pf_il[t][l]))
            
            meas_il[t][l] = i
           
        
            # Flow measurements only for CIGRE otherwise very poor convergence rate
            if not (grid=='ober' or grid=='ober2'):
                 pp.create_measurement(net, "i", "line", i , i * i_noise, element=l, side="from")
         

        # Bus measurements
        
        pm_bus= pd.DataFrame(data = {"p_mw": 0, "q_mvar": 0}, index= net.bus.index)
                 
        
        for i in net.bus.index:
            
            # Calculating pseudomeasurements
            load_index = net.load[net.load['bus']==i].index
            sgen_index = net.sgen[net.sgen['bus']==i].index
            
            
            for j in sgen_index:
                pm_bus["p_mw"][i] -= load_sgen[t%load_p.shape[1]][j] 
                pseudo_p[t][i] -= load_sgen[t%load_p.shape[1]][j] 
                
            for j in load_index:
                pm_bus["p_mw"][i] += load_p[t%load_p.shape[1]][j]
                pm_bus["q_mvar"][i] += load_q[t%load_p.shape[1]][j]
      
                pseudo_p[t][i] += load_p[t%load_p.shape[1]][j]
                pseudo_q[t][i] += load_q[t%load_p.shape[1]][j]
                
            if (pm_bus["p_mw"][i]!=0 and (not any(p_bus==i))):
                pp.create_measurement(net, "p", "bus", pm_bus["p_mw"][i], pm_noise*pm_bus["p_mw"][i], element=i)
            if (pm_bus["q_mvar"][i]!=0 and (not any(p_bus==i))):
                pp.create_measurement(net, "q", "bus", pm_bus["q_mvar"][i], pm_noise*pm_bus["q_mvar"][i], element=i) 
                
            
            # voltage measurements + wrong/missing meas.
            if any(v_bus==i):
                if any(v_wrong==i):
                    v = pf_vm[t][i] + np.random.normal(0,v_noise*3*np.abs(pf_vm[t][i]))
                else:
                    v = pf_vm[t][i] + np.random.normal(0,v_noise*np.abs(pf_vm[t][i]))
                    
                if any(v_miss==i):
                    v = pf_vm.loc[i].mean()
                    
                
                meas_vm[t][i] = v
                                     
                pp.create_measurement(net, "v", "bus", v, v  * v_noise, element=i)
                
                
            # Power injection measurements + wrong/missing meas.    
            if any(p_bus==i):
               
                p = pf_p[t][i] + np.random.normal(0,meas_noise * np.abs(pf_p[t][i]))
                q = pf_q[t][i] + np.random.normal(0,meas_noise * np.abs(pf_q[t][i]))
                
                meas_p[t][i] = p
                meas_q[t][i] = q                   
                pp.create_measurement(net, "p", "bus", p, p  * meas_noise, element=i)
                pp.create_measurement(net, "p", "bus", q, q  * meas_noise, element=i)
                
                
            # phasor measurements  
            if any(phi_bus==i):
                va = pf_va[t][i] + np.random.normal(0,v_noise * np.abs(pf_va[t][i]))
                
                meas_va[t][i] = va

             
        # Adding open lines as measurements to the WLS algorithm
        edge_ind=-1
        for i in net.switch.index:
             old_ind = edge_ind
             edge_ind = net.switch["element"][i]

             if edge_ind == old_ind:
                 if not (net.switch["closed"][i] and net.switch["closed"][i-1]):
                     pp.create_measurement(net, "p", "line", 0, 10e-3, element=edge_ind, side="from")
                     pp.create_measurement(net, "q", "line", 0, 10e-3, element=edge_ind, side="from")
   


        # Performing SE with WLS algorithm
        start_se = time.time()       
        success = est.estimate(net, algorithm=algorithm, init=init,estimator=estimator,zero_injection=zero_injection)        
        duration_se[s][t] = time.time() - start_se
        
        # Storing data
        V_est, delta_est = net.res_bus_est.vm_pu, net.res_bus_est.va_degree
        p_line_est,q_line_est = net.res_line_est.p_from_mw, net.res_line_est.q_from_mvar
        
        if success:
            se_vm[t] = V_est
            se_va[t] = delta_est
            se_pl[t] = p_line_est
            se_ql[t] = q_line_est
            se_loading[t] = net.res_line_est.loading_percent
            se_loading_traf[t] = net.res_trafo_est.loading_percent.values
            succ[t] = np.array([int(success)])
        
        else:
            count_div +=1
            se_vm[t] = 0.
            se_va[t] = 0.
            se_pl[t] = 0.
            se_ql[t] = 0.
            se_loading[t] = 0.
            se_loading_traf[t] = 0.
            se_fail.append(t)
            
            
        # Stopping loop if not enough convergence
        if (count_div > 15 and count_div > t * 0.6):
            break
        
        
        """ Building data for DSS model """
        nan = np.nan
       
     
        meas_bus = pd.DataFrame({"vm_pu": meas_vm[t], "va_rad":meas_va[t], "p_mw": meas_p[t], "q_mvar": meas_q[t]}, index = net.bus.index)
        
        edge_index = np.concatenate((net.line.index, net.trafo.index + net.line.index.size))     
        meas_edge = pd.DataFrame({"p_from_mw": meas_pl[t],  "q_from_mvar":meas_ql[t], "i_from_ka":meas_il[t]}, index = edge_index) 

        labels = pd.DataFrame({"vm_pu": pf_vm[t], "va_rad": pf_va[t]}, index = net.bus.index)
     
        pm_bus = pd.DataFrame()
        pm_bus["p_mw"] = pseudo_p[t]
        pm_bus["q_mvar"] = pseudo_q[t]

        meas_error = np.array([v_noise, v_noise,meas_noise , meas_noise, i_noise ,pm_noise, zero_inj_coef], dtype=float)
        
        # Process PandaPower data to DSS input
        start_dss = time.time()
        A,B,U = dssdata(net, pm_bus, meas_bus, meas_edge, meas_error, labels)
        A_flat = tf.cast(tf.expand_dims(A.flatten(),axis=0),dtype=tf.float32)
        B_flat = tf.cast(tf.expand_dims(B.flatten(),axis=0),dtype=tf.float32)
        U_flat = tf.cast(tf.expand_dims(U.flatten(),axis=0),dtype=tf.float32)
        dss_data = time.time()
        
        # Normalize data
        a_flat, b_flat, A0,B0, A = preprocess_data(A_flat,B_flat,U_flat,problem,grid)
        
        
        for m in range(num_rep):
            
            # Perform SE
            t_preproc = time.time()
            pred=models[m](a_flat,b_flat,U_flat,A0, training=False)
            t_pred=time.time()
            
            
            t_df[t][s] = t_pred-t_preproc
            
            # Un-normalize output
            y_pred = pred * problem.B_std[0:3:2]  + problem.B_mean[0:3:2] 
            y = tf.concat([y_pred[:,:,0:1], y_pred[:,:,1:] * ( 1. - B0[:,:,-1:])], axis=2)
  
            t_postproc = time.time()
        
            pred_v[t] = y[0,:,0]
            
            # Pass output in powrr flow equations to get line loading
            pf_pred, qf_pred, pt_pred, qt_pred, if_pred, it_pred, loading = get_pflow(y,A,problem,grid)
            loading_pred[t] = loading[0,:,0]
            
            
            # Plot error for specific time in bus and lines
            if any(np.array([2,10,21])==t):
                  
                  if grid=='ober':
                      x = np.arange(pred_v.index.size)
                      x2 = np.arange(loading_pred.index.size)
                      nt = 1
                      pf = pd.concat([pf_loading[t][:-5],pf_loading[t][-4:], pf_loading_traf[t]],axis=0)
                      se = pd.concat([se_loading[t][:-5],se_loading[t][-4:], se_loading_traf[t]],axis=0)
                  if grid=='cigre':
                      x = np.arange(pred_v.index.size)
                      x2 = np.arange(loading_pred.index.size)
                      nt = 2
                      pf = pd.concat([pf_loading[t][:12], pf_loading_traf[t]])
                      se = pd.concat([se_loading[t][:12], se_loading_traf[t]])
                  if grid=='cigre':
                        x = np.arange(pred_v.index.size)
                        x2 = np.arange(loading_pred.index.size)
                        nt = 2
                        pf = pd.concat([pf_loading[t][:12], pf_loading_traf[t]])
                        se = pd.concat([se_loading[t][:12], se_loading_traf[t]])
                 
                  w = 0.3
                 
                  fig = plt.figure(figsize=[5,3])
                  ax = fig.add_axes([0,0,1,1])
                  ax.bar(x-w, height = pf_vm[t], width=w, color='purple', align='center', label = 'PF')
                  ax.bar(x, height = y[0,:,0], width=w, color='salmon', align='center', label = 'DSS')                     
                  ax.bar(x+w, height = se_vm[t], width=w, color='teal', align='center', label = 'WLS')
                  ax.legend() 
                  ax.set_title("Voltage per bus at time "+str(t)+", for case study #"+str(s+1))
                  ax.set_ylabel("Voltage [p.u.]")
                  ax.set_xlabel("Bus index")
                  ax.set_ylim(0.95,1.05)
                 
                  fig = plt.figure(figsize=[5,3])
                  ax = fig.add_axes([0,0,1,1])
                  ax.bar(x2-w, height = pf, width=w, color='purple', align='center', label = 'PF')
                  ax.bar(x2, height = loading_pred[t], width=w, color='salmon', align='center', label = 'DSS')                     
                  ax.bar(x2+w, height = se, width=w, color='teal', align='center', label = 'WLS')
                  ax.set_title("Loading per line at time "+str(t)+", for case study #"+str(s+1))
                  ax.set_ylabel("Line loading [%]")
                  if (max(loading_pred[t])>100. or max(se)>100.):
                      ax.set_ylim(0.,100.)
                  ax.set_xlabel("Line index")
                  ax.legend()
                      
 
        
    # Plotting specific line or bus through time
    
    if grid=='cigre':
        b_list = np.array([1,3,8,6,7])
        l_list = np.array([0,1,2,6,8,10,11])
        tr_list = np.array([12])
        
    if grid=='ober':
        b_list = np.array([80,120,34,192,223,161])
        l_list = np.array([157,185,60,162,68])
        tr_list = np.array([114])
        
    if grid=='ober2':
            b_list = np.array([80,120,34,161,6,303,169,238])
            l_list = np.array([162,60,181,193,122,99,22])
            tr_list = np.array([194])
    
    for b in b_list:
    
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.plot(ts, pf_vm.loc[b], color='purple',linewidth=2.,label="PF")
        if any(v_bus==b):
            ax.plot(ts, meas_vm.loc[b], color='deepskyblue',linewidth=2.,label="Measurement")
        ax.plot(ts, pred_v.loc[b], color='salmon',linewidth=2.,label="DSS")        
        ax.plot(ts, se_vm.loc[b], color='teal',linewidth=2.,label="WLS")
        ax.legend() 
        ax.set_title("Voltage level at bus "+str(b)+" through a day, for case study #"+str(s+1))
        ax.set_ylabel("Voltage [p.u.]")
        ax.set_xlabel("Time [h]")
        ax.set_ylim(0.95,1.05)
        
        
    for l in l_list:
    
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.plot(ts, pf_loading.loc[l], color='purple',linewidth=2.,label="PF")
        if any(i_line==l):
            i_measurement = meas_il.loc[l]
            loading_measurement = i_measurement *100 /net.line.loc[l].max_i_ka
            ax.plot(ts, loading_measurement, color='deepskyblue',linewidth=2.,label="Measurement")
        ax.plot(ts, loading_pred.loc[l], color='salmon',linewidth=2.,label="DSS")        
        ax.plot(ts, se_loading.loc[l], color='teal',linewidth=2.,label="WLS")
        ax.legend() 
        ax.set_title("Loading in line "+str(l)+" through a day, for case study #"+str(s+1))
        ax.set_ylabel("Loading [%]]")
        ax.set_xlabel("Time [h]")
        if (max(loading_pred.loc[l])>100. or max(se_loading.loc[l])>100.):
            ax.set_ylim(0.,100.)
        ax.legend()
        
    for tr in tr_list:
    
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.plot(ts, pf_loading_traf.loc[tr], color='purple',linewidth=2.,label="PF")
        ax.plot(ts, loading_pred.loc[tr], color='salmon',linewidth=2.,label="DSS")        
        ax.plot(ts, se_loading_traf.loc[tr], color='teal',linewidth=2.,label="WLS")
        ax.legend() 
        ax.set_title("Loading in trafo "+str(tr)+" through a day, for case study #"+str(s+1))
        ax.set_ylabel("Loading [%]]")
        ax.set_xlabel("Time [h]")
        if (max(loading_pred.loc[tr])>100. or max(se_loading_traf.loc[tr])>100.):
            ax.set_ylim(0.,100.)
        ax.legend()

    

    """ Calculating and storing metrics for DSS """
    
    
    print("")   
    print("SET #"+str(s))        
     
         
    print(" ==== DSS === ")
    
    if grid=='cigre':

        pf = pd.concat([pf_loading[:12], pf_loading_traf])
        se = pd.concat([se_loading[:12], se_loading_traf])
        nt =2
    if grid=='ober':
        pf = pd.concat([pf_loading[:-5],pf_loading[-4:], pf_loading_traf],axis=0)
        se = pd.concat([se_loading[:-5],se_loading[-4:], se_loading_traf],axis=0)
        nt = 1
    if grid=='ober2':
        pf = pd.concat([pf_loading, pf_loading_traf],axis=0)
        se = pd.concat([se_loading, se_loading_traf],axis=0)
        nt = 2
        
    bus_vrmse[s] = pf_vm.subtract(pred_v).pow(2).mean(axis=1).pow(0.5)
    dss_metrics["RMSE V"][s] = bus_vrmse[s].mean()

    line_loadrmse[s] = pf.subtract(loading_pred).pow(2).mean(axis=1).pow(0.5)
    dss_metrics["RMSE load"][s] = line_loadrmse[s].mean()

    dss_metrics["RMSE load line only"][s] = line_loadrmse[s][:-nt].mean()

    print("RMSE V mean: "+str(dss_metrics["RMSE V"][s]))
    print("RMSE LOADING mean: "+str(dss_metrics["RMSE load"][s]))
    print("RMSE LOADING mean no traf: "+str(dss_metrics["RMSE load line only"][s]))
    print("")
    
    bus_vmae[s] = pf_vm.subtract(pred_v).abs().mean(axis=1)
    dss_metrics["MAE V"][s] = bus_vmae[s].mean()
    
    line_loadmae[s] = pf.subtract(loading_pred).abs().mean(axis=1)
    dss_metrics["MAE load"][s] = line_loadmae[s].mean()
    
    print("MAE V mean: "+str(dss_metrics["MAE V"][s]))
    print("MAE LOADING mean: "+str(dss_metrics["MAE load"][s]))
    print("")
    
    bus_vrmsep[s] = bus_vrmse[s].divide(pf_vm.mean(axis=1)).abs() * 100
    dss_metrics["RMSE% V"][s] = bus_vrmsep[s].mean()
    
    line_loadrmsep[s] = line_loadrmse[s].divide(pf).mean(axis=1).abs() * 100
    dss_metrics["RMSE% load"][s] = line_loadrmsep[s].mean()
    
    print("RMSE% V mean: "+str( dss_metrics["RMSE% V"][s]))
    print("RMSE% LOADING mean: "+str(dss_metrics["RMSE% load"][s]))
    
    print("")
    dss_metrics["Mean duration"][s] = t_df.loc[s].mean()
    print("Prediction time: "+str(dss_metrics["Mean duration"][s]))
    print("")
    
    
    """ Caclculating and storing metrics for WLS"""
    
    print(" ==== WLS === ")
    
    
    wls_metrics["Convergence rate"][s]=(1-count_div/(t+1))
   
    se_va *= (np.pi/180.)
    print('convergence rate = '+str(wls_metrics["Convergence rate"][s]))
    
    wls_metrics["Mean duration"][s] = duration_se[s].mean()
    print("average SE duration: "+ str(wls_metrics["Mean duration"][s]))
    
    if wls_metrics["Convergence rate"][s]>0.2:

     
        print("")
        print("--- Error measurement ---")
       
        
        pf_vm = pf_vm.drop(se_fail,axis=1)
        se_vm = se_vm.drop(se_fail,axis=1)
        pf_loading = pf_loading.drop(se_fail,axis=1)
        se_loading = se_loading.drop(se_fail,axis=1)
        
        delta_vm = pf_vm.subtract(se_vm)
        delta_loading = pf_loading.subtract(se_loading)
        delta_loading_traf = pf_loading_traf.subtract(se_loading_traf)
        
        relative_error_vm = delta_vm.divide(pf_vm).abs()
        relative_error_loading = delta_loading.divide(pf_loading).abs() 
        
        rmse_vm = delta_vm.pow(2).mean(axis=1).pow(0.5)
        rmse_loading = delta_loading.pow(2).mean(axis=1).pow(0.5)
        rmse_loading_traf = delta_loading_traf.pow(2).mean(axis=1).pow(0.5)
        
        rmse_vm_perc = rmse_vm.divide(pf_vm.mean(axis=1)).abs() * 100
        rmse_loading_perc = rmse_loading.divide(pf_loading.mean(axis=1)).abs() * 100
        rmse_loading_perc_traf = rmse_loading_traf.divide(pf_loading_traf.mean(axis=1)).abs() * 100
        
        step_mape_vm = relative_error_vm.mean(axis=1) * 100
        step_mape_loading = relative_error_loading.mean(axis=1) * 100
        
        step_mae_vm = delta_vm.abs().mean(axis=1)
        step_mae_loading = delta_loading.abs().mean(axis=1)
        step_mae_loading_traf = delta_loading_traf.abs().mean(axis=1)
   
        print("")
        print("--- RMSE ---")
        wls_metrics["RMSE V"][s] = rmse_vm.mean(axis=0)
        wls_metrics["RMSE load"][s] = pd.concat([rmse_loading[:12], rmse_loading_traf]).mean(axis=0)
        wls_metrics["RMSE load line only"][s] = rmse_loading.mean()

        wls_metrics["MAE V"][s] = step_mae_vm.mean(axis=0)
        wls_metrics["MAE load"][s] = pd.concat([step_mae_loading[:12], step_mae_loading_traf]).mean(axis=0)
        
        wls_metrics["RMSE% V"][s] = rmse_vm_perc.mean(axis=0)
        wls_metrics["RMSE% load"][s] = pd.concat([rmse_loading_perc[:12], rmse_loading_perc_traf]).mean(axis=0)
        
        print('vm rmse:' + str(wls_metrics["RMSE V"][s]))
        print('load rmse:' + str(wls_metrics["RMSE load"][s]))
        print('load rmse no traf:' + str(wls_metrics["RMSE load line only"][s] ))
        print('max rmse:' + str(rmse_vm.max(axis=0)))
        
        print("")
        print('vm mae:' + str(wls_metrics["MAE V"][s]))
        print('load mae:' + str(wls_metrics["MAE load"][s]))
        print('max mae:' + str(step_mae_vm.max(axis=0)))
        
        print("")
        print('vm rmse%:' + str(wls_metrics["RMSE% V"][s]))
        print('load rmse%:' + str(wls_metrics["RMSE% load"][s]))
        print('max rmse%:' + str(rmse_vm_perc.max(axis=0)))
        print("")

        
        """ Compare performances for one case study """

        if grid=='cigre':
            x = np.arange(pred_v.index.size)
            x2 = np.arange(loading_pred.index.size)
            serm = pd.concat([rmse_loading[:12], rmse_loading_traf])
            semae =  pd.concat([step_mae_loading[:12], step_mae_loading_traf])
        if grid=='ober':
            x = np.arange(pred_v.index.size)
            x2 = np.arange(loading_pred.index.size)
            serm = pd.concat([rmse_loading[:-5],rmse_loading[-4:], rmse_loading_traf],axis=0)
            semae = pd.concat([step_mae_loading[:-5],step_mae_loading[-4:], step_mae_loading_traf],axis=0)
        if grid=='ober2':
            x = np.arange(pred_v.index.size)
            x2 = np.arange(loading_pred.index.size)
            serm = pd.concat([rmse_loading, rmse_loading_traf],axis=0)
            semae = pd.concat([step_mae_loading, step_mae_loading_traf],axis=0)
       
        
        w = 0.4
       
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.bar(x-w/2, height = bus_vrmse[s], width=w, color='salmon', align='center', label = 'DSS')
        ax.bar(x+w/2, height = rmse_vm.values, width=w, color='teal', align='center', label = 'WLS')
        ax.set_title("RMSE of Voltage magnitude per bus, for case study #"+str(s+1))
        ax.set_ylabel("[-]")
        ax.set_xticks(x)
        ax.set_xlabel("Bus index")
     
        ax.axhline(y=0.01,linestyle='--',color='darkorange')
        ax.axhline(y=0.005,linestyle='--',color='green')
        ax.axhline(y=0.02,linestyle='--',color='red')
        ax.legend()
        
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.bar(x2-w/2, height = line_loadrmse[s], width=w, color='salmon', align='center', label = 'DSS')
        ax.bar(x2+w/2, height = serm, width=w, color='teal', align='center', label = 'WLS')
        ax.set_title("RMSE of line loading, for case study #"+str(s+1))
        if (max(line_loadrmse[s])>45. or max(serm)>45.):
            ax.set_ylim(0.,45.)
        ax.set_ylabel("[%]")
        ax.set_xlabel("Line index")        
        ax.legend()
   
        
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.bar(x-w/2, height = bus_vmae[s], width=w, color='salmon', align='center', label = 'DSS')
        ax.bar(x+w/2, height = step_mae_vm.values, width=w, color='teal', align='center', label = 'WLS')
        ax.set_title("MAE of Voltage magnitude per bus, for case study #"+str(s+1))
        ax.set_ylabel("[-]")
        ax.set_xlabel("Bus index")
        ax.set_xticks(x)
        ax.legend()
        ax.axhline(y=0.01,linestyle='--',color='darkorange')
        ax.axhline(y=0.005,linestyle='--',color='green')
        ax.axhline(y=0.02,linestyle='--',color='red')
        
        fig = plt.figure(figsize=[5,3])
        ax = fig.add_axes([0,0,1,1])
        ax.bar(x2-w/2, height = line_loadmae[s], width=w, color='salmon', align='center', label = 'DSS')
        ax.bar(x2+w/2, height = semae, width=w, color='teal', align='center', label = 'WLS')
        ax.set_title("MAE of line loading, for case study #"+str(s+1))
        ax.set_ylabel("[%]")
        if (max(line_loadmae[s])>45. or max(semae)>45.):
                ax.set_ylim(0.,45.)
        ax.set_xlabel("Line index")           
        ax.legend()

         
       
"Plot performance per case study"
         
x = np.arange(1,len(sets)+1)


w = 0.4

fig = plt.figure(figsize=[5,3])
ax = fig.add_axes([0,0,1,1])
ax.bar(x-w/2, height = dss_metrics["RMSE V"], width=w, color='salmon', align='center', label = 'DSS')
ax.bar(x+w/2, height = wls_metrics["RMSE V"], width=w, color='teal', align='center', label = 'WLS')
ax.set_title("RMSE of Voltage magnitude per case study")
ax.set_ylabel("[-]")
ax.set_xlabel("Case study")
ax.set_xticks(x)
ax.legend()

fig = plt.figure(figsize=[5,3])
ax = fig.add_axes([0,0,1,1])
ax.bar(x-w/2, height = dss_metrics["RMSE load"], width=w, color='salmon', align='center', label = 'DSS')
ax.bar(x+w/2, height = wls_metrics["RMSE load"], width=w, color='teal', align='center', label = 'WLS')
ax.set_title("RMSE of line loading per case study")
ax.set_ylabel("[%]")
ax.set_xlabel("Case study")
ax.set_xticks(x)
ax.legend()

fig = plt.figure(figsize=[5,3])
ax = fig.add_axes([0,0,1,1])
ax.bar(x-w/2, height = dss_metrics["Convergence rate"]*100, width=w, color='salmon', align='center', label = 'DSS')
ax.bar(x+w/2, height = wls_metrics["Convergence rate"]*100., width=w, color='teal', align='center', label = 'WLS')
ax.set_title("Convergence rate per case study")
ax.set_ylabel("[%]")
ax.set_xlabel("Case study")
ax.set_xticks(x)
ax.legend(loc='lower right') 

fig = plt.figure(figsize=[5,3])
ax = fig.add_axes([0,0,1,1])
ax.bar(x-w/2, height = dss_metrics["Mean duration"]*1000, width=w, color='salmon', align='center', label = 'DSS')
ax.bar(x+w/2, height = wls_metrics["Mean duration"]*1000, width=w, color='teal', align='center', label = 'WLS')
ax.set_title("Duration per case study")
ax.set_ylabel("[ms]")
ax.set_xlabel("Case study")
ax.set_xticks(x)
ax.legend(loc='lower right') 
            
wls_metrics.to_csv(path_or_buf="saved_metrics/wls_metrics_"+str(grid)+"_betterm_meas1.csv")
dss_metrics.to_csv(path_or_buf="saved_metrics/dss_metrics_"+str(grid)+"_betterm_meas1.csv")    
