"""
This script trains a GNN (Graph Neural Network) model for power system state estimation using the GSP-WLS (Graph Signal Processing - Weighted Least Squares) algorithm. It uses the PyTorch library for deep learning and the Pandapower library for power flow calculations.

The script imports various modules and defines the necessary hyperparameters for the GNN model. It then loads the training and test datasets, initializes the GNN model, and sets up the optimizer for training.

The training loop iterates over the specified number of epochs and performs forward and backward passes through the model using mini-batches of the training dataset. The GSP-WLS loss function is calculated and used to update the model parameters.

After each epoch, the model is evaluated on the test dataset to compute various performance metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) for voltage magnitudes, voltage angles, line loadings, and transformer loadings.

"""
import pandas as pd
import pandapower as pp
import numpy as np
import torch
from networks import gnn_dsse, GINE_DSSE,GAT_DSSE, MPN, SkipMPN, PFN, SkipPFN
from data import data_from_pickles, get_pflow, gsp_wls, gsp_wls_edge
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random
from loadsampling import progressBar

from torchmetrics.regression import MeanAbsoluteError
import torch.optim as optim


# choose data grid to run from data folder
# data created in data.py in pickle files
case = 'cigre14'
folder = 'data/'+case+'/'



# Setting up
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
epochs=600

# get number of features and values per measurements
# To keep fixed for now - linked to data
phase_shift = True
num_nfeat = 8
num_efeat = 6
num_nmeas = 4
num_emeas = 2

# Set the measurement indices for each grid
if 'cigre' in case:
    meas_v = np.array([0,1,12,7,11,14])
    meas_pflow = np.array([0,10])
else:
    meas_v = np.array([35,16,52,47,6,48,59,27,37,56])
    meas_pflow = np.array([40,43,11,21,54,57])

# get data input
dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pflow)

random.shuffle(dataset)

split_coef = 0.9
train_dataset = dataset[0:int(split_coef*len(dataset))]
test_dataset = dataset[int(split_coef*len(dataset)):]

# Take a look at the training versus test graphs
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# dim_hid, gnn_layers, heads, K, dropout and L to tune as wanted
hyperparameters = {
    'dim_nodes': 8, # V, Theta, P, Q and Covs
    'dim_lines': 6, # P, Q and their Cov,B,G
    'dim_out': 2, # V, Theta
    'dim_hid': 32, 
    'gnn_layers': 8,
    'heads': 1,
    'K': 2,
    'dropout_rate': 0.3,
    'L': 5
}

# Setting up GNN model
model_name = 'gat'
model = GAT_DSSE(dim_feat= hyperparameters['dim_nodes'], dim_dense=hyperparameters['dim_hid'], dim_out=hyperparameters['dim_out'], heads=hyperparameters['heads'], num_layers=hyperparameters['gnn_layers'], edge_dim=hyperparameters['dim_lines'])

# model = SkipPFN(dim_featn=hyperparameters['dim_nodes'], dim_feate=hyperparameters['dim_lines'], dim_out=hyperparameters['dim_out'], dim_hid=hyperparameters['dim_hid'], n_gnn_layers=hyperparameters['gnn_layers'], K=hyperparameters['K'], dropout_rate=hyperparameters['dropout_rate'], L=hyperparameters['L'])

# Generate the optimizers.
lr = 3e-3
optimizer = getattr(optim, 'Adamax')(model.parameters(), lr=lr)

# in case you want to resume a model training
load_model = False
model_path = ''

if load_model:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# coefficients for training loss, to tune to find good training conditions/balance
mu_v = 1e-1
reg_coefs = {
'mu_v': mu_v,
'mu_theta': mu_v,
'lam_v': 1e-4,
'lam_p': 1e-8,
'lam_pf': 1e-6,
'lam_reg': 1e2
}

train_list = []
thresh = 0

# Initialize lists for testing values
rmse_v_list = []
mae_v_list = []
rmse_th_list = []
mae_th_list = []
rmse_loading_list = []
mae_loading_list = []
rmse_loading_trafos_list = []
mae_loading_trafos_list = []
prop_std_v_list = []
prop_std_th_list = []


# Training of the model.
for epoch in progressBar(range(epochs), prefix = 'Progress:', suffix = 'Complete', length = 50):
    train_loss = 0
    model.train()
    for data in train_loader:
        num_samples = data.batch[-1]+1
        
        optimizer.zero_grad()   
        out = model(data.x[:,:num_nfeat], data.edge_index, data.edge_attr[:,:num_efeat])
        
        loss = gsp_wls_edge(input=data.x[:,:num_nfeat], edge_input=data.edge_attr[:,:num_efeat], output= out, x_mean=x_mean, x_std=x_std, edge_mean = pflow_mean, edge_std = pflow_std, edge_index=data.edge_index, reg_coefs = reg_coefs,num_samples=num_samples, node_param=data.x[:,num_nfeat:], edge_param = data.edge_attr[:,num_efeat:])
        
        loss.backward()
        optimizer.step()
        train_loss += loss
        
        
    train_list.append(float((train_loss/len(train_loader)).detach().float().numpy()))
    
    # dynamic weights for varying training condition of the model
    # if epoch>thresh+10:
    #     if train_list[epoch]*100<train_list[thresh]:
    
    #         reg_coefs = {
    #             'mu_v': mu_v,
    #             'mu_theta': mu_v,
    #             'lam_v': reg_coefs['lam_v']*10,
    #             'lam_p': reg_coefs['lam_p']*10,
    #             'lam_pf': reg_coefs['lam_pf']*10,
    #             'lam_reg': reg_coefs['lam_reg']*10
    #             }
    #         thresh = epoch
    
    # Validation of the model
    
    model.eval()
    pred = 0
    pred_mae = 0
    pred_loading_lines = 0
    pred_loading_lines_mae = 0
    pred_loading_trafos = 0
    pred_loading_trafos_mae = 0
    pred_th = 0
    pred_mae_th = 0
    prop_std_v =0
    prop_std_th =0
    
    with torch.no_grad():
        for data in test_loader:

            num_samples = data.batch[-1]+1
            
            out = model(data.x[:,:num_nfeat], data.edge_index, data.edge_attr[:,:num_efeat])  
            out = torch.concat([out[:,0:1]*x_std[:1] + x_mean[:1], out[:,1:]], axis=1)
            out[:,1:] *= (1.-data.x[:,9:10])
            
            pred += torch.sqrt(F.mse_loss(out[:,:1], data.y[:,:1]))
            pred_th += torch.sqrt(F.mse_loss(out[:,1:], data.y[:,1:]))
            
            mae = MeanAbsoluteError()
            pred_mae += mae(out[:,:1], data.y[:,:1])
            pred_mae_th += mae(out[:,1:], data.y[:,1:])
            
            true_loading_lines, true_loading_trafos = get_pflow(data.y, data.edge_index, node_param = data.x[:,num_nfeat:], edge_param=data.edge_attr[:,num_efeat:])[0:2]
            out_loading_lines, out_loading_trafos = get_pflow(out, data.edge_index, node_param = data.x[:,num_nfeat:], edge_param=data.edge_attr[:,num_efeat:])[0:2]
            
            true_loading_lines2 = true_loading_lines[true_loading_lines.nonzero()]
            out_loading_lines = out_loading_lines[true_loading_lines.nonzero()]
            
            true_loading_trafos2 = true_loading_trafos[true_loading_trafos.nonzero()]
            out_loading_trafos = out_loading_trafos[true_loading_trafos.nonzero()]

            pred_loading_lines_mae += mae(out_loading_lines, true_loading_lines2)
            pred_loading_lines += torch.sqrt(F.mse_loss(out_loading_lines, true_loading_lines2))
            
            pred_loading_trafos += torch.sqrt(F.mse_loss(out_loading_trafos, true_loading_trafos2))
            pred_loading_trafos_mae += mae(out_loading_trafos, true_loading_trafos2)
            
            prop_std_v += ((out.std(axis=0)/data.y.std(axis=0))*100)[0]
            prop_std_th += ((out.std(axis=0)/data.y.std(axis=0))*100)[1]
            
            

    rmse_v=float((pred/len(test_loader)).detach().float().numpy())
    mae_v=float((pred_mae/len(test_loader)).detach().float().numpy())
    rmse_th=float((pred_th/len(test_loader)).detach().float().numpy())
    mae_th=float((pred_mae_th/len(test_loader)).detach().float().numpy())
    
    rmse_loading=float((pred_loading_lines/len(test_loader)).detach().float().numpy())   
    mae_loading=float((pred_loading_lines_mae/len(test_loader)).detach().float().numpy())
    rmse_loading_trafos=float((pred_loading_trafos/len(test_loader)).detach().float().numpy())   
    mae_loading_trafos=float((pred_loading_trafos_mae/len(test_loader)).detach().float().numpy())
    
    prop_std_v = float((prop_std_v/len(test_loader)).detach().float().numpy())
    prop_std_th = float((prop_std_th/len(test_loader)).detach().float().numpy())
    
    # Append values to lists
    rmse_v_list.append(rmse_v)
    mae_v_list.append(mae_v)
    rmse_th_list.append(rmse_th)
    mae_th_list.append(mae_th)
    rmse_loading_list.append(rmse_loading)
    mae_loading_list.append(mae_loading)
    rmse_loading_trafos_list.append(rmse_loading_trafos)
    mae_loading_trafos_list.append(mae_loading_trafos)
    prop_std_v_list.append(prop_std_v)
    prop_std_th_list.append(prop_std_th)


    
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'MAE V magn.': mae_v,
            'MAE line loading': mae_loading
            }, f'{model_name}.pt')



