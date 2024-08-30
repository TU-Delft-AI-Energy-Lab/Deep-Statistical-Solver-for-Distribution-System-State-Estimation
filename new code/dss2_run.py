import pandas as pd
import pandapower as pp
import numpy as np
import torch
from networks import gnn_dsse, GINE_DSSE, MPN, SkipMPN, PFN, SkipPFN
from data import data_from_pickles, get_pflow, gsp_wls, gsp_wls_edge
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random
from loadsampling import progressBar

from torchmetrics.regression import MeanAbsoluteError
import wandb
import optuna
from optuna.trial import TrialState
import torch.optim as optim


# choose data grid to run from data folder
# data created in data.py in pickle files
folder = 'data/cigre14/'

# choose a GNN model, current using SkipPFN from PowerFlowNet
model_name = 'skip_pfn'

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
epochs=900

# get number of features and measurements
phase_shift = True
num_nfeat = 8
num_efeat = 6
num_nmeas = 4
num_emeas = 2

# get data input and training
dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(folder, num_nfeat, num_efeat, num_nmeas, num_emeas)

random.shuffle(dataset)

split_coef = 0.9
train_dataset = dataset[0:int(split_coef*len(dataset))]
test_dataset = dataset[int(split_coef*len(dataset)):]

# Take a look at the training versus test graphs: gsp_wls_edg
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hyperparameters = {
    'dim_nodes': 8, # V, Theta, P, Q and Covs
    'dim_lines': 6, # P, Q and their Cov,B,G
    'dim_out': 2, # V, Theta
    'dim_hid': 64, 
    'gnn_layers': 3,
    'K': 2,
    'dropout_rate': 0.3,
    'L': 5
}



model = SkipPFN(dim_featn=hyperparameters['dim_nodes'], dim_feate=hyperparameters['dim_lines'], dim_out=hyperparameters['dim_out'], dim_hid=hyperparameters['dim_hid'], n_gnn_layers=hyperparameters['gnn_layers'], K=hyperparameters['K'], dropout_rate=hyperparameters['dropout_rate'], L=hyperparameters['L'])

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

# coefficients for training loss, to tune to find good training conditions
mu_v = 1e-1
reg_coefs = {
'mu_v': mu_v,
'mu_theta': mu_v,
'lam_v': 1e-4,
'lam_p': 1e-8,
'lam_pf': 1e-6,
'lam_reg': 1e2
}

num_train = 300

""" Initialize WandB if needed """
# id = np.random.randint(0,200)
# if load_model:
#     wandb.init(project="GSP-WLS",
#         name=f"{model_name}_training_resume",
#         id=str(id))
# else:
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="GSP-DSSE-multigraph",
#         name=f"{model_name}_training",
#         id=str(id),
#         # track hyperparameters and run metadata
#         config={
#         "learning_rate": lr,
#         "epochs": epochs,
#         "reg_coefs": reg_coefs,
#         "Hyperparameters": hyperparameters
#         }
#     )


train_list = []
thresh = 0
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
    
    if epoch>thresh+10:
        if train_list[epoch]*100<train_list[thresh]:
    
            reg_coefs = {
                'mu_v': mu_v,
                'mu_theta': mu_v,
                'lam_v': reg_coefs['lam_v']*10,
                'lam_p': reg_coefs['lam_p']*10,
                'lam_pf': reg_coefs['lam_pf']*10,
                'lam_reg': reg_coefs['lam_reg']*10
                }
            thresh = epoch
    
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
    

    # wandb.log({"training loss": train_list[epoch],"voltage magn. RMSE": rmse_v, "voltage magn. MAE": mae_v, "voltage angle RMSE": rmse_th, "voltage angle MAE": mae_th, "loading lines RMSE": rmse_loading, "loading lines MAE": mae_loading, "loading trafos RMSE": rmse_loading_trafos, "loading trafos MAE": mae_loading_trafos, "Voltage magn. proportional std [%]": prop_std_v, "Voltage angle proportional std [%]": prop_std_th})
    
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'MAE V magn.': mae_v,
            'MAE line loading': mae_loading
            }, f'{model_name}_training_new_ep{epochs*num_train+epochs}.pt') # skippfn_multigrid_training_new_ep900.pt



