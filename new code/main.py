import pandas as pd
import pandapower as pp
import numpy as np
import torch
from networks import gnn_dsse
from data import data_from_pickles, get_pflow, gsp_wls
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random
import wandb
from torchmetrics.regression import MeanAbsoluteError


hyperparameters = {
    'dim_feat': 8,
    'dim_dense': 4,
    'num_layers': 6,
    'main_param': 0.1 # alpha for gcn2, eps for fagcn, none for tagcn
}

reg_coefs = {
    'mu_v': torch.tensor(10),
    'mu_theta': torch.tensor(10),
    'lam_v': torch.tensor(1e10),
    'lam_th': torch.tensor(1e8),
    'lam_load': torch.tensor(1e7)
}

grid = 'cigre14'
folder = 'data/'+grid+'/'
model = 'tagcn'

if grid == 'cigre14':
    V_n = 20.
    V_hv = 110.
if (grid == 'ober' or grid == 'ober2'):
    V_n = 20.
    V_hv = 110.
if grid == 'lv':
    V_n = 0.416
    V_hv = 11.


tagcn = gnn_dsse(dim_feat = hyperparameters['dim_feat'], dim_dense = hyperparameters['dim_dense'], dim_out = 2, num_layers = hyperparameters['num_layers'],main_param = hyperparameters['main_param'], cached = False, model=model)


meas_v = np.array([0,1,2,6,8])
phase_shift = True
dataset, x_mean, x_std = data_from_pickles(folder,grid, meas_v, phase_shift)

num_nodes = torch._shape_as_tensor(dataset[0].x)[0]
batch_size = 64

split_coef = 0.9
train_dataset = dataset[0:int(split_coef*len(dataset))]
test_dataset = dataset[int(split_coef*len(dataset)):]

# Take a look at the training versus test graphs
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tagcn_dev = tagcn.to(device)

lr = 0.001

optimizer = torch.optim.AdamW(tagcn_dev.parameters(), lr)

epochs=600

train_list = []
metric_mae = []
metric_loading_mae = []
metric = []
metric_loading = []

wandb.init(
    # set the wandb project where this run will be logged
    project="gsp-wls",
    name="testing_tagcn_training",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": model,
    "dataset": grid,
    "epochs": epochs,
    "reg_coefs": reg_coefs,
    "Hyperparameters": hyperparameters
    }
)

for epoch in range(epochs):

    tagcn_dev.train()

    train_loss = 0
    for data in train_loader:

        num_samples = data.batch[-1]+1
        optimizer.zero_grad()

        out = tagcn_dev(data.x, data.edge_index)
        loss = gsp_wls(input=data.x, output= out, x_mean=x_mean, x_std=x_std, edge_index=data.edge_index, reg_coefs = reg_coefs, grid=grid,num_samples=num_samples)

        loss.backward()
        train_loss += loss
        optimizer.step()
            
    train_list.append(float((train_loss[0][0]/len(train_loader)).detach().float().numpy()))
    wandb.log({"TAGCN training loss": train_list[epoch]})

    tagcn_dev.eval()
    

    pred = 0
    pred_loading = 0
    pred_mae = 0
    pred_loading_mae = 0
    
    for data in test_loader:  # Iterate in batches over the test dataset.
    
        num_samples = data.batch[-1]+1
        out = tagcn_dev(data.x, data.edge_index)  
        out = torch.concat([out[:,0:1]*x_std[:1] + x_mean[:1], out[:,1:]], axis=1)
        pred += torch.sqrt(F.mse_loss(out, data.y))/V_n
        mae = MeanAbsoluteError()
        pred_mae += mae(out, data.y) / V_n
        
        true_loading = get_pflow(data.y, grid, edge_index=data.edge_index, num_samples = num_samples)[0]
        out_loading = get_pflow(out, grid, edge_index=data.edge_index, num_samples = num_samples)[0]

        pred_loading_mae += mae(out_loading, true_loading)
        pred_loading += torch.sqrt(F.mse_loss(out_loading, true_loading))
    metric.append(float((pred/len(test_loader)).detach().float().numpy()))
    
    metric_loading.append(float((pred_loading/len(test_loader)).detach().float().numpy()))
    metric_mae.append(float((pred_mae/len(test_loader)).detach().float().numpy()))
    metric_loading_mae.append(float((pred_loading_mae/len(test_loader)).detach().float().numpy()))

    
    wandb.log({"TAGCN voltage RMSE": metric[epoch], "TAGCN loading RMSE": metric_loading[epoch], "TAGCN voltage MAE": metric_mae[epoch], "TAGCN loading MAE": metric_loading_mae[epoch]})
    




