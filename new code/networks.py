import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.nn as nn_geo
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCN2Conv, FAConv, TAGConv, GINEConv, MessagePassing, GCNConv, ChebConv
from torch.nn import Linear, LeakyReLU
from torch_geometric.utils import degree

class gnn_dsse(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, nonlin = 'leaky_relu', main_param =0.1, K = 3, bias = True, dropout = 0., theta = None, shared_weights = True, cached = True, add_self_loops = True, normalize = True, model = 'gcn2'):
        super().__init__()
        self.channels = dim_feat
        self.main_param = main_param
        self.dim_out = dim_out
        self.K = K
        self.dropout = dropout
        self.bias = bias
        self.theta = theta
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = nn.LeakyReLU()
        else:
            raise Exception('invalid activation type')
        
        nn_layer = []
        if model == 'gcn2':
            for l in range(self.num_layers-1):
                hyperparameters = {"channels":self.channels,"alpha":self.main_param,"theta":self.theta,
                "shared_weights":self.shared_weights,
                "cached":self.cached,
                "normalize":self.normalize,
                "add_self_loops":self.add_self_loops}
                nn_layer.extend([(GCN2Conv(**hyperparameters), 'x, x_0, edge_index -> x'), self.nonlin])
        elif model == 'fagcn':
            for l in range(self.num_layers-1):
                hyperparameters = {"channels":self.channels,"eps":self.main_param,"dropout":self.dropout,
                "cached":self.cached,
                "normalize":self.normalize,
                "add_self_loops":self.add_self_loops}
                nn_layer.extend([(FAConv(**hyperparameters), 'x, x_0,  edge_index -> x'), self.nonlin])
        elif model == 'tagcn':
            for l in range(self.num_layers-1):
                hyperparameters = {"in_channels":self.channels, "out_channels":self.channels,
                "K":self.K,
                "bias":self.bias,
                "normalize":self.normalize}
                nn_layer.extend([(TAGConv(**hyperparameters), 'x, edge_index -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')
        
        nn_layer.extend([Linear(in_features=dim_feat, out_features=dim_dense)])
        nn_layer.extend([Linear(in_features=dim_dense, out_features=dim_out)])
        
        self.model = nn_geo.Sequential('x, x_0, edge_index', nn_layer)

    def forward(self,x, edge_index):
        x_0 = x
        return self.model(x, x_0, edge_index)

class GINE_DSSE(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim, nn = 'mlp', nonlin = 'leaky_relu', eps =0., train_eps = False, model = 'gine'):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat=dim_feat
        self.dim_dense = dim_dense
        self.eps = eps
        self.train_eps = train_eps
        self.edge_dim = edge_dim
        self.dim_hidden = dim_feat
        
        if nn == 'mlp':
            self.nn = Linear(in_features = self.dim_feat, out_features = self.dim_hidden)
        else:
            raise Exception('invalid nn type')
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU()
        else:
            raise Exception('invalid activation type')
        
        nn_layer = []
        if model == 'gine':
            for l in range(self.num_layers-1):
                hyperparameters = {"nn":self.nn, "eps":self.eps, "train_eps": self.train_eps, "edge_dim":self.edge_dim}
                nn_layer.extend([(GINEConv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')
        
        nn_layer.extend([Linear(in_features=self.dim_hidden, out_features=self.dim_dense)])
        nn_layer.extend([Linear(in_features=self.dim_dense, out_features=self.dim_out)])
        
        self.model = nn_geo.Sequential('x, edge_index,edge_attr', nn_layer)

    def forward(self,x, edge_index, edge_attr):
        return self.model(x, edge_index,edge_attr)


class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features

    """
    def __init__(self, dim_featn, dim_feate, dim_hid, dim_out):
        super().__init__(aggr='add')
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out

        # self.linear = nn.Linear(dim_featn, dim_out) 
        self.edge_aggr = nn.Sequential(
            nn.Linear(dim_featn*2 + dim_feate, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_out)
        )
        
    def message(self, x_i, x_j, edge_attr):
        '''
        x_j:        shape (N, dim_featn,)
        edge_attr:  shape (N, dim_feate,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, dim_featn,)
            edge_attr:  shape (N, num_edges, dim_feate,)
            
        output:
            out:        shape (N, num_nodes, dim_out,)
        '''
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP
        
        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation
        
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        
        return out


class MPN(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        self.convs = nn.ModuleList()


        for l in range(n_gnn_layers):
            if l==n_gnn_layers-1:
                self.convs.append(TAGConv(dim_hid, dim_out, K=K))
            else:
                self.convs.append(TAGConv(dim_hid, dim_hid, K=K))

    def is_directed(self, edge_index):
        'determine if a graph is directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                # [edge_attr, edge_attr],
                [edge_attr, torch.cat([-edge_attr[:,0:1], edge_attr[:,1:2], -edge_attr[:,2:3], edge_attr[:,3:]],dim=1)],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, x, edge_index, edge_attr):
        edge_features = edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
class SkipMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added skip connection
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        self.convs = nn.ModuleList()

        for l in range(n_gnn_layers):
            if l==n_gnn_layers-1:
                self.convs.append(TAGConv(dim_hid, dim_out, K=K))
            else:
                self.convs.append(TAGConv(dim_hid, dim_hid, K=K))

    def is_directed(self, edge_index):
        'determine if a graph is directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                # [edge_attr, edge_attr],
                [edge_attr, torch.cat([-edge_attr[:,0:1], edge_attr[:,1:2], -edge_attr[:,2:3], edge_attr[:,3:]],dim=1)],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, x, edge_index, edge_attr):
        input_x = x # problem if there is inplace operation on x, so pay attention
        edge_features = edge_attr
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        # skip connection
        x = input_x + x
        
        return x
    
class PFN(nn.Module):
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate, L):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.L = L
        self.mpns = nn.ModuleList()
        
        for l in range(self.L):
            if l==self.L-1:
                self.mpns.append(MPN(self.dim_featn, self.dim_feate, self.dim_out, self.dim_hid, self.n_gnn_layers, self.K, self.dropout_rate))
            else:
                self.mpns.append(MPN(self.dim_featn, self.dim_feate, self.dim_featn, self.dim_hid, self.n_gnn_layers, self.K, self.dropout_rate))
        
    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.mpns)):
            x = self.mpns[i](x, edge_index, edge_attr)
            
        return x
    
class SkipPFN(nn.Module):
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate, L):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.L = L
        self.mpns = nn.ModuleList()
        
        for l in range(self.L):
            if l==self.L-1:
                self.mpns.append(MPN(self.dim_featn, self.dim_feate, self.dim_out, self.dim_hid, self.n_gnn_layers, self.K, self.dropout_rate))
            else:
                self.mpns.append(SkipMPN(self.dim_featn, self.dim_feate, self.dim_featn, self.dim_hid, self.n_gnn_layers, self.K, self.dropout_rate))
        
    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.mpns)):
            x = self.mpns[i](x, edge_index, edge_attr)
            
        return x

class MaskEmbdMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added embedding for mask
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(dim_hid, dim_out, K=K))
        else:
            self.convs.append(TAGConv(dim_hid, dim_hid, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(dim_hid, dim_hid, K=K))
            
        self.convs.append(TAGConv(dim_hid, dim_out, K=K))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(dim_featn, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_featn)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.dim_featn * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.dim_featn] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.dim_featn:]# last few dimensions: mask.
        
        x = self.mask_embd(mask) + x
        
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x


class MultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        else:
            self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))
            
        # self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_out))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x


class MaskEmbdMultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        else:
            self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))
            
        # self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_out))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(dim_featn, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_featn)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed. 
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.dim_featn * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.dim_featn] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.dim_featn:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x
    
    
class MaskEmbdMultiMPN_NoMP(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            # self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        else:
            # self.layers.append(EdgeAggregation(dim_featn, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))

        for l in range(n_gnn_layers-2):
            # self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_hid))
            self.layers.append(TAGConv(dim_hid, dim_hid, K=K))
            
        # self.layers.append(TAGConv(dim_hid, dim_out, K=K))
        self.layers.append(EdgeAggregation(dim_hid, dim_feate, dim_hid, dim_out))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(dim_featn, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_featn)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.dim_featn * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.dim_featn] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.dim_featn:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x

class WrappedMultiConv(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, K, **kwargs):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(ChebConv(in_channels, out_channels, K, normalization=None, **kwargs))
        
    def forward(self, x, edge_index_list, edge_weights_list):
        out = 0.
        for i in range(self.num_convs):
            edge_index = edge_index_list[i]
            edge_weights = edge_weights_list[i]
            out += self.convs[i](x, edge_index, edge_weights)

        return out

class MultiConvNet(nn.Module):
    """Wrapped Message Passing Network
        - No Message Passing to aggregate edge features into node features
        - Multi-level parallel Conv layers for different edge features
    """
    def __init__(self, dim_featn, dim_feate, dim_out, dim_hid, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.dim_featn = dim_featn
        assert dim_feate == 5
        dim_feate = dim_feate - 3 # should be 2, only these two are meaningful
        self.dim_feate = dim_feate
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_trans = nn.Sequential(
            nn.Linear(dim_feate, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_feate)
        )
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(WrappedMultiConv(dim_feate, dim_featn, dim_out, K=K))
        else:
            self.convs.append(WrappedMultiConv(dim_feate, dim_featn, dim_hid, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(WrappedMultiConv(dim_feate, dim_hid, dim_hid, K=K))
            
        self.convs.append(WrappedMultiConv(dim_feate, dim_hid, dim_out, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.dim_featn * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.dim_featn] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.dim_featn:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        edge_features = edge_features[:, :2] + self.edge_trans(edge_features[:, :2]) # only take the first two meaningful features
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, 
                            edge_index_list=[edge_index]*self.dim_feate,
                            edge_weights_list=[edge_features[:,i] for i in range(self.dim_feate)])
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, 
                            edge_index_list=[edge_index]*self.dim_feate,
                            edge_weights_list=[edge_features[:,i] for i in range(self.dim_feate)])
        
        return x
    
    
