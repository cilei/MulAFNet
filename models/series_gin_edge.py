import os, sys, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter
import math
import numpy as np



dtype = torch.float32


class SerGINE(nn.Module):
    def __init__(self, num_atom_layers=3, num_fg_layers=2, latent_dim=128,
                 atom_dim=101, fg_dim=73, bond_dim=11, fg_edge_dim=101,
                 atom2fg_reduce='mean', pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers
        self.num_fg_layers = num_fg_layers
        self.atom2fg_reduce = atom2fg_reduce

        # embedding
        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)
        self.bond_embedding = nn.ModuleList(
            [nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_edge_embedding = nn.ModuleList(
            [nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)]
        )

        # gnn
        self.atom_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_atom_layers)]
        )
        self.atom_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_fg_layers)]
        )
        self.fg_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)]
        )
        self.atom2fg_lin = nn.Linear(latent_dim, latent_dim)
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        global_dim = 128
        local_dim = 128

        self.flow_field_generator = nn.Sequential(
            nn.Linear(global_dim + local_dim, local_dim),  # 拼接后过一个线性层
            nn.ReLU(),
            nn.Linear(local_dim, local_dim)  # 输出与局部特征维度一致
        )

        self.output_network = nn.Sequential(
            nn.Linear(local_dim + global_dim, local_dim + global_dim),  # 拼接后的处理
            nn.ReLU(),
            nn.Linear(local_dim + global_dim, local_dim + global_dim)  # 最终输出的维度
        )

        

    def forward(self, data):
        atom_x, atom_edge_index, atom_edge_attr, atom_batch = data.x, data.edge_index, data.edge_attr, data.batch
        fg_x, fg_edge_index, fg_edge_attr, fg_batch = data.fg_x, data.fg_edge_index, data.fg_edge_attr, data.fg_x_batch
        atom_idx, fg_idx = data.atom2fg_index
        seqEncoderTensor = data.seqEncoderTensor

        # one-hot to vec
        atom_x = self.atom_embedding(atom_x)
        fg_x = self.fg_embedding(fg_x)

        # atom-level gnn
        for i in range(self.num_atom_layers):
            atom_x = self.atom_gin[i](atom_x, atom_edge_index, self.bond_embedding[i](atom_edge_attr))
            atom_x = self.atom_bn[i](atom_x)
            if i != self.num_atom_layers-1:
                atom_x = self.relu(atom_x)
            atom_x = self.dropout(atom_x)
        
        #671,128


        # atom-level to FG-level
        # atom2fg_x = scatter(atom_x[atom_idx], index=fg_idx, dim=0, dim_size=fg_x.size(0), reduce=self.atom2fg_reduce)
        # atom2fg_x = self.atom2fg_lin(atom2fg_x)
        # fg_x = fg_x + atom2fg_x

        #280, 128
        # print("fg_x.shape:")
        # print(fg_x.shape)        

        # fg-level gnn
        for i in range(self.num_fg_layers):
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, self.fg_edge_embedding[i](fg_edge_attr))
            fg_x = self.fg_bn[i](fg_x)
            if i != self.num_fg_layers-1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)
        
        # print(smiTokenizer.shape)

        # seqOutput = self.seqEncode.encode(torch.t(smiTokenizer.reshape(128,-1)))
        
        # print(seqOutput.shape)

        fg_graph = self.pool(fg_x, fg_batch)

        atom_graph = self.pool(atom_x, atom_batch)
       

        combined_features = torch.cat([atom_graph, fg_graph], dim=-1)
        flow_field = self.flow_field_generator(combined_features)
        refined_local_features = fg_graph + flow_field
        final_features = torch.cat([refined_local_features, atom_graph], dim=-1)
        output = self.output_network(final_features)

        return output
    
class SubGraphGINE(nn.Module):
    def __init__(self,num_fg_layers=2, latent_dim=128,
                 fg_dim=73,fg_edge_dim=101,
                 pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_fg_layers = num_fg_layers
        

        # embedding
        
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)

        self.fg_edge_embedding = nn.ModuleList(
            [nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)]
        )

        # gnn
        self.fg_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_fg_layers)]
        )
        self.fg_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)]
        )
        
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)




        

    def forward(self, data):
        fg_x, fg_edge_index, fg_edge_attr, fg_batch = data.fg_x, data.fg_edge_index, data.fg_edge_attr, data.fg_x_batch
        

        # one-hot to vec
        
        fg_x = self.fg_embedding(fg_x)

        
        #671,128      

        # fg-level gnn
        for i in range(self.num_fg_layers):
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, self.fg_edge_embedding[i](fg_edge_attr))
            fg_x = self.fg_bn[i](fg_x)
            if i != self.num_fg_layers-1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)


        fg_graph = self.pool(fg_x, fg_batch)

       
       
        return fg_graph
    
class AtomGraphGINE(nn.Module):
    def __init__(self, num_atom_layers=3, latent_dim=128,
                 atom_dim=101,bond_dim=11,
                pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers


        # embedding
        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        
        self.bond_embedding = nn.ModuleList(
            [nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)]
        )


        # gnn
        self.atom_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_atom_layers)]
        )
        self.atom_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)]
        )


        
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        

    def forward(self, data):
        atom_x, atom_edge_index, atom_edge_attr, atom_batch = data.x, data.edge_index, data.edge_attr, data.batch
       

        # one-hot to vec
        atom_x = self.atom_embedding(atom_x)
       

        # atom-level gnn
        for i in range(self.num_atom_layers):
            atom_x = self.atom_gin[i](atom_x, atom_edge_index, self.bond_embedding[i](atom_edge_attr))
            atom_x = self.atom_bn[i](atom_x)
            if i != self.num_atom_layers-1:
                atom_x = self.relu(atom_x)
            atom_x = self.dropout(atom_x)


        atom_graph = self.pool(atom_x, atom_batch)
       

        return atom_graph



