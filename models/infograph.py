# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
import dgl

class InfoGraph(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.5):
        super(InfoGraph, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.gin_layers.append(GINConv(
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_features),
            ), learn_eps=True
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gin_layers.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features),
                    nn.ReLU(),
                    nn.Linear(hidden_features, hidden_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_features),
                ), learn_eps=True
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        
        # Output layer
        self.gin_layers.append(GINConv(
            nn.Sequential(
                nn.Linear(hidden_features, out_features),
                nn.ReLU(),
                nn.BatchNorm1d(out_features),
            ), learn_eps=True
        ))
        self.batch_norms.append(nn.BatchNorm1d(out_features))
        
        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
    
    def forward(self, g):
        h = g.ndata['attr']
        
        for i in range(self.num_layers):
            h = self.gin_layers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        
        return self.readout(hg), h