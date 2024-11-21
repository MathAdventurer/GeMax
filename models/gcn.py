# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import dgl
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gcn_layers = nn.ModuleList()
        

        self.gcn_layers.append(GraphConv(in_features, hidden_features))
        

        for _ in range(num_layers - 2):
            self.gcn_layers.append(GraphConv(hidden_features, hidden_features))
        

        self.gcn_layers.append(GraphConv(hidden_features, out_features))
        

        self.readout = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
    
    def forward(self, g):
        h = g.ndata['attr']
        
        for i in range(self.num_layers):
            h = self.gcn_layers[i](g, h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        
        return self.readout(hg), h