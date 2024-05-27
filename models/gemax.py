# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import dgl

def loss_entropy(Z, theta, phi, A_set):
    batch_size = Z.batch_size
    loss_entropy = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    start_idx = 0
    for j in range(batch_size):
        num_nodes = Z.batch_num_nodes()[j]
        Z_j = Z.ndata['h'][start_idx:start_idx+num_nodes]
        start_idx += num_nodes
        a_j = A_set[j][:num_nodes]  
        P_i = torch.sigmoid(torch.matmul(Z_j, theta.t()))
        
        if P_i.size(0) != a_j.size(0):  
            raise ValueError(f"Size mismatch: P_i has size {P_i.size(0)}, a_j has size {a_j.size(0)}")
        
        a_j_expanded = a_j.unsqueeze(1).expand_as(P_i)
        loss_entropy = loss_entropy - torch.sum(P_i * torch.log(a_j_expanded))
    
    return loss_entropy

def loss_orthogonal(Z, phi):
    batch_size = Z.batch_size
    loss_orth = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    start_idx = 0
    for j in range(batch_size):
        num_nodes = Z.batch_num_nodes()[j]
        Z_j = Z.ndata['h'][start_idx:start_idx+num_nodes]
        start_idx += num_nodes
        n_j = Z_j.size(0)
        M_j = torch.eye(n_j, device=Z.device) - Z.adjacency_matrix().to_dense()[:n_j, :n_j]
        term = torch.matmul(Z_j, Z_j.t()) - torch.eye(n_j, device=Z.device)
        loss_orth = loss_orth + torch.norm(M_j * term, 'fro')**2  
    
    return loss_orth

def loss_sub_vertex_packing(Z, A_set, theta, phi):
    batch_size = Z.batch_size
    loss_svp = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    start_idx = 0
    for j in range(batch_size):
        num_nodes = Z.batch_num_nodes()[j]
        Z_j = Z.ndata['h'][start_idx:start_idx+num_nodes]
        start_idx += num_nodes
        a_j = A_set[j][:num_nodes]  
        D_a_j = torch.diag(a_j)
        term = torch.matmul(torch.matmul(D_a_j, Z_j), Z_j.t()) - torch.matmul(D_a_j, D_a_j)
        loss_svp = loss_svp + torch.norm(term, 'fro')**2  
    
    return loss_svp

def objective_J1(Z, theta, phi, A_set, mu, gamma):
    loss_Hk = loss_entropy(Z, theta, phi, A_set)
    loss_orth = loss_orthogonal(Z, phi)
    loss_svp = loss_sub_vertex_packing(Z, A_set, theta, phi)
    
    J1 = loss_Hk - mu * loss_orth - gamma * loss_svp
    
    return J1

def objective_J2(Z, theta, phi, A_set, gamma):
    loss_Hk = loss_entropy(Z, theta, phi, A_set)
    loss_svp = loss_sub_vertex_packing(Z, A_set, theta, phi)
    
    J2 = loss_Hk + gamma * loss_svp
    
    return J2