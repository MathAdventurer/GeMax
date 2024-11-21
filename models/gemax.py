# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import dgl

def compute_P_i(g_i, z_i):
    distances = torch.norm(z_i - g_i.unsqueeze(0), dim=1) ** 2
    P_i = torch.softmax(-distances, dim=0)
    return P_i

def loss_orth(phi_list, adj_matrices):
    loss = 0
    for i in range(len(phi_list)):
        z = phi_list[i]  
        adj = adj_matrices[i]  
        M = torch.ones_like(adj) - adj  
        ortho_term = M * (z @ z.T - torch.eye(z.size(0), device=z.device))
        loss += torch.norm(ortho_term, p='fro') ** 2
    return loss

def loss_svp(phi_list, A_set):
    loss = 0
    for i in range(len(phi_list)):
        z = phi_list[i]
        a = A_set[i]
        D_a = torch.diag(a)
        svp_term = D_a @ (z @ z.T) @ D_a - D_a ** 2
        loss += torch.norm(svp_term, p='fro') ** 2
    return loss

def loss_Hk(batch_graphs, theta_list, phi_list, A_set):
    loss = 0
    for i, g in enumerate(dgl.unbatch(batch_graphs)):
        P_i = compute_P_i(theta_list[i], phi_list[i])
        a_i = A_set[i]
        loss += -torch.sum(P_i * torch.log(a_i + 1e-8))
    return loss

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

def objective_J1(batch_graphs, theta, phi, A_set, adj_matrices, mu, gamma):
    loss_Hk_value = loss_Hk(batch_graphs, theta, phi, A_set)
    loss_orth_value = loss_orth(phi, adj_matrices)
    loss_svp_value = loss_svp(phi, A_set)
    J1 = - (loss_Hk_value - mu * loss_orth_value - gamma * loss_svp_value)
    return J1

def objective_J2(batch_graphs, theta, phi, A_set, gamma):
    loss_Hk_value = loss_Hk(batch_graphs, theta, phi, A_set)
    loss_svp_value = loss_svp(phi, A_set)
    J2 = loss_Hk_value + gamma * loss_svp_value
    return J2