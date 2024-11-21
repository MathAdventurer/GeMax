# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import dgl
from models.gin import GIN
from models.infograph import InfoGraph
from models.gcn import GCN
from models.gemax import loss_Hk, loss_orth, loss_svp
from utils.data_processing import load_dataset, get_dataloader, IndexedGraphDataset, collate
from utils.evaluation import evaluate_embedding
from evaluate import evaluate_gemax
from torch.utils.data import DataLoader


def experiment_gemax(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config["dataset"])
    print(f"Training and Evaluation on {config['dataset']} dataset...")
       
    graphs = [g for g, _ in dataset]
    labels = [l for _, l in dataset]
    indexed_dataset = IndexedGraphDataset(graphs, labels)

    data_loader = DataLoader(indexed_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    in_features = dataset[0][0].ndata['attr'].shape[1]
    
    if config["model"] == "infograph":
        model = InfoGraph(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    elif config["model"] == "gin":
        model = GIN(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    elif config["model"] == "gcn":
        model = GCN(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    else:
        try:
            exec(f"from models.{config['model']} import {config['model']}")
            exec(f"model = {config['model']}(in_features=in_features, hidden_features=config['hidden_dim'], out_features=config['out_dim']).to(device)")
        except:
            raise FileNotFoundError(f"Please add the corresponding GNN model script: {config['model']} in models directory.")

    mu = config["mu"]
    gamma = config["gamma"] 
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    A_set = []
    for g in dataset:
        num_nodes = g[0].num_nodes()
        init_prob = torch.full((num_nodes,), 1.0 / num_nodes, device=device, requires_grad=True)
        A_set.append(init_prob)

    optimizer_A = optim.Adam(A_set, lr=config["lr_A"])

    epoch_losses = []
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_graphs, batch_labels, batch_indices) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)
            batch_indices = batch_indices.to(device)
            batch_A_set = [A_set[idx] for idx in batch_indices.tolist()]

            theta, phi = model(batch_graphs)

            num_nodes_list = batch_graphs.batch_num_nodes().tolist()
            phi_list = list(torch.split(phi, num_nodes_list))
            theta_list = [theta[i] for i in range(theta.size(0))]
            
            adj_matrices = [g.adj().to_dense() for g in dgl.unbatch(batch_graphs)]

            # Compute losses for J1
            loss_Hk_value = loss_Hk(batch_graphs, theta_list, phi_list, batch_A_set)
            loss_orth_value = loss_orth(phi_list, adj_matrices)
            loss_svp_value = loss_svp(phi_list, batch_A_set)
            loss_J1 = - (loss_Hk_value - mu * loss_orth_value - gamma * loss_svp_value)

            optimizer.zero_grad()
            loss_J1.backward()
            optimizer.step()

            # Compute J2
            with torch.no_grad():
                theta, phi = model(batch_graphs)
            # Recompute phi_list and theta_list since after J1 optimization, phi and theta are detached
            phi_list = list(torch.split(phi, num_nodes_list))
            theta_list = [theta[i] for i in range(theta.size(0))]

            loss_Hk_value = loss_Hk(batch_graphs, theta_list, phi_list, batch_A_set)
            loss_svp_value = loss_svp(phi_list, batch_A_set)
            loss_J2 = loss_Hk_value + gamma * loss_svp_value

            optimizer_A.zero_grad()
            loss_J2.backward()
            optimizer_A.step()

            with torch.no_grad():
                for a_j in batch_A_set:
                    a_j.clamp_(0, 1)

            epoch_loss += loss_J1.item()

        epoch_loss /= len(data_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

        if (epoch + 1) % config["eval_every"] == 0:
            valid_loader = get_dataloader(dataset, batch_size=config["batch_size"], shuffle=True)
            evaluate_gemax(config, model, valid_loader, device)

    return model, epoch_losses