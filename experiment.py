# GeMax: Learning Graph Representation via Graph Entropy Maximization
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import dgl
from models.gin import GIN
from models.infograph import InfoGraph
from models.gcn import GCN
from models.gemax import objective_J1
from utils.data_processing import load_dataset, get_dataloader
from utils.evaluation import evaluate_embedding
from evaluate import evaluate_gemax


def experiment_gemax(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(config["dataset"])
    
    print(f"Training and Evaluation on {config['dataset']} dataset...")
    
    data_loader = get_dataloader(dataset, batch_size=config["batch_size"], shuffle=True)

    in_features = dataset[0][0].ndata['attr'].shape[1]
    
    if config["model"] == "infograph":
        model = InfoGraph(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    elif config["model"] == "gin":
        model = GIN(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    elif config["model"] == "gcn":
        model = GCN(in_features=in_features, hidden_features=config["hidden_dim"], out_features=config["out_dim"]).to(device)
    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    epoch_losses = []
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch_graphs, batch_labels in data_loader:
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)

            A_set = []
            for g in dgl.unbatch(batch_graphs):
                num_nodes = g.num_nodes()
                init_prob = torch.ones(num_nodes, device=device) / num_nodes
                A_set.append(init_prob)

            theta, phi = model(batch_graphs)
            loss = objective_J1(batch_graphs, theta, phi, A_set, config["mu"], config["gamma"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

        if (epoch + 1) % config["eval_every"] == 0:
            evaluate_gemax(config, model, data_loader, device)

    return model, epoch_losses

