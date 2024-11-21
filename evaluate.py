# GeMax: Learning Graph Representation via Graph Entropy Maximization
import numpy as np  
import torch
import warnings
warnings.filterwarnings("ignore")
import dgl
from utils.data_processing import load_dataset, get_dataloader
from utils.evaluation import evaluate_embedding

def evaluate_gemax(config, model=None, data_loader=None, device=None):
    if model is None:
        model = torch.load(config["model_path"])
    model.eval()

    if data_loader is None:
        dataset = load_dataset(config["dataset"])
        data_loader = get_dataloader(dataset, batch_size=config["batch_size"], shuffle=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_embeddings = []
    labels = []
    with torch.no_grad():
        for batch_graphs, batch_labels in data_loader:
            batch_graphs = batch_graphs.to(device)
            _, phi = model(batch_graphs)
            graph_emb = dgl.mean_nodes(batch_graphs, 'h')
            graph_embeddings.append(graph_emb.cpu().numpy())
            labels.append(batch_labels.numpy())

    graph_embeddings = np.concatenate(graph_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    accuracy, std = evaluate_embedding(graph_embeddings, labels, search=True, device=device)
    print(f"Unsupervised Accuracy: {accuracy:.2f} ± {std:.2f}")