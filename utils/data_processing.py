# GeMax: Learning Graph Representation via Graph Entropy Maximization
import warnings
warnings.filterwarnings("ignore")
import dgl
from dgl.data import GINDataset
import torch
from torch.utils.data import DataLoader

def load_dataset(dataset_name):
    """Load graph dataset."""
    dataset = GINDataset(name=dataset_name, self_loop=True) # raw_dir='data/'
    return dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    """Create dataloader for graph dataset."""
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class IndexedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        return graph, label, idx  # Return the index along with the graph and label

def collate(samples):
    graphs, labels, indices = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    indices = torch.tensor(indices)
    return batched_graph, labels, indices