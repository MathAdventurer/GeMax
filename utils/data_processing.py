# GeMax: Learning Graph Representation via Graph Entropy Maximization
import warnings
warnings.filterwarnings("ignore")
import dgl
from dgl.data import GINDataset

def load_dataset(dataset_name):
    """Load graph dataset."""
    dataset = GINDataset(name=dataset_name, self_loop=True) # raw_dir='data/'
    return dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    """Create dataloader for graph dataset."""
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader