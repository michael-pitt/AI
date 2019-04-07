"""
PyTorch dataloader and handle
"""

from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

if torch.cuda.is_available(): torch_ft = torch.cuda.FloatTensor
else: torch_ft = torch.FloatTensor

Graph = namedtuple('Graph', ['X', 'Is', 'y'])

def load_graph(filename):
    #print('loading',filename)
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
def sparse_to_graph(X, Is, y):
    return Graph(X, Is, y)

class trackDataLoader(Dataset):
    def __init__(self, filenames, n_samples=None):
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def collate_fn(graphs):

    batch_size = len(graphs)
    
    if batch_size == 1:
        g = graphs[0]
        
        batch_target = g.y
        batch_inputs = [torch_ft(g.X), g.Is]
        
        return batch_inputs, batch_target
    
    #construct batches
    batch_X=[]; batch_Is=[]; batch_target=[]
    n_hits = 0
    for i, g in enumerate(graphs):
        batch_target.append(g.y)
        batch_X.append(g.X)
        batch_Is.append(g.Is + n_hits)
        n_hits += g.X.shape[0]
    batch_X = np.concatenate(batch_X)
    batch_Is = np.concatenate(batch_Is)
    batch_target = np.concatenate(batch_target)
    
    batch_inputs = [torch_ft(batch_X), batch_Is]
    
    return batch_inputs, batch_target


