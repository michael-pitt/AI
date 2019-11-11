"""
PyTorch dataloader and handle
"""
from torch.utils.data import Dataset, DataLoader
import torch

class DataLoader(Dataset):
    def __init__(self, treeLR, treeHR, br_name='cell_Energy', nLayers=6):
        """
        Args:
            treeRH,  treeLR (ROOT.TTree): tree contains HR and LR images respectively
            treeName (string, optional): Name of the event tree
            the LR images are list of nLayer matrixes of different size
        """
        self.treeLR = treeLR
        self.treeHR = treeHR
        self.br_name = br_name
        self.nLayers = nLayers
     
    def __getitem__(self, index):
        x=[]
        if isinstance(index, slice):
            for il in range(1,self.nLayers+1):
                xi = self.treeLR.array(self.br_name+('_L%d'%il), entrystart=index.start, entrystop=index.stop)
                x.append(xi)
            y = self.treeHR.array(self.br_name, entrystart=index.start, entrystop=index.stop)
        else:
            for il in range(1,self.nLayers+1):
                xi = self.treeLR.array(self.br_name+('_L%d'%il), entrystart=index, entrystop=index+1)
                x.append(xi)
            y = self.treeHR.array(self.br_name, entrystart=index, entrystop=index+1)
        return x, y

    def __len__(self):
        return self.treeHR.numentries   


