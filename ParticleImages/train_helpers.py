import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
import time
from particleImages_helpers import printProgressBar
import GPUtil as GPU

''' 
Data Loaders:
 1. SRDataLoader(tree1, tree2): DataLoader for the SR task. The input is two trees, one with LR and one with HR images. The dataloader will load event by event from both trees the data, and convert it to pytorch tensors
'''

class SRDataLoader(Dataset):
    def __init__(self, treeLR, treeHR, br_name='cell_Energy'):
        """
        Args:
            treeRH,  treeLR (ROOT.TTree): tree contains HR and LR images respectively
            treeName (string, optional): Name of the event tree
        """
        self.treeLR = treeLR
        self.treeHR = treeHR
        self.br_name = br_name
     
    def __getitem__(self, index):
        if isinstance(index, slice):
            x = self.treeLR.array(self.br_name, entrystart=index.start, entrystop=index.stop)
            y = self.treeHR.array(self.br_name, entrystart=index.start, entrystop=index.stop)
        else:
            x = self.treeLR.array(self.br_name, entrystart=index, entrystop=index+1)
            y = self.treeHR.array(self.br_name, entrystart=index, entrystop=index+1)
        return x.squeeze(), y.squeeze()

    def __len__(self):
        return self.treeHR.numentries   

class SRDataLoaderLayers(Dataset):
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
        
'''
Trainer and Tester
'''

def trainMe(train_loader, model, optimizer, criterion, epochs=1000, cache={'loss':[]}, device=torch.device("cpu")):
    print('len of cache is = ',len(cache['loss']))
    isGPU = torch.cuda.is_available() and 'cuda'==device.type
    if isGPU and not next(model.parameters()).is_cuda:
        print('copy the model to GPU')
        model.to(device)
    tic = time.time()
    for epoch in range(epochs):
        for x, y in train_loader:
            y = y.squeeze()
            if isGPU:
                if isinstance(x,type([])):
                    x = [k.to(device) for k in x]
                else: x = x.to(device)
                y = y.to(device)
            yhat = model(x)
            loss=criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cache['loss'].append(loss.item())
        printProgressBar(epoch, epochs, loss.item())
    toc = time.time()
    print('total time: %2.2f sec' %(toc-tic))
    if isGPU:
        print('cached',torch.cuda.memory_cached(device)/1e6,' MB')
        torch.cuda.empty_cache() 
        print('Cache cleaned\nRemaining cached memory',torch.cuda.memory_cached(device)/1e6,' MB')
    return  cache

def CreateCash(model):
    cache={}
    for c in (['loss']+[*model.state_dict().keys()]):
        cache[c]=[]
    return cache        

def TestMe(test_loader, model, device=torch.device("cpu")):
    isGPU = torch.cuda.is_available() and 'cuda'==device.type
    if isGPU and not next(model.parameters()).is_cuda:
        print('copy the model to GPU')
        model.to(device)
    tic = time.time()
    y_pred = []
    for x, y in test_loader:
        if isGPU:
            if isinstance(x,type([])):
                x = [k.to(device) for k in x]
            else: x = x.to(device)
            y = y.to(device)
        y_pred.append(model(x).cpu().detach().numpy())
    toc = time.time()
    print('total time: %2.2f sec' %(toc-tic))
    print('cached',torch.cuda.memory_cached(device)/1e6,' MB')
    torch.cuda.empty_cache() 
    print('Cache cleaned\nRemaining cached memory',torch.cuda.memory_cached(device)/1e6,' MB')
    return np.concatenate(y_pred)

def GetDevice(device_number = 0):
    use_cuda = torch.cuda.is_available()
    print('Availability of CUDA:',use_cuda)
    print('Availability of CUNN:',torch.backends.cudnn.enabled)
    print('Total number of GPU devices: ',torch.cuda.device_count())
    device = torch.device("cuda:"+str(device_number) if torch.cuda.is_available() else "cpu")
    if use_cuda:
        gpu = GPU.getGPUs()[0]
        torch.cuda.set_device(device_number)
        idevice = torch.cuda.current_device()
        print('Will work on device number',idevice,', named: ',torch.cuda.get_device_name(idevice))
        print("GPU RAM Free: {0:.0f}GB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree/1e3, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    else: print('will run on CPU, using',torch.get_num_threads(),'cores')
    return device

	
