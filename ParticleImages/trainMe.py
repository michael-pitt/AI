import uproot
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from operator import mul
import GPUtil as GPU  #this one used to see the GPU memory

#model includes
from models import NewConv2d
from models import data_loader

#helpers
from train_helpers import trainMe, CreateCash

def PrintGPUInfo(device_number = 0):
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
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree/1e3, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    else: print('will run on CPU, using',torch.get_num_threads(),'cores')
    return device
    
def LoadData():
    inputfileHR = 'data/events_6D64x64.root'
    inputfileLR = 'data/events_6D64x64_ATLAS_resolution.root'
  
    f = uproot.open(inputfileHR)
    treeHR = f['EventTree']

    f = uproot.open(inputfileLR)
    treeLR = f['EventTree']

    print('Total number of events in both files is = '+str(treeHR.numentries)+' (HR) and '+str(treeLR.numentries)+' (LR)')
    
    return treeLR, treeHR

def ReadShapes(data_generator):
    imageLR, imageHR = data_generator[0]
    HR_shape = imageHR.shape[2:4]
    LR_shapes = []
    for i in range(len(imageLR)):
        LR_shapes.append(imageLR[i].shape[1:3])
    return LR_shapes, HR_shape

def main(args):
    
    parser = argparse.ArgumentParser(description='usage: %prog [options]')
    parser.add_argument('-l','--layer', dest='Layer_i', type=int, required=True,
                        help='layer number')
    parser.add_argument('-n','--Nepoch', dest='nepoch', type=int, default=10,
                        help='number of epoch [default=%default]')
    parser.add_argument('-bs','batchSize', dest='bs', type=int, default=64,
                        help='batch size [default=%default]')
    parser.add_argument('-lr','learningRate', dest='lr', type=float, default=0.05,
                        help='Learning rate [default=%default]')
    parser.add_argument('-d','--device', dest='deviceN', type=int, default=0,
                        help='device number (for multiple GPUs) [default=%default]')
    
    opt=parser.parse_args(args)
    print("Will start training for layer ",opt.Layer_i)

    device = PrintGPUInfo(opt.deviceN)
    
    # Load the data
    print('Load data')
    treeLR, treeHR = LoadData()
    train_dataset = data_loader.DataLoader(treeLR, treeHR)
    train_loader=DataLoader(dataset=train_dataset,batch_size=opt.bs ) #train_size

    #construct a model
    print('Constract the model')
    LR_shapes, HR_shape = ReadShapes(train_dataset)
    model = NewConv2d.model(LR_shapes, HR_shape, opt.Layer_i)
    print('model parameters = ',sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    #construct the loss function
    def criterion(yhat, y):
        vec = (yhat - y[:,opt.Layer_i,:,:]).pow(2)
        return  torch.mean( vec ) 
    
    #create data loader
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    cacheSR = CreateCash(model)
    
    #train the model
    cacheSR = trainMe(train_loader, model, optimizer, criterion, opt.nepoch, cacheSR, device)

    #saving the model
    torch.save({
        'model_state_dict' : model.cpu().state_dict(),
        'cache' : cacheSR,
        },
        'models/UNET_SR_dipion_dict_L%d.pt'%opt.Layer_i)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
      
