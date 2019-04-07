"""
Python code used for training PyTorch DNN
"""
import torch
import time
import numpy as np

def CreateCash(model):
    cache={}
    for c in (['loss']+['val_loss']+[*model.state_dict().keys()]):
        cache[c]=[]
    return cache

def get_inputs(X, Is, device):

    n_hits = X.shape[0]
    n_edges = Is.shape[0]
    edge_idx = np.arange(n_edges)
    Ri = torch.sparse.FloatTensor(torch.LongTensor([Is[:,1],edge_idx]),
                        torch.ones(n_edges),
                        torch.Size([n_hits,n_edges]))
    Ro = torch.sparse.FloatTensor(torch.LongTensor([Is[:,0],edge_idx]),
                        torch.ones(n_edges),
                        torch.Size([n_hits,n_edges]))
    
    if 'cuda'==device.type: return [X, Ro.to(device), Ri.to(device)]  
    return [X, Ro, Ri]

def get_batch_weights(batch_target, weighted, pretrain):
    if(weighted):
        frac_connected_edges = batch_target.sum()/batch_target.shape[0]
        batch_weights_real = batch_target * (1.0 - frac_connected_edges)
        batch_weights_fake = (1 - batch_target) * frac_connected_edges
        batch_weights = batch_weights_real + batch_weights_fake
    else:
        batch_weights_real = batch_target
        batch_weights_fake = (1 - batch_target)
        batch_weights = batch_weights_real + batch_weights_fake
        
    if not pretrain:
        batch_weights = batch_weights_real*(1 - frac_connected_edges) + batch_weights_fake
    
    return batch_weights
 
def train(data_loader, model, optimizer, criterion, epochs=1, 
          cache={'loss':[]}, device=torch.device("cpu"), 
          pretrain=None, weighted = False, validation_set = None):
    
    print('len of cache is = ',len(cache['loss']))
    isGPU = torch.cuda.is_available() and 'cuda'==device.type
    if isGPU: torch_ft = torch.cuda.FloatTensor
    else: torch_ft = torch.FloatTensor
    if isGPU and not next(model.parameters()).is_cuda:
        print('copy the model to GPU')
        model.to(device)
    tic = time.time()
    for epoch in range(epochs):
        for i, (batch_input, batch_target) in enumerate(data_loader):
            
            #training performed in two steps, once on the full data, remove back edges bellow the threshold
            #then train again (with pretrain=pretrained model)
            threshold = 0.2
            
            X, Is = batch_input
            batch_input = get_inputs(X, Is, device)

            batch_target = torch_ft(batch_target)

            if pretrain:
                pretrain.eval()
                with torch.no_grad():
                    e = torch.sigmoid(pretrain(batch_input))
                    
                #filter inputs
                mask_edges = ((e + batch_target) > threshold).nonzero().squeeze().cpu()
                Is_filter = Is[mask_edges]
                e = e[mask_edges]
                batch_target = batch_target[mask_edges]
                batch_input = get_inputs(X, Is_filter, device)
                batch_input.append(e)
                
                #batch_weights = batch_target + (1 - batch_target)

            batch_weights = get_batch_weights(batch_target, weighted, pretrain)
            
            #evaluate the model, and compute the gradients
            batch_output = model(batch_input)
            loss=criterion(batch_output, batch_target, weight=batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        cache['loss'].append(loss.item())
        #evaluate validation loss:
        if validation_set:
            with torch.no_grad():
                test_inputs, test_target = next(iter(test_loader))
                test_X, test_Is = test_inputs
                test_inputs = get_inputs(test_X, test_Is, device)
                test_target = torch_ft(test_target)
                
                if pretrain:
                    test_e = torch.sigmoid(pretrain(test_inputs))

                    #filter first training:
                    test_mask_edges = (test_e > threshold).nonzero().squeeze().cpu()
                    Is_filter = test_Is[test_mask_edges]
                    test_e = test_e[test_mask_edges]
                    test_target = test_target[test_mask_edges]
                    test_inputs = get_inputs(test_X, Is_filter, device)
                    test_inputs.append(test_e)
                    
                test_weights = get_batch_weights(test_target, weighted, pretrain)
    
                #evaluate
                test_pred = model(test_inputs)
                vloss = criterion(test_pred, test_target, weight=test_weights)
                cache['val_loss'].append(vloss.item())
                
            printProgressBar(epoch, epochs, [loss.item(),vloss.item()])
        else: printProgressBar(epoch, epochs, [loss.item()])
    toc = time.time()
    print('total time: %2.2f sec' %(toc-tic))
    return cache

def printProgressBar (iteration, total, losses = [], decimals = 1, length = 50):
    total = total - 1 #since usually we start from 0 till n-1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '#' * filledLength + '-' * (length - filledLength)
    if len(losses)==1:
        print('\rprogress |%s| %s%% loss - %s' % (bar, percent, str(losses[0])), end = '\r')
    else:
        print('\rprogress |%s| %s%% loss - %2.5f | validation - %2.5f' % 
              (bar, percent, float(losses[0]), float(losses[1])), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
  