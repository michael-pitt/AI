'''
wrapper for graph neural network used for trackML challenge
'''
import torch
import numpy as np

class model_loader():
    def __init__(self):
        super().__init__()
        self.model1 = None
        self.model2 = None
        self.threshold = 0.0
        
    def set_model(self, model, pre=True):
        if pre:
            self.model1 = model
        else:
            self.model2 = model
    
    def set_threshold(self, threshold):
        self.threshold = threshold

        
    def load_weights(self, path, files):
        if not self.model1 or not self.model2:
            print('model_loader() error: model is not defined, call set_model(model) before loading the weights')
            return
        self.model1.load_state_dict(torch.load(path+'/'+files[0], map_location=torch.device('cpu')))
        self.model1.eval()
        self.model2.load_state_dict(torch.load(path+'/'+files[1], map_location=torch.device('cpu')))
        self.model2.eval()

    
    def get_inputs(self, X, Is):

        n_hits = X.shape[0]
        n_edges = Is.shape[0]
        edge_idx = np.arange(n_edges)
        Ri = torch.sparse.FloatTensor(torch.LongTensor([Is[:,1],edge_idx]),
                            torch.ones(n_edges),
                            torch.Size([n_hits,n_edges]))
        Ro = torch.sparse.FloatTensor(torch.LongTensor([Is[:,0],edge_idx]),
                            torch.ones(n_edges),
                            torch.Size([n_hits,n_edges]))
        
        return [X, Ro, Ri]
    
    def fit_predict(self, X, Is):
   	    
        X = torch.FloatTensor(X)
        inputs = self.get_inputs(X, Is)
        weights = torch.sigmoid(self.model1(inputs))
        
        #filter first training
        mask_edges = (weights > self.threshold).nonzero().squeeze()
        Is_filter = Is[mask_edges]
        e_masked = weights[mask_edges]
        inputs = self.get_inputs(X, Is_filter)
        inputs.append(e_masked)
        
        
        weights_masked = self.model2(inputs)
        weights[mask_edges] = weights_masked
        weights = weights.detach().numpy()
        
        return weights


