"""
This file contains GNN architecture used for the trackML challenge.
author: Michael Pitt (michael.pitt@cern.ch)
date: March 2019
"""

import torch
from torch import nn
       
class PreTrainModel(nn.Module):
    def __init__(self, input_dim=6, hidden_features=40, hidden_activation=nn.Tanh):
        super(PreTrainModel, self).__init__()
        
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),            
            hidden_activation())
                
        self.network2 = nn.Sequential(
            nn.Linear(input_dim + hidden_features, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),   
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),            
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation())
        
        self.network3 = nn.Sequential(
            nn.Linear(input_dim + 2*hidden_features, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features//2, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features//2, hidden_features//4, bias=True),
            hidden_activation(),            
            nn.Linear(hidden_features//4, 1, bias=False))
        
        
    def forward(self, inputs):
        X, Ro, Ri = inputs
        
        #combine two nodes to construct edge:
        bo = Ro.transpose(0,1).mm(X)
        bi = Ri.transpose(0,1).mm(X)
        E = bo - bi
        
        #reshape input parameters:
        ReshapeEdges(E)
        
        E1 = self.network1(E)
        E2 = self.network2(torch.cat([E1, E], dim=-1))
        e = self.network3(torch.cat([E1, E2, E], dim=-1))
        return e.squeeze(-1) 


class EdgeClasification(nn.Module):
    def __init__(self, input_dim, hidden_features, hidden_activation=nn.Tanh):
        super(EdgeClasification, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features//2, bias=True),            
            hidden_activation(),
            nn.Linear(hidden_features//2, hidden_features//4, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features//4, 1, bias=True),
            nn.Sigmoid())
        
    def forward(self, H, Ro, Ri, e):
        eH = e[:,None]*H
        Ho = Ri.transpose(0,1).mm(Ro.mm(eH))
        Hi = Ro.transpose(0,1).mm(Ri.mm(eH))
        return self.network(torch.cat([Ho, H, Hi], dim=1))
        #return self.network(H)

class EdgeRepresentation(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_features, hidden_activation=nn.Tanh):
        super(EdgeRepresentation, self).__init__()
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation())
        
        self.network2 = nn.Sequential(
            nn.Linear(input_dim + hidden_features, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features, bias=True),
            hidden_activation())
               
        self.network3 = nn.Sequential(
            nn.Linear(input_dim + 2*hidden_features, hidden_features, bias=False),
            hidden_activation(),
            nn.Linear(hidden_features, hidden_features//2, bias=True),
            hidden_activation(),
            nn.Linear(hidden_features//2, edge_dim, bias=False),
            hidden_activation())
        
    def forward(self, E):
        H1 = self.network1(E)
        H2 = self.network2(torch.cat([H1, E], dim=-1))
        return self.network3(torch.cat([H1, H2, E], dim=-1))

class GNNmodel(nn.Module):
    def __init__(self, input_dim=6, edge_dim = 8, hidden_dim=40, niter=2, hidden_activation=nn.Tanh):
        super(GNNmodel, self).__init__()
        
        self.edge_repr = EdgeRepresentation(input_dim, edge_dim, hidden_dim, hidden_activation)
                             
        self.clasifier = EdgeClasification(edge_dim, hidden_dim, hidden_activation)
        
        self.niter = niter
                
    def forward(self, inputs):
        X, Ro, Ri, w = inputs
        
        #combine two nodes to construct edge:
        bo = Ro.transpose(0,1).mm(X)
        bi = Ri.transpose(0,1).mm(X)
        E = bo - bi
        ReshapeEdges(E)
        
        H = self.edge_repr(E)
            
        # evaluate edge weight
        for i in range(self.niter):
            e = self.clasifier(H, Ro, Ri, w)
        
        # learn from neighbours
        #e = self.clasifier(H, Ro, Ri, e) 
        #optimize future training for high edge weights only
        
        return e.squeeze(-1) 
    


def ReshapeEdges(E):
    E[:,0] = E[:,0]/1300
    E[:,1] = E[:,1]/1300
    E[:,2] = E[:,2]/250
    E[:,3] = torch.acos(torch.cos(E[:,3]))/1.57
    E[:,4] = torch.abs(E[:,4])/2.0
    E[:,5] = -E[:,5]/210.0
    