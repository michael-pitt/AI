########################################################################
# ======================  TrackML CHALLENGE MODEL  =====================
########################################################################
# Author: Isabelle Guyon, Victor Estrade
# Date: Apr 10, 2018

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# PARIS-SUD UNIVERSITY, THE ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL PARIS-SUD UNIVERSITY AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from GraphBuilder.model_loader import model_loader
from GraphBuilder.model.model import GNNmodel, PreTrainModel

from preprocess import preprocess
from preprocess import weights_to_labels

__authors__ = ['Sabrina Amrouche', 'David Rousseau', 'Moritz Kiehn', 'Ilija Vukotic']


class Model():
    def __init__(self, verbose=1, n_eta_bins = 1, threshold = 0.2):
        super().__init__()
        
        self.verbose = verbose
        
        #set model loader
        self.graph_model = model_loader()
        
        #load model and set trained weights
        self.graph_model.set_threshold(threshold)
        self.graph_model.set_model( PreTrainModel(hidden_features = 32), pre = True)
        self.graph_model.set_model( GNNmodel(edge_dim = 16, hidden_dim = 32, niter = 8), pre = False)
        
        self.graph_model.load_weights('GraphBuilder/data',['weights_pretrain.pt','weights_gnn_filtered.pt'])
        
        self.n_eta_bins = n_eta_bins
        self.eta_bins = np.linspace(-1.5,1.5, self.n_eta_bins+1)
                

    def predict_one_event(self, event_id, event, cells=None):
        
        #add cell info
        #hits = event.merge(cells[['hit_id','value']], on='hit_id')
        
        #generate inputs for different eta regions
        #list_X, list_Is, list_hits_id, list_labels = preprocess(event.copy(), self.phi_bins, self.eta_bins)
        
        features = ['x', 'y', 'z', 'phi', 'eta', 'r']
        list_y, list_X, list_Is, list_hits_id, list_labels = preprocess(event.copy(), self.eta_bins, feature_names=features)
        
        #constuct dataframe for outputs
        sub = pd.DataFrame(columns=['hit_id','track_id']).astype(int)
        
        #loop over all regions to construct tracks
        #for X, Is, hits_id, dummy_labels in zip(list_X, list_Is, list_hits_id, list_labels):
        for y, X, Is, hits_id, dummy_labels in zip(list_y, list_X, list_Is, list_hits_id, list_labels):
        
          #evaluate weights from the model
          weights = np.ones(Is.values.shape[0])
          threshold = 0.90
          weights = self.graph_model.fit_predict(X, Is.values)
          
          #evaluate purity and efficiency for the thredhold
          weights_to_y = weights
          weights_to_y[weights_to_y>threshold] = 1
          weights_to_y[weights_to_y<=threshold] = 0
          print('total weights = ',len(weights_to_y))
          print('sigments with 1 =',sum(y))
          print('weights above the thredhold =',sum(weights_to_y),'with thredhold of ',threshold)
          print('signal eff = %2.2f%% are above threshold'%(y[weights>threshold].sum()/y.sum()*100))
          print('signal purity = above the threshold %2.2f%% are truth edges'%(y[weights>threshold].sum()/(weights>threshold).sum()*100))
          #weights = weights_to_y
          #weights = y
          
          #propogate labels through the graph using the weights to obain tracks
          labels = weights_to_labels(X, Is, weights, dummy_labels, hits_id, threshold = threshold) + sub.shape[0]
          

          sub = pd.concat([sub,pd.DataFrame(data=np.column_stack((hits_id, labels)),
              columns=["hit_id", "track_id"]).astype(int)])

        sub['event_id'] = event_id
               
        #group = event.groupby(by=['particle_id'])
        #track_idx = group.indices
        #ntracks_truth = 0
        #for pid, idx in track_idx.items():
        #    if len(idx) > 4:ntracks_truth += 1
            
        #group = sub.groupby(by=['track_id'])
        #track_idx = group.indices
        #ntracks_rec = 0
        #for pid, idx in track_idx.items():
        #    if len(idx) > 4:ntracks_rec += 1

        #print('coubnt tracks with >4 hits')
        #print('found ',ntracks_rec,'tracks, but have only ', ntracks_truth ,' tracks')

                  
        return sub
