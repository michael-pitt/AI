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

from preprocess import preprocess
from preprocess import weights_to_labels



class Model():
    def __init__(self, n_phi_bins=5, n_eta_bins = 6):
        super().__init__()
        
        
        self.n_phi_bins = n_phi_bins
        self.n_eta_bins = n_eta_bins
        self.phi_bins = np.linspace(-3.15,3.15,self.n_phi_bins+1)
        self.eta_bins = np.linspace(-4.6,4.6,self.n_eta_bins+1)
               

    def predict_one_event(self, event_id, event, cells=None):
        
        #generate inputs for different eta regions
        list_X, list_Is, list_hits_id, list_labels = preprocess(event.copy(), self.phi_bins, self.eta_bins)
        
        #constuct dataframe for outputs
        sub = pd.DataFrame(columns=['hit_id','track_id']).astype(int)
        
        #loop over all regions to construct tracks
        for X, Is, hits_id, dummy_labels in zip(list_X, list_Is, list_hits_id, list_labels):
        
          #set all weights to 1 
          weights = np.ones(Is.values.shape[0])
          
          #propogate labels through the graph using the weights to obain tracks
          labels = weights_to_labels(X, Is, weights, dummy_labels, hits_id, threshold = 0.01) + sub.shape[0]

          sub = pd.concat([sub,pd.DataFrame(data=np.column_stack((hits_id, labels)),
              columns=["hit_id", "track_id"]).astype(int)])

        sub['event_id'] = event_id
               
        return sub
