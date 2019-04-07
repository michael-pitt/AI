#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import time

import numpy as np
import pandas as pd

from trackml.dataset import load_dataset
from trackml.score import score_event

from model import Model


PATH_TO_DATA = '/home/data'
MAX_TIME_PER_EVENT = 600
ACCURACY_INF = 0.5
get_clock = time.perf_counter


def mixt_score(accuracy_mean, time_per_event):
    if time_per_event <= 0 or time_per_event > MAX_TIME_PER_EVENT:
        return -1  # Something went terribly wrong
    if accuracy_mean < ACCURACY_INF:
        return 0
    speed = np.log(1.0 + ( MAX_TIME_PER_EVENT / time_per_event ))
    score = np.sqrt(speed * (accuracy_mean - ACCURACY_INF)**2)
    return score

def getLayer(volume_id, layer_id):
    if volume_id==8:
        return layer_id//2
    elif volume_id==13:
        return layer_id//2 + 4
    elif volume_id==17:
        return layer_id//2 + 8
    else:
        return -1  

def main():
    tracker = Model()
    time_spent = 0
    n_event = 0
    score_sum = 0
    for event_id, hits, cells, truth in load_dataset(PATH_TO_DATA, parts=['hits', 'cells', 'truth']):
        print("Runing event", event_id, "...", flush=True)
            
        #############################################################
        #MP hack - use only the barrel for now:
        new_hits = hits.copy()
        new_truth = truth.copy()
        print('total hits ' ,new_hits.shape[0])
        
		#add particle ID to hit dataset, store original keys, add layer to truth info:
        new_hits = new_hits.merge(new_truth[['hit_id','particle_id']], on='hit_id').copy()
        new_hits_keys = new_hits.keys()
        new_truth_keys = new_truth.keys()
        new_truth = new_truth.merge(new_hits[['hit_id','volume_id','layer_id']], on='hit_id').copy()

		#set eta and layer number in hits:
        r = np.sqrt(new_hits['x']**2 + new_hits['y']**2)
        theta = np.arctan2(r,new_hits['z'])
        new_hits['eta'] = -np.log(np.tan(0.5*theta))
        new_hits['layer'] = new_hits.apply(lambda x: getLayer(x['volume_id'],x['layer_id']), axis=1)

		#set eta and layer number in truth:
        r = np.sqrt(new_truth['tx']**2 + new_truth['ty']**2)
        theta = np.arctan2(r,new_truth['tz'])
        new_truth['eta'] = -np.log(np.tan(0.5*theta))
        new_truth['layer'] = new_truth.apply(lambda x: getLayer(x['volume_id'],x['layer_id']), axis=1)

		#remove noise and keep only tracks within the barrel
        group_hits = new_hits.groupby(by=['particle_id'])
        new_hits = group_hits.filter(lambda x: x['layer'].min() > 0)
        new_hits = new_hits.loc[new_hits['particle_id']>0]
        print('remove noise and tracks outside the barrel: ',new_hits.shape[0])
		
		#do the same for truth info
        group_hits = new_truth.groupby(by=['particle_id'])
        new_truth = group_hits.filter(lambda x: x['layer'].min() > 0)
        new_truth = new_truth.loc[new_truth['particle_id']>0]
       
        #filter to (deta) window
        eta_cut = (-1.5,1.5); 
        new_hits = (new_hits.loc[(new_hits['eta']>eta_cut[0]) & (new_hits['eta']<=eta_cut[1])])
        new_truth = (new_truth.loc[(new_truth['eta']>eta_cut[0]) & (new_truth['eta']<=eta_cut[1])])
        print('eta cut ' ,new_hits.shape[0])
		
        #filter to tracks with at least 10 hits
        group_hits = new_hits.groupby(by=['particle_id'])
        track_idx = group_hits.indices
        new_hits = pd.concat([group_hits.get_group(pid).assign(nhits=len(idx)) for pid, idx in track_idx.items()])
        new_hits = new_hits.loc[(new_hits['nhits']>9)]
        print('n_hits cut' ,new_hits.shape[0])
		
		#do the same for truth info
        group_hits = new_truth.groupby(by=['particle_id'])
        track_idx = group_hits.indices
        new_truth = pd.concat([group_hits.get_group(pid).assign(nhits=len(idx)) for pid, idx in track_idx.items()])
        new_truth = new_truth.loc[(new_truth['nhits']>9)]		

        print('remaining number of hits after the filtering = ',new_hits.shape[0])
        print('remaining number of tracks after the filtering(truth) = ',(new_truth.groupby(by=['particle_id'])).ngroups)
		
		#restore original keys:
        hits = new_hits[new_hits_keys]
        truth = new_truth[new_truth_keys]
		
        #############################################################
			
        # Make predictions
        t_start = get_clock()
        sub = tracker.predict_one_event(event_id, hits)
        t_end = get_clock()
        # Compute accuracy score
        score = score_event(truth, sub)
        # accumulate time, score, number of events
        time_spent += t_end - t_start
        score_sum  += score
        n_event += 1
        time_per_event = time_spent / n_event
        score = score_sum / n_event
        # Print information
        print("event", event_id, "accuracy score :", score)
        print("event", event_id, 'time spent     :', t_end - t_start)
        print('total time spent:', time_spent)
        print("running speed   : {:.3f} sec per event".format(time_spent  / n_event))
        print("running score   :", mixt_score(score, time_per_event))
        print('-----------------------------------', flush=True)
    if n_event == 0:
        print("Warning: no event where found in the given directory.")
        exit()
    if time_spent <= 0:
        print("Warning : execution time <= 0. Something went wrong !")

    time_per_event = time_spent / n_event
    score = score_sum / n_event

    print("Accuracy mean      :", score)
    print("Time per event     :", time_per_event)
    print("Overall mixt score :", mixt_score(score, time_per_event))


if __name__ == '__main__':
    main()
