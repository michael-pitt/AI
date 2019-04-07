
import numpy as np
import pandas as pd

cimport cython
cimport numpy as cnp
from numpy cimport ndarray as ar
from libc.math cimport sqrt, sin, atan2, log, tan, fabs
#from libc.stdlib cimport malloc, free

# cython parallelism
#from cython.parallel import prange

FLOAT_32 = np.float32
INT_32 = np.int32
INT_64 = np.int64

ctypedef cnp.float32_t FLOAT_32_t
ctypedef cnp.int32_t INT_32_t
ctypedef cnp.int64_t INT_64_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)  

cpdef _augmentation(ar[FLOAT_32_t, ndim=1] x, ar[FLOAT_32_t, ndim=1] y, ar[FLOAT_32_t, ndim=1] z, ar[FLOAT_32_t, ndim=1] phi, ar[FLOAT_32_t, ndim=1] r, ar[FLOAT_32_t, ndim=1] eta, ar[INT_32_t, ndim=1] layer, ar[INT_32_t, ndim=1] volume_id, ar[INT_32_t, ndim=1] layer_id):
    assert x.dtype == FLOAT_32 and y.dtype == FLOAT_32 and z.dtype == FLOAT_32 and r.dtype == FLOAT_32 and phi.dtype == FLOAT_32 and eta.dtype == FLOAT_32 and layer.dtype == INT_32 and volume_id.dtype == INT_32 and layer_id.dtype == INT_32
    
    cdef Py_ssize_t size = x.shape[0]
    cdef FLOAT_32_t theta = 0
	
    for i in range(size):
        phi[i] = atan2(y[i], x[i])
        r[i] = sqrt(x[i]**2 + y[i]**2)
        theta = atan2(r[i], z[i])
        eta[i] = -log(tan(0.5*theta))
        
        if volume_id[i]==8:
            layer[i] = layer_id[i]/2
        elif volume_id[i]==13:
            layer[i] = layer_id[i]/2 + 4
        elif volume_id[i]==17:
            layer[i] = layer_id[i]/2 + 8
        elif volume_id[i]==7:
            layer[i] = 8 - layer_id[i]/2
        elif volume_id[i]==9:
            layer[i] = layer_id[i]/2
        elif volume_id[i]==12:
            layer[i] = 7 - layer_id[i]/2
        elif volume_id[i]==14:
            layer[i] = layer_id[i]/2
        elif volume_id[i]==16:
            layer[i] = 7 - layer_id[i]/2
        elif volume_id[i]==18:
            layer[i] = layer_id[i]/2
        else: layer[i] = -1

cdef inline bint far_pairs(float r1, float r2, float z1, float z2, float phi1, float phi2, float x1, float x2, float y1, float y2):
    
    cdef float rho2_min_times4 = 4 * 300*300 # for 0.9GeV tracks. 
    #previously used 300 which corresponding to 300*0.6 = 0.18 GeV
    cdef float z0_max = 150 # previously used 150
    cdef float dphi = 0
    cdef float sindphi = 0
    cdef float dx = 0
    cdef float dy = 0
    cdef float dz = 0
    cdef float dr = 0
    cdef float zr_cross = 0
    
    #constrain z_0
    zr_cross = fabs(r1*z2 - z1*r2)
    if( zr_cross > z0_max*fabs(r1 - r2) ): return True

    #constrain curvature radius
    dphi = (phi1 - phi2)
    if dphi > 1.57 and dphi < 4.71: return True
    if dphi < -1.57 and dphi > -4.71: return True	
    sindphi = sin(dphi)
    dx =  (x1-x2)
    dx *= dx
    dy =  (y1-y2)
    dy *= dy
    dr = dx + dy
    if ( rho2_min_times4 * sindphi * sindphi > dr ): return True
    return False

cpdef _get_hitpairs(float[:] phi1, float[:] phi2, float[:] x1, float[:] x2, float[:] y1, float[:] y2, float[:] z1, float[:] z2, float[:] r1, float[:] r2):

    cdef Py_ssize_t size1 = phi1.shape[0]
    cdef Py_ssize_t size2 = phi2.shape[0]
    cdef INT_64_t[:] result_o = np.empty(size1*size2, INT_64)
    cdef INT_64_t[:] result_i = np.empty(size1*size2, INT_64)
    cdef int iterator = 0
    cdef long i, j
	
    #loop over all possible pairs
    for i in range(size1):
        for j in range(size2):
            if far_pairs(r1[i], r2[j], z1[i], z2[j], phi1[i], phi2[j], x1[i], x2[j], y1[i], y2[j]): continue
            
            #store indeces of the selected pair of hits (need to optimize memore ussage - bottleneck)
            result_o[iterator] = i
            result_i[iterator] = j
            iterator = iterator + 1
            
    return np.concatenate([result_o[:iterator,None], result_i[:iterator,None]], axis=1)

cpdef _get_hitpairs_truth(long[:] pid1, long[:] pid2, float[:] phi1, float[:] phi2, float[:] x1, float[:] x2, float[:] y1, float[:] y2, float[:] z1, float[:] z2, float[:] r1, float[:] r2):

    cdef Py_ssize_t size1 = phi1.shape[0]
    cdef Py_ssize_t size2 = phi2.shape[0]
    cdef INT_64_t[:] result_o = np.empty(size1*size2, INT_64)
    cdef INT_64_t[:] result_i = np.empty(size1*size2, INT_64)
    cdef INT_64_t[:] labels = np.empty(size1*size2, INT_64)
    cdef int iterator = 0
    cdef long i, j
	
    #loop over all possible pairs
    for i in range(size1):
        for j in range(size2):
            if far_pairs(r1[i], r2[j], z1[i], z2[j], phi1[i], phi2[j], x1[i], x2[j], y1[i], y2[j]): continue
            
            #store indeces of the selected pair of hits (need to optimize memore ussage - bottleneck)
            result_o[iterator] = i
            result_i[iterator] = j
            if (pid1[i] == pid2[j]):
              labels[iterator] = 1
            else:
              labels[iterator] = 0
            iterator = iterator + 1
            
    return np.concatenate([result_o[:iterator,None], result_i[:iterator,None], labels[:iterator,None]], axis=1)

   
cpdef _get_edgepairs(ar[INT_64_t, ndim=1] idx1_i, ar[INT_64_t, ndim=1] idx2_o):
    assert idx1_i.dtype == INT_64 and idx2_o.dtype == INT_64
    
    cdef Py_ssize_t size1 = idx1_i.shape[0]
    cdef Py_ssize_t size2 = idx2_o.shape[0]
#    cdef INT_32_t *arr_Ic_o = <INT_32_t *> malloc(size1 * size2 * sizeof(INT_32_t))
#    cdef INT_32_t *arr_Ic_i = <INT_32_t *> malloc(size1 * size2 * sizeof(INT_32_t))
    cdef INT_64_t[:] arr_Ic_o = np.empty(size1*size2, INT_64)
    cdef INT_64_t[:] arr_Ic_i = np.empty(size1*size2, INT_64)
    cdef INT_32_t iterator = 0
    cdef bint indx1_found_inseg2 = False
    
    #loop over all possible pairs
    for i in range(size1):
        for j in range(size2):
            if (idx1_i[i] == idx2_o[j]):
                arr_Ic_i[iterator] = i
                arr_Ic_o[iterator] = j
                iterator += 1
                indx1_found_inseg2 = True
            elif indx1_found_inseg2: 
                indx1_found_inseg2 = False
                break
            
    #print('edge maching, Ic matrix should be of size of ',iterator)
    return np.concatenate([arr_Ic_i[:iterator,None], arr_Ic_o[:iterator,None]], axis=1)

cpdef _propogate_labels(long n_hits, ar[INT_64_t, ndim=1] outgouing, ar[INT_64_t, ndim=1] incomming, float[:] weight, ar[INT_64_t, ndim=1] labels, ar[INT_32_t, ndim=1] hid, float threshold):
    assert outgouing.dtype == INT_64 and incomming.dtype == INT_64 and labels.dtype == INT_64 and hid.dtype == INT_32

    cdef float THRESHOLD = threshold
    cdef Py_ssize_t size = weight.shape[0]
    cdef int i
    cdef FLOAT_32_t[:] label_weight = np.zeros((n_hits), FLOAT_32)
    labels[incomming[0]] = labels[outgouing[0]]
    label_weight[incomming[0]] = weight[0]
    for i in range(1, size):
        if( (incomming[i] == incomming[i-1]) and (weight[i] < label_weight[incomming[i]])):
            #propogate the label back to incoming node:
            if( (weight[i] > THRESHOLD) and (labels[outgouing[i]] < labels[incomming[i]])): labels[outgouing[i]] = labels[incomming[i]]
            continue
        if( weight[i] < THRESHOLD ): continue # if weight below the threshold, dont propogate labels
        labels[incomming[i]] = labels[outgouing[i]]
        label_weight[incomming[i]] = weight[i]
    return labels

cpdef _get_merge_hits_labels(float[:] phi, float[:] r, float[:] z, int[:] vol):
    cdef Py_ssize_t size1 = phi.shape[0]
    cdef INT_32_t[:] label = np.empty(size1, INT_32)
    cdef int iterator = 1
    cdef int i
    
    label[0] = iterator
    #loop over all possible pairs
    for i in range(size1 - 1):
        if (phi[i+1] - phi[i]) < 0.002:
            if ((vol[i]==8) and (fabs(r[i+1] - r[i])>0.025) and (fabs(z[i+1] - z[i])<10.)): 
                label[i+1] = iterator
                continue
            if ((vol[i]!=8) and (fabs(z[i+1] - z[i])>0.25) and (fabs(r[i+1] - r[i])<2.)): 
                label[i+1] = iterator
                continue
        iterator = iterator + 1
        label[i+1] = iterator
    
    return np.array(label)

    
def get_hitpairs(hits1, hits2, with_truth=False):
    phi1 = hits1.phi.values
    phi2 = hits2.phi.values
    x1 = hits1.x.values
    x2 = hits2.x.values
    y1 = hits1.y.values
    y2 = hits2.y.values
    z1 = hits1.z.values
    z2 = hits2.z.values
    r1 = hits1.r.values
    r2 = hits2.r.values
    if with_truth:
        pid1 = hits1.particle_id.values
        pid2 = hits2.particle_id.values
        seg = pd.DataFrame(data=_get_hitpairs_truth(pid1,pid2,phi1,phi2,x1,x2,y1,y2,z1,z2,r1,r2), columns=['index_1','index_2','label'])
    else:
        seg = pd.DataFrame(data=_get_hitpairs(phi1,phi2,x1,x2,y1,y2,z1,z2,r1,r2), columns=['index_1','index_2'])
    return seg
    
def get_labels(hits_layer):
    phi = hits_layer.phi.values
    r = hits_layer.r.values
    z = hits_layer.z.values
    vol = hits_layer.volume_id.values
    
    return _get_merge_hits_labels(phi,r,z, vol)
        
def get_edgepairs(seg1, seg2):
    #print('total edges in first two segments = ',seg1.shape[0], seg2.shape[0])
    idx1_i = seg1.index_2.values
    idx2_o = seg2.index_1.values
    return pd.DataFrame(data=_get_edgepairs(idx1_i,idx2_o), columns=['index_o','index_i'])

def get_segments(_hits, feature_names = [], with_truth = False):
    
    layer_groups = _hits.groupby('layer')
    count_hits = (0,0)
    #store hits and labels for the 1st layer (hits are merged if close enough)
    hits_layer = [(layer_groups.get_group(1)).sort_values(by=['phi'])]
    labels = [get_labels(hits_layer[0])]
    
    #store hits and labels for the 2nd layer + compute connections (hits are merged if close enough)
    hits_layer.append((layer_groups.get_group(2)).sort_values(by=['phi']))
    count_hits = (count_hits[1],count_hits[1] + hits_layer[0].shape[0])
    labels.append(get_labels(hits_layer[1])+count_hits[1])
    seg = get_hitpairs(hits_layer[0], hits_layer[1], with_truth=with_truth)
    seg.index_2 += count_hits[1]
    Is_seg = [seg]
    
    #store hits and labels for the 3rd layer + compute connections (hits are merged if close enough)
    hits_layer.append((layer_groups.get_group(3)).sort_values(by=['phi']))
    seg = get_hitpairs(hits_layer[1], hits_layer[2], with_truth=with_truth)
    count_hits = (count_hits[1],count_hits[1] + hits_layer[1].shape[0])
    labels.append(get_labels(hits_layer[2])+count_hits[1])
    seg.index_1 += count_hits[0]
    seg.index_2 += count_hits[1]
    Is_seg.append(seg)
    
    #loop over rest of the layers store hits and compute connections (no hit merging applied)
    n_layers = np.array([int(k) for k in layer_groups.groups]).max()
    for layer in range(3, n_layers):
        hits_layer.append(layer_groups.get_group(layer+1).sort_values(by=['phi']))
        seg = get_hitpairs(hits_layer[layer-1], hits_layer[layer], with_truth=with_truth)
        count_hits = (count_hits[1],count_hits[1] + hits_layer[layer-1].shape[0])
        labels.append(np.arange(count_hits[1],count_hits[1]+hits_layer[layer].shape[0]))
        seg.index_1 += count_hits[0]
        seg.index_2 += count_hits[1]
        Is_seg.append(seg)

    
    all_hits = pd.concat(hits_layer)
    all_sigments = pd.concat(Is_seg)
    X_seg = all_hits[feature_names].values
    Is = all_sigments[['index_1','index_2']]
    hit_id_seg = all_hits.hit_id.values
    labels = np.concatenate(labels)
    if with_truth: yseg = all_sigments.label.values
        
    if with_truth: return yseg, X_seg, Is, hit_id_seg, labels
    return X_seg, Is, hit_id_seg, labels

def propogate_labels(n_nodes, edges, labels, hits_id, threshold):
    if threshold < 0:
        return _propogate_labels(n_nodes, edges.index_2.values, edges.index_1.values, edges.weight.values.astype(np.float32), labels, hits_id, -threshold)
    else:
        return _propogate_labels(n_nodes, edges.index_1.values, edges.index_2.values, edges.weight.values.astype(np.float32), labels, hits_id, threshold)
    
def preprocess(hits, eta_bins, feature_names = ['x', 'y', 'z', 'phi', 'eta', 'r']):
       
    hits['phi'] = FLOAT_32(1)
    hits['r'] = FLOAT_32(1)
    hits['eta'] = FLOAT_32(1)
    hits['layer'] = INT_32(1)
    x = hits.x.values
    y = hits.y.values
    z = hits.z.values
    phi = hits.phi.values
    r = hits.r.values
    eta = hits.eta.values
    layer = hits.layer.values
    volume_id = hits.volume_id.values
    layer_id = hits.layer_id.values
    _augmentation(x, y, z, phi, r, eta, layer, volume_id, layer_id)
    
    
    Xs = []
    Is = []
    hits_ids = []
    labels1 = []
    ys = []
     
    for i_eta in range(len(eta_bins)-1):  

        _hits = (hits.loc[(hits['eta']>eta_bins[i_eta]) & (hits['eta']<=eta_bins[i_eta+1])])
   
        #order acording to the layer number and reindex
        sorted_hits = (_hits.sort_values(by=['layer'])).reset_index()
        
        #X_seg, Is_seg, hit_id_seg, labels_layer1 = get_segments(sorted_hits, feature_names = feature_names)
        y, X_seg, Is_seg, hit_id_seg, labels_layer1 = get_segments(sorted_hits, feature_names = feature_names, with_truth = True)
                
        #append inputs (as numpy ndarrays )
        Xs.append(X_seg)
        Is.append(Is_seg)
        labels1.append(labels_layer1)
        hits_ids.append( hit_id_seg )
        ys.append( y )
        
        #print('total number of segments = ',segments.shape[0])

    
    return ys, Xs, Is, hits_ids, labels1
    #return Xs, Is, hits_ids, labels1

def weights_to_labels(X, Is, weights, labels, hits_id, threshold = 0.5):

    n_nodes = X.shape[0]

    weight, edges = weights, Is
    edges['weight'] = weight
    labels = propogate_labels(n_nodes, edges.sort_values(by='index_2'), labels, hits_id, threshold)
    #propogate once more backwards to collect more hits per track
    #labels = propogate_labels(n_nodes, edges.sort_values(by='index_1', ascending=False), labels, hits_id, -threshold)

    return labels