"""
This code contains several useful functions that are using in this project
These functions used to help to visualize the data.
"""


import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

def AnalyzeThreshold(test_pred, test_target, log=True):
    thresh=0.5
    y_true = (test_target > thresh)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, test_pred)
    
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,5))

    # Plot the model outputs
    binning=dict(bins=50, range=(0,1), histtype='bar', log=log)
    ax0.hist(test_pred[test_target<thresh], label='fake', **binning, alpha=0.7)
    ax0.hist(test_pred[test_target>thresh], label='true', **binning, alpha=0.7)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot the ROC curve
    auc = sklearn.metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve, AUC = %.3f' % auc)

    plt.tight_layout()


def MaskXYZ(x, phi_range):
    return (np.arctan2(x[:,1],x[:,0]) > phi_range[0]) & (np.arctan2(x[:,1],x[:,0]) < phi_range[1]) 

def draw_sample(inputs, yhat, y, cmap='bwr_r', figsize=(15, 7)):
    
    #set filter criteria:
    phi_range = (-np.pi,np.pi)
    #get the data:
    X, Is, _ = inputs
    X = X.squeeze()
    # Select the i/o node features for each segment
    feats_o = X[Is[:,0]]
    feats_i = X[Is[:,1]]

    #mask the tensors:
    mask_x = MaskXYZ(X,phi_range)
    
    mask_o = MaskXYZ(feats_o,phi_range)
    mask_i = MaskXYZ(feats_i,phi_range)
    mask_y = mask_o & mask_i
    
    X = X[mask_x]
    yhat = yhat[mask_y]
    y = y[mask_y]
    feats_o = feats_o[mask_y]
    feats_i = feats_i[mask_y]
    
    
    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap(cmap)
    
    # Draw the hits (r, z)
    ax0.scatter(X[:,2], np.sqrt(X[:,0]**2+X[:,1]**2), c='k')
    ax1.scatter(X[:,2], np.sqrt(X[:,0]**2+X[:,1]**2), c='k')

    # Draw the correct segments
    print('plot stats: ',X.shape[0],'nodes',y.shape[0],'edges to draw')
    for j in range(y.shape[0]):
        seg_args_y = dict(c='k', alpha=float(y[j]))
        seg_args_yhat = dict(c=cmap(float(yhat[j])))
        ax0.plot([feats_o[j,2], feats_i[j,2]],
                 [np.sqrt(feats_o[j,0]**2+feats_o[j,1]**2), np.sqrt(feats_i[j,0]**2+feats_i[j,1]**2)], '-', **seg_args_y)
        ax1.plot([feats_o[j,2], feats_i[j,2]],
                 [np.sqrt(feats_o[j,0]**2+feats_o[j,1]**2), np.sqrt(feats_i[j,0]**2+feats_i[j,1]**2)], '-', **seg_args_yhat)
    

    # Adjust axes
    ax0.set_title('Truth labels',fontsize=20)
    ax0.set_xlabel('$z$',fontsize=20)
    ax1.set_xlabel('$z$',fontsize=20)
    ax0.set_ylabel('$R$',fontsize=20)
    ax1.set_ylabel('$R$',fontsize=20)
    ax1.set_title('Predicted labels',fontsize=20)
    plt.tight_layout()
  