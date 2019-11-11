"""
This file contains custom CNN architecture used for the Calorimeter images with varying granularity
author: Michael Pitt (michael.pitt@cern.ch)
date: November 2019
"""

import torch
from torch import nn
import torch.nn.functional as F
from operator import mul
import numpy as np
       
class model(nn.Module):
    def __init__(self, LR_shapes, HR_shape, layer_number, kernel_size = (3,3), padding = (1,1), debug = False):
        super(model, self).__init__()
        
        self.upscale_factor = (int(HR_shape[0]/LR_shapes[layer_number][0]),int(HR_shape[1]/LR_shapes[layer_number][1]))
        self.output_dim = self.upscale_factor[0]*self.upscale_factor[1]
        self.kernel = kernel_size
        self.padding = padding
        self.nLayers = len(LR_shapes)
        self.LR_shapes = LR_shapes #shapes of all layers
        self.ls0 = LR_shapes[layer_number]
        self.layer_number = layer_number
        self.debug = debug
        
        #construct weight matrix depend on the layer shape
        if self.output_dim>1:
            weights_input_dim = 0
            for j_layer in range(self.nLayers):
                neigh_Npix = int(LR_shapes[j_layer][0]/LR_shapes[layer_number][0])*int(LR_shapes[j_layer][1]/LR_shapes[layer_number][1])
                weights_input_dim += neigh_Npix * np.prod(kernel_size)
            self.weight = nn.Parameter(torch.rand(weights_input_dim,self.output_dim))
            stdv = 1. / (weights_input_dim * self.output_dim)
            self.weight.data.uniform_(-stdv, stdv)
			
            if debug: print('Model initialized, for L=%d, with weight matrix of size = '%layer_number,self.weight.size())
        
    def forward(self, input):
        
        # selected layer to apply upscaling
        Xsel = input[self.layer_number]

        #if no upscaling needed, return the layer as it is
        if self.output_dim==1:
            return Xsel
        #convolute the weight matrix with the input layers
        Xunfold = []
        for j_layer in range(self.nLayers):
            s = (int(self.LR_shapes[j_layer][0]/self.ls0[0]),
                int(self.LR_shapes[j_layer][1]/self.ls0[1]))
            if not np.prod(s): continue #skip layer with smaller granularity
            f = tuple(map(mul, self.kernel, s))
            p = tuple(map(mul, self.padding, s))
            Xunfold.append(F.unfold(input[j_layer],kernel_size=f,padding=p,stride=s))
        Xunfold = torch.cat(Xunfold,dim=1)
        out_unf = Xunfold.transpose(1, 2).matmul(self.weight)
        
        #apply reLU
        out_unf = F.relu(out_unf)
        
        #apply softmax on the output dimention
        out_unf_soft = out_unf.softmax(dim=-1)
        out_unf_soft = out_unf_soft[:,:,:,None].transpose(2, 3)
        
        #multiply the input layer with the output weights
        out = F.unfold(Xsel,kernel_size=1).transpose(1, 2)[:,:,:,None]
        out = out.matmul(out_unf_soft)
        out = out.squeeze(2).transpose(1, 2)
        
        #fold back to the input image shape, and expland to the HR image
        out = F.fold(out,self.ls0,1)
        out = self.pixel_shuffle(out)
        return out
    
    def pixel_shuffle(self, input):
        if len(input.shape)<4:
            raise ValueError('Error input layer for shuffling, expect 4D, got'+str(input.shape))
        n = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        if input.shape[1]!=self.upscale_factor[0]*self.upscale_factor[1]:
            raise ValueError('Error input dim,',input.shape,' will not work with upscale_factor',self.upscale_factor)
        #output
        oh = h * self.upscale_factor[0];
        ow = w * self.upscale_factor[1];
        input_reshaped = input.reshape((n, 1, self.upscale_factor[0], self.upscale_factor[1], h, w))
        return input_reshaped.permute(0,1,4,2,5,3).reshape((n,1,oh,ow))        

    