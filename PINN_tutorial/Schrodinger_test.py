# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:52:01 2023

@author: danie
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from scipy.stats import qmc

def get_derivative(y,x,n):
    """
    Returns derivative using pytorch automatic differentiation
    """    
    if n == 0:
        return y
    
    else:
        dydx = grad(y,x,torch.ones_like(y),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True)
        return get_derivative(dydx,x,n-1)
    
class SchrodingerPINN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(SchrodingerPINN,self).__init__()
        
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.tanh = nn.Tanh()
        
    def forward(self,x,t):
        
        out_x = self.tanh(self.l1(x))
        out_x = self.tanh(self.l2(out_x))
        out_x = self.tanh(self.l2(out_x))
        out_x = self.tanh(self.l2(out_x))
        out_x = self.tanh(self.l3(out_x))
        
        out_t = self.tanh(self.l1(t))
        out_t = self.tanh(self.l2(out_t))
        out_t = self.tanh(self.l2(out_t))
        out_t = self.tanh(self.l2(out_t))
        out_t = self.tanh(self.l3(out_t))
        
        h = torch.cat((out_x,out_t),1)
        
        return h
    
