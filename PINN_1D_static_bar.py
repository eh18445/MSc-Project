# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:25:36 2023

@author: danie
"""

import torch
from torch.autograd import grad
import torch.nn as nn
import math

#Create neural network to predict displacement u(x)

def buildModel(input_dim,hidden_dim,output_dim):
    """
    A single hidden layer of dimensions hiddenm_dim is used.
    The input and output dimensions are given as input_dim and output_dim respectively.
    Linear transformation functions are used and hyperbolic tangent activation functions.
    No activation function in the final layer so the range of outputs is nmopt limited. 
    """
    
    model = torch.nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim,output_dim))
    
    return model

#The first part of the PINN is defined.
#A displacement prediction u at x=0.5 can be made

model = buildModel(1,10,1)
u_pred = model(torch.tensor([0.5]))
print('Displacement prediction of x = 0.5:',u_pred)

#Second part of the network requires computation of derivatioves wrt x
#Uses pytorch's automatic differentiation 

def get_derivative(y,x):
    """
    Returns derivative using pytorch automatic differentiation
    """
    
    dydx = grad(y,x,torch.ones(x.size()[0],1),
                create_graph=True,
                retain_graph=True)
    
    return dydx

def f(model,x,EA,p):
    """
    f=d/dx(EA*du/dx) + p is calculated
    """
    
    u = model(x)
    u_x = get_derivative(u,x)
    EAu_xx = get_derivative(EA(x)*u_x,x)
    f = EAu_xx + p(x)
    
    return f

#Calculate the loss function: MSE_f
#Example values are inserted into the model

#model = buildModel(1,10,1)
x = torch.linspace(0,1,10,requires_grad=True).view(-1,1)
EA = lambda x: 1+ 0*x
p = lambda x: 4 * math.pi**2 * torch.sin(2*math.pi*x)

f = f(model,x,EA,p)
MSE_f = torch.sum(f**2)

#Calculate the cost function using MSE_b

#model = buildModel(1,10,1)
u0 = 10
u1 = 0

u0_pred = model(torch.tensor([0.]))
u1_pred = model(torch.tensor([1.]))

MSE_b = (u0_pred-u0)**2 + (u1_pred-u1)**2






 
