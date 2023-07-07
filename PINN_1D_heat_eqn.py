# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:53:55 2023

@author: danie
"""

import torch
from torch.autograd import grad
import numpy as np
import torch.nn as nn

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

class PINN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PINN,self).__init__()
            
        #the neural network layers go here

    def forward(self,x):
        #x is input vector
        
        #neural network goes here    
        
        return

    def f_nn(self,t,x):
        """
        Calculates:
            f = c*du/dt - d/dx(k*du/dx) - s
              = c*du/dt - dk/du*du/dx*du/dx - k*d**2u/dx**2 - s
        """
    
        u = self.u_nn(t,x)
        u_t = get_derivative(u,t,1)
        u_x = get_derivative(u,x,1)
        u_xx = get_derivative(u,x,2)
        k = 0.01 * u + 7
        k_u = 0.01
        c = 0.0005 * u**2 + 500
        s = self.source_term(t,x)
        f = c * u_t - k_u * u_x * u_x - k * u_xx - s
    
        return f

    def source_term(self,t,x):
        """
        returns the source term s according to:
        s = ku/sigma**2 + u(x-p)/sigma**2 * [c*dp/dt - (x-p)/sigma**2 * (k+u*dk/du)]
        """
    
        t_max = 0.5
        sigma = 0.02
        u_max = 1
        p = 0.25 * torch.cos(2*np.pi*t/t_max) + 0.5
        p_t = -0.5 * torch.sin(2*np.pi*t/t_max) * np.pi/t_max
        u_sol = u_max * torch.exp(-(x-p)**2/(2*sigma**2))
        k_sol = 0.01 * u_sol + 7
        k_u_sol = 0.01
        c_sol = 0.0005 * u_sol**2 + 500
        factor = 1/(sigma**2)
        s = factor * k_sol * u_sol + u_sol * (x-p) * factor * (c_sol * p_t - (x-p) * factor * (k_sol + u_sol * k_u_sol))
    
        return s

    def cost_function(self,tO,xO,t_lb,x_lb,t_ub,x_ub,t_f,x_f,uO):
        """
        Calculates value of the cost function
        C = MSE_O + MSE_b + MSE_f
        """
    
        uO_pred = self.u_nn(tO,xO)
        u_lb_pred = self.u_nn(t_lb,x_lb)
        u_x_lb_pred = get_derivative(u_lb_pred,x_lb,1)
        u_ub_pred = self.u_nn(t_ub,x_ub)
        u_x_ub_pred = get_derivative(u_ub_pred,x_ub,1)
        f_pred = self.f_nn(t_f,x_f)
        mse_O = torch.mean((uO-uO_pred)**2)
        mse_b = torch.mean(u_x_lb_pred**2) + torch.mean(u_x_ub_pred**2)
        mse_f = torch.mean((f_pred)**2)
    
        return mse_O, mse_b, mse_f




