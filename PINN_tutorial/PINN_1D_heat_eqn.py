# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:53:55 2023

@author: danie
"""

import torch
from torch.autograd import grad
import numpy as np
import torch.nn as nn
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

class PINN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PINN,self).__init__()
            
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.tanh = nn.Tanh()

    def u_nn(self,t,x):
        """
        Neural Network approximating u
        """
        
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
        
        u = torch.cat((out_x,out_t),1)
        
        print(u)
        
        return u

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
        returns the manufactured solution s according to:
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

    def cost_function(self,t0,x0,t_lb,x_lb,t_ub,x_ub,t_f,x_f,u0):
        """
        Calculates value of the cost function
        C = MSE_O + MSE_b + MSE_f
        """
    
        u0_pred = self.u_nn(t0,x0)
        u_lb_pred = self.u_nn(t_lb,x_lb)
        u_x_lb_pred = get_derivative(u_lb_pred,x_lb,1)
        u_ub_pred = self.u_nn(t_ub,x_ub)
        u_x_ub_pred = get_derivative(u_ub_pred,x_ub,1)
        f_pred = self.f_nn(t_f,x_f)
        #enforces the initial conditon
        mse_0 = torch.mean((u0-u0_pred)**2)
        #enforces Neumann boundary conditions
        mse_b = torch.mean(u_x_lb_pred**2) + torch.mean(u_x_ub_pred**2)
        #enforces governing equations
        mse_f = torch.mean((f_pred)**2)
    
        return mse_0, mse_b, mse_f

model = PINN(1,10,1)

#x and t range
x = torch.linspace(0,1,10,requires_grad=True).view(-1,1)
t = torch.linspace(0,0.5,10,requires_grad=True).view(-1,1)

print(x,t)

#collocation samples
N_f = 100
sampler = qmc.LatinHypercube(d=N_f)
collocation = sampler.random(n=1)[0]
print(collocation)

#hyperparameters
learning_rate = 0.01
optimiser1 = torch.optim.Adam(model.parameters(),lr=learning_rate)
optimiser2 = torch.optim.LBFGS(model.parameters(),lr=learning_rate)

#training
num_epochs = 5000
for epoch in range(num_epochs):
    
    model.u_nn(t,x)
    
    if (epoch+1) % 100 == 0:
        print(epoch+1)