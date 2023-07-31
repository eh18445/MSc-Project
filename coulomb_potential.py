# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:28:38 2023

@author: Daniel
"""

#AIM: calculate electronic repulsion potential

#using Hartree-Fock coulomb integral

#Using example atom calculate integral value

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import time
# import copy

from os import path
import pickle
from scipy.integrate import simps
  
import warnings
warnings.filterwarnings('ignore')

dtype = torch.double    
torch.set_default_tensor_type('torch.DoubleTensor')

lineW = 3
lineBoxW = 2

font = {'size': 11}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 12
plt.rcParams.update({"text.usetex": True})
# plt. close('all')

# Check to see if gpu is available. If it is, use it else use the cpu
if torch.cuda.is_available():
    # device = torch.device('cuda')
    device = torch.device("cuda:0") 
    print('Using ', device, ': ', torch.cuda.get_device_name())   
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)  
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')
    print('No GPU found, using cpu')
    
def integra3d(x,y,z,f):   
    # 3d integration using Simpson method of scipy
    f = f.detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    z = z.detach().numpy()
    I = simps( [simps( [simps(fx, x) for fx in fy], y) for fy in f ]  ,z)
    return I

def atomicUnit(self,x,y,z,R,Ry,Rz,Z):
    """
    Takes the 2 inputs r and R.
    Returns the hydrogen atomic s-orbitals for each ion.
    """
    # Cartesian Translation & Scaling: 
    x1 = x - R
    y1 = y - Ry 
    z1 = z - Rz        
    rVec1 = torch.cat((x1,y1,z1),1)
    
    #parameters for orbital
    r1 = toR(rVec1)
    theta1 = toTheta(rVec1)
    phi1 = toPhi(rVec1)
    polarVec1 = torch.cat((r1,theta1,phi1),1)
    
    # ATOMIC ORBITAL ACTIVATION
    fi_r1 = atomicAct(polarVec1,Z) 

    # -- 
    x2 = x + R; 
    y2 = y + Ry; z2 = z + Rz        
    rVec2=torch.cat((x2,y2,z2),1)
    
    r2 = toR(rVec2)         
    theta2 = toTheta(rVec2)
    phi2 = toPhi(rVec2)
    polarVec2 = torch.cat((r2,theta2,phi2),1)
    
    fi_r2 = atomicAct(polarVec2,Z)
    
    return fi_r1, fi_r2

def lcao_solution(self,fi_r1,fi_r2):
    """
    LCAO solution: Linear combination
    Psi_LCAO = fi_r1 +/- fi_r2
    Use + version for symmetric which we do if self.P=1
    """
    N_LCAO = (fi_r1 + self.P*fi_r2)
    
    #take only the real part of LCAO
    N_LCAO = N_LCAO.real
    N_LCAO = N_LCAO.type(torch.DoubleTensor)
    
    return N_LCAO

def toR(rVecPolar):
    r2 = rVecPolar[:,0].pow(2) + rVecPolar[:,1].pow(2) + rVecPolar[:,2].pow(2)
    r = torch.sqrt(r2) 
    r = r.reshape(-1,1)
    return r

def toTheta(rVecPolar):
    r2 = rVecPolar[:,0].pow(2) + rVecPolar[:,1].pow(2) + rVecPolar[:,2].pow(2)
    r = torch.sqrt(r2)
    theta = torch.arccos(rVecPolar[:,2]/r)
    theta = theta.reshape(-1,1)
    return theta
    
def toPhi(rVecPolar):
    phi = torch.sgn(rVecPolar[:,1])*torch.arccos(rVecPolar[:,0]/(torch.pow(rVecPolar[:,0],2)+torch.pow(input[:,1],2)))
    phi = phi.reshape(-1,1)
    return phi

def orbital(r,theta,phi,Z,orbital_name):
    """
    Inputs r,theta,phi polar co-ords.
    With atom located at centre.
    Returns the orbital shape given the orbital name.
    """
    
    if orbital_name == '1s':
        chi = Z**(3/2)*torch.exp(-r*Z)
    elif orbital_name == '2s':
        chi = Z**(3/2)*(2-(r*Z))*torch.exp(-r*Z/2)
    elif orbital_name == '2pz':
        chi = Z**(3/2)*(r*Z)*torch.exp(-r*Z/2)*torch.cos(theta)
    elif orbital_name == '2py' or orbital_name == '2px':
        chi = Z**(3/2)*(r*Z)*torch.exp(-r*Z/2)*torch.sin(theta)
        if orbital_name == '2px':
            chi = chi.mul(torch.exp(phi*1j))
        else:
            chi = chi.mul(torch.exp(-1*phi*1j))
    elif orbital_name == '3s':
        chi = Z**(3/2)*(27-18*(r*Z)+2*torch.pow(r*Z,2))*torch.exp(-r*Z/3)
    elif orbital_name == '3pz':
        chi = Z**(3/2)*(6*r-torch.pow(r*Z,2))*torch.exp(-r*Z/3)*torch.cos(theta)
    elif orbital_name == '3py' or orbital_name == '3px':
        chi = Z**(3/2)*(6*r-torch.pow(r*Z,2))*torch.exp(-r*Z/3)*torch.sin(theta)
        if orbital_name == '2px':
            chi = chi.mul(torch.exp(phi*1j))
        else:
            chi = chi.mul(torch.exp(-1*phi*1j))
    elif orbital_name == '3dz2':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*(3*torch.cos(theta)**2-1)
    elif orbital_name == '3dyz' or orbital_name == '3dxz':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*torch.sin(theta)*torch.cos(theta)
        if orbital_name == '3dxz':
            chi = chi.mul(torch.exp(phi*1j))
        else: 
            chi = chi.mul(torch.exp(-1*phi*1j))
    elif orbital_name == '3dxy' or orbital_name == '3dx2y2':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*torch.sin(theta)**2
        if orbital_name == '3dx2y2':
            chi = chi.mul(torch.exp(phi*1j))
        else: 
            chi = chi.mul(torch.exp(-1*phi*1j))
    else:
        raise Exception("orbital_name invalid. A value of {} was entered. Allowed inputs:".format(orbital_name)+
                        "'1s', '2s', '2px', '2py, '2pz', '3s', '3px', '3py', '3pz', '3dz2', '3dyz', '3dxz', '3dxy', '3dx2y2'.")
    chi = chi.reshape(-1,1)

    return chi

def atomicAct(polarVec,Z):
    """
    Input vector in polar co-ordinates.
    Sums together the different atomic orbitals for the atom.
    Returns atomic orbitals for atom.
    """

    orbital_list = ['1s','2s','2pz','2px','2py','3s','3pz','3py','3px','3dz2','3dyz','3dxz','3dxy','3dx2y2']
    
    AO_sum = torch.zeros(len(polarVec))
    AO_sum = AO_sum.reshape(-1,1)
    
    #fill Z electron orbitals
    #!!!!!!!!!!!!!!!Only works up to Z=9 currently
    for i in range(Z):    
        if i <= 1: 
            #1s
            AO_sum = AO_sum.add(orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[0]))
        elif i <= 3:
            #2s
            AO_sum = AO_sum.add(orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[1]))
        elif i == 4 or i == 7:
            #2pz
            AO_sum = AO_sum.add(orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[2]))
        elif i == 5 or i == 8:
            #2px
            AO_sum = AO_sum.add(orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[3]))
        elif i == 6 or i == 9:
            #2py
            AO_sum = AO_sum.add(orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[4]))
    
    return  AO_sum

#construct vectors for x,y,z,R
