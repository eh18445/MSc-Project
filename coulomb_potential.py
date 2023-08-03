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

def atomicUnit(x,y,z,R,Ry,Rz,Z):
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
    fi_r1, orbArray1 = atomicAct(polarVec1,Z) 

    # -- 
    x2 = x + R; 
    y2 = y + Ry; z2 = z + Rz        
    rVec2 = torch.cat((x2,y2,z2),1)
    
    r2 = toR(rVec2)         
    theta2 = toTheta(rVec2)
    phi2 = toPhi(rVec2)
    polarVec2 = torch.cat((r2,theta2,phi2),1)
    
    fi_r2, orbArray2 = atomicAct(polarVec2,Z)

    orbArray = torch.cat((orbArray1,orbArray2),1)
    
    return fi_r1, fi_r2, orbArray

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
    phi = torch.sgn(rVecPolar[:,1])*torch.arccos(rVecPolar[:,0]/(torch.pow(rVecPolar[:,0],2)+torch.pow(rVecPolar[:,1],2)))
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
        raise ValueError("orbital_name invalid. A value of {} was entered. Allowed inputs:".format(orbital_name)+
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
            orb = orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[0])
            AO_sum = AO_sum.add(orb)
            
            #orbArray will contain all the orbitals so the can be used to calculate the coulomb potential
            if i == 0:
                orbArray = orb
            else:
                orbArray = torch.cat((orbArray,orb),1)
        elif i <= 3:
            #2s
            orb = orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[1])
            AO_sum = AO_sum.add(orb)
            orbArray = torch.cat((orbArray,orb),1)
        elif i == 4 or i == 7:
            #2pz
            orb = orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[2])
            AO_sum = AO_sum.add(orb)
            orbArray = torch.cat((orbArray,orb),1)
        elif i == 5 or i == 8:
            #2px
            orb = orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[3])
            AO_sum = AO_sum.add(orb)
            orbArray = torch.cat((orbArray,orb),1)
        elif i == 6 or i == 9:
            #2py
            orb = orbital(polarVec[:,0],polarVec[:,1],polarVec[:,2],Z,orbital_name=orbital_list[4])
            AO_sum = AO_sum.add(orb)
            orbArray = torch.cat((orbArray,orb),1)   
    
    return  AO_sum, orbArray

#construct vectors for x,y,z,R
n_points = 100000
x = torch.linspace(-18,18,n_points,requires_grad=False)
y = torch.linspace(-18,18,n_points,requires_grad=False)
z = torch.linspace(-18,18,n_points,requires_grad=False)
R = torch.linspace(0.2,4,n_points,requires_grad=False)

x = x.reshape(-1,1); y = y.reshape(-1,1); z = z.reshape(-1,1); R = R.reshape(-1,1)

Z = 8 
Ry = 0
Rz = 0

fi_r1, fi_r2, orbArray = atomicUnit(x,y,z,R,Ry,Rz,Z)

def radial(x,y,z,R,Ry,Rz):
    """
    Returns the radial part from cartesian coordinates
    """
    Rx = R
    Ry = Ry
    Rz = Rz
    r1 = torch.sqrt((x-Rx).pow(2)+(y-Ry).pow(2)+(z-Rz).pow(2))
    r2 = torch.sqrt((x+Rx).pow(2)+(y+Ry).pow(2)+(z+Rz).pow(2))
    return r1, r2

def V(x,y,z,R,Ry,Rz,orbArray):
    """
    Potential energy function.
    For each electron calculate coulomb potential from all other electrons
    """
    #distance from atom to x,y,z values
    r1,r2 = radial(x,y,z,R,Ry,Rz)
    
    #nuclear interaction
    potential = -Z/r1 -Z/r2
    
    #repulsion from other electrons
    #Hartree Fock integral
    # SS |Chi_i|^2 * 1/r_ij * |Chi_j|^2 dr_i dr_j for each orbital chi_i and chi_j
    #Needs to be calculated for each orbital in molecule
    
    #r_ij is distance between r_i and r_j
    r_ij = torch.abs(r1-r2)
    r1 = r1.detach().numpy()
    r2 = r2.detach().numpy()

    for i in range(1,orbArray.shape[1]):
        for j in range(0,i):
            chi_i = orbArray[:,i].abs().pow(2)
            chi_j = orbArray[:,j].abs().pow(2)
            chi_i = chi_i.reshape(-1,1); chi_j = chi_j.reshape(-1,1)
            chi_i = chi_i.cpu(); chi_j = chi_j.cpu()
            f = chi_i * 1/r_ij * chi_j
            
            #needs to integrate over r1 and r2
            f = f.detach().numpy()
            f2 = simps(f, r1)
            f2 = f2.reshape(-1,1)
            potential += simps(f2,r2).reshape(-1,1)            
            
    return potential

x = x.cpu(); y = y.cpu(); z = z.cpu(); R = R.cpu()
potential = V(x,y,z,R,Ry,Rz,orbArray)




