# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:12:18 2023

@author: danie
"""

#Create function that outputs the LCAO solutions for different atomic or molecular setups

#First take the inputs of 2 atoms
#input is r(position) and R output is Psi_LCAO

#sum the overlapping orbitals together to get the outermost bonding

#############################
#work out what the output of original code is doing
#aim to achieve the same LCAO output of the original code

import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import time

font = {'size': 11}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = 12

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
    
def toR(vec):
    r2 = vec[:,0].pow(2) + vec[:,1].pow(2) + vec[:,2].pow(2)
    r = torch.sqrt(r2)
    r = r.reshape(-1,1)
    return r
    
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
            chi *= torch.exp(torch.tensor([phi*1j]))
        else:
            chi *= torch.exp(torch.tensor([-1*phi*1j]))
    elif orbital_name == '3s':
        chi = Z**(3/2)*(27-18*(r*Z)+2*torch.pow(r*Z,2))*torch.exp(-r*Z/3)
    elif orbital_name == '3pz':
        chi = Z**(3/2)*(6*r-torch.pow(r*Z,2))*torch.exp(-r*Z/3)*torch.cos(theta)
    elif orbital_name == '3py' or orbital_name == '3px':
        chi = Z**(3/2)*(6*r-torch.pow(r*Z,2))*torch.exp(-r*Z/3)*torch.sin(theta)
        if orbital_name == '2px':
            chi *= torch.exp(torch.tensor([phi*1j]))
        else:
            chi *= torch.exp(torch.tensor([-1*phi*1j]))
    elif orbital_name == '3dz2':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*(3*torch.cos(theta)**2-1)
    elif orbital_name == '3dyz' or orbital_name == '3dxz':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*torch.sin(theta)*torch.cos(theta)
        if orbital_name == '3dxz':
            chi *= torch.exp(torch.tensor([phi*1j]))
        else: 
            chi *= torch.exp(torch.tensor([-1*phi*1j]))
    elif orbital_name == '3dxy' or orbital_name == '3dx2y2':
        chi = Z**(3/2)*torch.pow(r*Z,2)*torch.exp(-r*Z/3)*torch.sin(theta)**2
        if orbital_name == '3dx2y2':
            chi *= torch.exp(torch.tensor([phi*1j]))
        else: 
            chi *= torch.exp(torch.tensor([-1*phi*1j]))
    else:
        raise Exception("orbital_name invalid. A value of {} was entered. Allowed inputs:".format(orbital_name)+
                        "'1s', '2s', '2px', '2py, '2pz', '3s', '3px', '3py', '3pz', '3dz2', '3dyz', '3dxz', '3dxy', '3dx2y2'.")
            
    return chi

def atomicUnit(x,y,z,Rx,Ry,Rz,Z):
    """
    Takes the 2 inputs r and R.
    Z is the atomic number of the atom.
    R is the position of the atom.
    Returns the hydrogen atomic orbitals for the atom.
    """
    
    time_start = time.time()
    
    #convert cartesian co-ordinates to polar
    #calculate the orbital
    #convert back to cartesian
    
    #cartesian translation and scaling
    x1 = x - Rx
    y1 = y - Ry
    z1 = z - Rz
    
    #print(x)
                               
    rVec1 = torch.cat((x1,y1,z1),1)
    print(rVec1)
    #print(rVec0)
    
    #convert to radius
    r1 = toR(rVec1)
    
    #covert to polar co-ords
    theta1 = torch.arccos(z1/r1)
    phi1 = torch.sgn(y1)*torch.arccos(x1/(torch.pow(x1,2)+torch.pow(y1,2)))
        
    print(theta1)
    print(phi1)
    
    phi_r1 = orbital(r1,theta1,phi1,Z,orbital_name='1s')
    
    print('LCAO Runtime (min):',(time.time()-time_start)/60)
    
    return phi_r1

def LCAO_Solution():
    """
    Linearly combine atomic units.
    """
    return


##############functions from original code
def actAO_s(r):
    return  torch.exp(-r)

def atomicUnit_original(x,y,z,Rx,Ry,Rz):
    """
    Takes the 2 inputs r and R.
    Returns the hydrogen atomic s-orbitals for each ion.
    """
    x1 = x - Rx; 
    #y1 = y - Ry 
    #z1 = z - Rz     # Cartesian Translation & Scaling: 
    #rVec1 = torch.cat((x1,y1,z1),1)
    #r1 = toR(rVec1) 
    print(x1)
    fi_r1 = actAO_s(x1);  # s- ATOMIC ORBITAL ACTIVATION
    #print(x1)
    #print(rVec1)
    #print(r1)
    #print(fi_r1)
    # -- 
    x2 = x + Rx; 
    #y2 = y + Ry 
    #z2 = z + Rz        
    #rVec2=torch.cat((x2,y2,z2),1)
    #r2 = toR(rVec2);         
    fi_r2 = actAO_s(x2)
    
    return fi_r1, fi_r2

#Rx, Ry and Rz are positions of atom
Rx1 = -4
Ry1 = 0
Rz1 = 0

Rx2 = 4
Ry2 = 0
Rz2 = 0

x = torch.linspace(-10,10,1000)
y = torch.linspace(-10,10,1000)
z = torch.linspace(-10,10,1000)

Z = 1

x,y,z = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)

xg, yg, zg = torch.meshgrid(x,y,z)
#xgg = xg.reshape(-1,1) 
#ygg = yg.reshape(-1,1)
#zgg = zg.reshape(-1,1)
print(xg)

phi1 = atomicUnit(x,y,z,Rx1,Ry1,Rz1,Z)
#print(phi1)
phi2 = atomicUnit(x,y,z,Rx2,Ry2,Rz2,Z)

#add and normalise
Psi = (phi1 + phi2)#*(1/np.sqrt(2))

#plt.plot(x.cpu(),phi1.cpu())
#plt.xlim(-5,5)
#plt.ylabel('$|\Psi|$')
#plt.xlabel('x')
#plt.show()

plt.plot(x.cpu(),Psi.cpu())
plt.title('New')
plt.xlim(-5,5)
plt.ylabel('$|\Psi|$')
plt.xlabel('x')
plt.show()







phi1,phi2 = atomicUnit_original(x,y,z,Rx1,Ry1,Rz1)
#print(phi1)

#add and normalise
Psi = (phi1 + phi2)#*(1/np.sqrt(2))

plt.plot(x.cpu(),phi1.cpu())
plt.title('Original')
#plt.xlim(-5,5)
plt.ylabel('$|\Psi|$')
plt.xlabel('x')
plt.show()




