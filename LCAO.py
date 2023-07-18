# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:12:18 2023

@author: danie
"""

#Create function that outputs the LCAO solutions for different atomic or molecular setups

#First take the inputs of 2 atoms

import torch
    
def toR(vec):
    r2 = vec[:,0].pow(2) + vec[:,1].pow(2) + vec[:,2].pow(2)
    r = torch.sqrt(r2)
    r = r.reshape(-1,1)
    return r
    
def atomicAct_s(r):
    return torch.exp(-r)

def atomicUnit(x,y,z,Rx,Ry,Rz):
    """
    Takes the 2 inputs r and R.
    Returns the hydrogen atomic s-orbitals for each ion.
    """
    #cartesian translation and scaling
    x1 = x - Rx
    y1 = y- Ry
    z1 = z - Rz  
    print(x1,y1,z1)
    rVec1 = torch.cat((x1,y1,z1),1)
    print(rVec1)
    r1 = toR(rVec1)
    
    #s atomic orbital activation
    phi_r1 = atomicAct_s(r1)        
    
    return phi_r1

a = torch.tensor([0,1,2])
b = torch.tensor([3,4,5])
c = torch.tensor([6,7,8])

print(torch.cat((a,b,c),0))

#Rx, Ry and Rz are positions of atom
Rx = 0
Ry = 0
Rz = 0

x = torch.arange(-100,100)
y = torch.arange(-100,100)
z = torch.arange(-100,100)

print(torch.cat((x,y,z),1))

phi = atomicUnit(x,y,z,Rx,Ry,Rz)


