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
from scipy.integrate import simps
from torch.autograd import grad

font = {'size': 11}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = 12
dtype = torch.double

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

def radial(x,y,z,R):
    """
    Returns the radial part from cartesian coordinates
    """
    Rx = R
    Ry = 0
    Rz = 0
    r1 = torch.sqrt((x-Rx).pow(2)+(y-Ry).pow(2)+(z-Rz).pow(2))
    r2 = torch.sqrt((x+Rx).pow(2)+(y+Ry).pow(2)+(z+Rz).pow(2))
    return r1, r2

def V(x,y,z,R,Z1,Z2):
    """
    Potential energy function.
    For each electron calculate coulomb potential from all other electrons
    """
    #positions of each atom
    r1,r2 = radial(x,y,z,R)
    
    #effective nuclear charge
    eff_charge = {1:1, 2:1.688, 3:1.279, 4:1.912, 5:2.421, 6:3.136, 7:3.834, 8:4.453, 9:5.1, 10:5.758,
                  11:2.507, 12:3.308, 13:4.066, 14:4.285, 15:4.886, 16:5.482, 17:6.116, 18:6.764, 19:3.495, 20:4.398,
                  21:7.12, 22:8.141, 23:8.983, 24:9.757, 25:10.582, 26:11.18, 27:11.855, 28:12.53, 29:13.201, 30:13.878}
    
    Z1_eff = eff_charge[Z1]
    Z2_eff = eff_charge[Z2]
    
    potential = -Z1_eff/r1 -Z2_eff/r2
   
    return potential

def effectivePotential(Z,orb):
    """
    Takes an input atomic number and outermost orbital 
    returns the effective potential
    """
    return
    
def dfx(x,f):
    """
    Returns 1st derivative
    """
    return grad([f],[x],grad_outputs=torch.ones(x.shape,dtype=dtype),create_graph=True)[0]

def d2fx(x,f):
    """
    Returns 2nd derivative
    """
    return grad(dfx(x,f),[x],grad_outputs=torch.ones(x.shape,dtype=dtype),create_graph=True)[0]

def lapl(x,y,z,f):
    """
    Returns value of the laplacian operator at x,y,z for function f
    """

    f_xx, f_yy, f_zz = d2fx(x,f), d2fx(y,f), d2fx(z,f)
    return f_xx + f_yy + f_zz

def hamiltonian(x,y,z,R,psi,Z1,Z2):
    """
    Returns Hamiltonian for this setup
    """
    laplacian = lapl(x,y,z,psi)    
    return  -0.5*laplacian + V(x,y,z,R,Z1,Z2)*psi
        
    
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
        if orbital_name == '3px':
            chi = chi.mul(torch.exp(phi*1j))
        else:
            chi = chi.mul(torch.exp(-1*phi*1j))
    #4s goes here
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
    
    return chi.real

def atomicUnit(x,y,z,R,Ry,Rz,Z1,Z2):
    """
    Takes the 2 inputs r and R.
    Z1 and Z2 is the atomic numbers of the atoms.
    R is the positions of the atoms along x axis.
    Returns the hydrogen atomic orbitals for the atoms in an array.
    """

    #convert cartesian co-ordinates to polar
    #calculate the orbital
    #convert back to cartesian
    
    #cartesian translation and scaling
    x1 = x - R
    y1 = y - Ry
    z1 = z - Rz
                               
    rVec1 = torch.cat((x1,y1,z1),1)
    
    #covert to polar co-ords
    r1 = toR(rVec1)
    theta1 = torch.arccos(z1/r1)
    phi1 = torch.sgn(y1)*torch.arccos(x1/(torch.pow(x1,2)+torch.pow(y1,2)))
    
    #for each relevant orbital calculate phi vector
    
    chi = orbital(r1,theta1,phi1,Z1,orbital_name='1s')
    
    #Other atom
    x2 = x + R
    y2 = y - Ry
    z2 = z - Rz
                               
    rVec2 = torch.cat((x2,y2,z2),1)
    
    #covert to polar co-ords
    r2 = toR(rVec2)
    theta2 = torch.arccos(z2/r2)
    phi2 = torch.sgn(y2)*torch.arccos(x2/(torch.pow(x2,2)+torch.pow(y2,2)))
    
    chi = torch.cat((chi,orbital(r2,theta2,phi2,Z2,orbital_name='1s')),1)
    
    chi = torch.cat((chi,orbital(r2,theta2,phi2,Z2,orbital_name='2s')),1)
    
    chi = torch.cat((chi,orbital(r2,theta2,phi2,Z2,orbital_name='2pz')),1)
    
    chi = torch.cat((chi,orbital(r2,theta2,phi2,Z2,orbital_name='2px')),1)
    
    chi = torch.cat((chi,orbital(r2,theta2,phi2,Z2,orbital_name='2py')),1)
    
    return chi

def LCAO_Solution(x,y,z,R,chi,Z1,Z2):
    """
    Calculate the LCAO solution for the atomic orbitals given in chi.
    Determine H_matrix which is NxN matrix where each entry H_ij is <chi_i|H|chi_j>
    Determine S_matrix where each entry is <chi_i|chi_j>
    Solve S^-1*H*c = E*c eigen equation
    Return values of c and E
    """
    time_start = time.time()
    
    print(chi.shape)
    
    #calculate H and S
    n_orbitals = chi.shape[1]
    H = torch.zeros((n_orbitals,n_orbitals))
    S = torch.zeros((n_orbitals,n_orbitals))
    
    for i in range(0,n_orbitals):
        for j in range(i+1):
            #print(i,j)
            print(chi[:,j])
            #######TYhis is supposed tpo be integrasated!!!!
            H_chi = hamiltonian(x,y,z,R,chi[:,j].reshape(-1,1),Z1,Z2)
            print('<chi|:',chi[:,j])
            print('H|chi>:',H_chi)
            H[i,j] = torch.matmul(chi[:,j],H_chi)
            H[j,i] = H[i,j]
            print('<chi|H|chi>',H[i,j])
            S[i,j] = torch.matmul(chi[:,i],chi[:,j].reshape(-1,1))
            S[i,j] = S[j,i]
    
    #Solve eigen equation
    print(H)
    print(S)
    S_inv = torch.linalg.inv(S)
    S_inv_H = torch.matmul(S_inv,H)
    
    #use torch.linalg.eig(MATRIX=S^-1H)
    E, c = torch.linalg.eig(S_inv_H)
    
    print('Time to compute LCAO:',(time.time()-time_start)/60)
    
    return E.real, c.real

#R, Ry and Rz are positions of atom
Ry1 = 0
Rz1 = 0

Ry2 = 0
Rz2 = 0

x = torch.linspace(-10,10,10000,requires_grad=True)
y = torch.linspace(-10,10,10000,requires_grad=True)
z = torch.linspace(-10,10,10000,requires_grad=True)
R = torch.linspace(0.2,4,10000,requires_grad=True)

#H
Z1 = 1

#O
Z2 = 8

x,y,z,R = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1),R.reshape(-1,1)

chi = atomicUnit(x,y,z,R,Ry1,Rz1,Z1,Z2)

E, c = LCAO_Solution(x,y,z,R,chi,Z1,Z2)

print('c',c,'E',E)
print(c[:,0],chi[:,0])

#6 different solutions



n_orbitals = chi.shape[1]
for i in range(n_orbitals):
    lcao = c[i,0]*chi[:,0]
    for j in range(1,n_orbitals):
        print(i,j,c[j,i])
        lcao = torch.add(lcao,c[j,i]*chi[:,j])
        
    print(lcao)

