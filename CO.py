# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:30:39 2023

@author: danie
"""

# C-O molecule with LCAO
#From: http://lampz.tugraz.at/~hadley/ss1/molecules/hueckel/CO/mo_CO.php
########################

import numpy as np
import time
from numpy import savetxt
import torch

# constants

a0 = 5.2917721*10**(-11)
hbar = 1.0545718*10**(-34)
m = 9.10938356*10**(-31)
e = 1.6021766208*10**(-19)
ep = 8.854187817*10**(-12)
blength = 112.8e-12 # bond length between the atoms
x01 = -blength/2 # x coordinate of one of the atoms - here C
x02 = blength/2 # x coordinate of the other atom - here = O
Z1 = 3.25 # using slater's rules the effective charge for C for the 2s and 2pxyz orbitals is 3.25
Z2 = 4.55 # using slater's rules the effective charge for O for the 2s and 2pxyz orbitals is 4.55

# calculation parameters
bound = 10
Npoints = 140
delta = bound*a0/Npoints


# moleculare orbitals in cartesian coordinates

def s2(x,y,z,x0,y0,z0,Z): # 2s atomic orbital
    return 0.25*np.sqrt(Z**3/(2*np.pi*a0**3))*(2-Z/a0*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))*np.exp(-Z/(2*a0)*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))

def px2(x,y,z,x0,y0,z0,Z): # 2px atomic orbiatl
    return 0.25*np.sqrt(Z**5/(2*np.pi*a0**5))*(x-x0)*np.exp(-Z/(2*a0)*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))

def py2(x,y,z,x0,y0,z0,Z): # 2py atomic orbital
    return 0.25*np.sqrt(Z**5/(2*np.pi*a0**5))*y*np.exp(-Z/(2*a0)*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))

def pz2(x,y,z,x0,y0,z0,Z): # 2pz atomic orbital
    return 0.25*np.sqrt(Z**5/(2*np.pi*a0**5))*z*np.exp(-Z/(2*a0)*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))

def Ls2(x,y,z,x0,y0,z0,Z): # lagrange operator used on the 2s orbital
    return 0.25*np.sqrt(Z**5/(2*np.pi*a0**5))*(5*Z/(2*a0)-4/np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)-Z**2*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(4*a0**2))*np.exp(-Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(2*a0))

def Lpx2(x,y,z,x0,y0,z0,Z): # lagrange operator used on the 2px orbital
    return np.sqrt(Z**7/(np.pi*2**5*a0**7))*(Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(4*a0)-2)*(x-x0)/np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)*np.exp(-Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(2*a0))

def Lpy2(x,y,z,x0,y0,z0,Z): # lagrange operator used on the 2py orbital
    return np.sqrt(Z**7/(np.pi*2**5*a0**7))*(Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(4*a0)-2)*y/np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)*np.exp(-Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(2*a0))

def Lpz2(x,y,z,x0,y0,z0,Z): # lagrange operator used on the 2pz orbital
    return np.sqrt(Z**7/(np.pi*2**5*a0**7))*(Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(4*a0)-2)*z/np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)*np.exp(-Z*np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)/(2*a0))

# molecular oribal hamiltonian
def Hamilton(f,Lf,x,y,z,wavepos,atompos1,atompos2,wavecharge,atomcharge1,atomcharge2):
    return (-hbar**2/(2*m)*Lf(x,y,z,wavepos,0,0,wavecharge)-(atomcharge1*e**2/(4*np.pi*ep*np.sqrt((x-atompos1)**2+y**2+z**2))+atomcharge2*e**2/(4*np.pi*ep*np.sqrt((x-atompos2)**2+y**2+z**2)))*f(x,y,z,wavepos,0,0,wavecharge))

# you give a three dimensional index input and get the indexes of the 8 corners of a cube that is around the input
#basically makes a cube around your position

def cube_indices(i,j,k):
    return [[i,j,k],[i+1,j,k],[i,j+1,k],[i,j,k+1],[i+1,j+1,k],[i+1,j,k+1],[i,j+1,k+1],[i+1,j+1,k+1]]

# sums over all the cubes, each cube is multiplied by (delta**3)/8 because the lengths of the sides of the cube are delta and the /8 is to average the results

def sum_cube(F, Npoints, delta):
    S_ = 0
    for i in range(Npoints-1):
        for j in range(Npoints-1):
            for k in range(Npoints-1):
                for indices in cube_indices(i,j,k):
                    S_ += F[indices[0]][indices[1]][indices[2]]

    S_ = S_ * (delta**3)/8.
    return S_

# function to compute an element of the Hamilton matrix, creates a Npoints x Npoints x Npoints matrix and then puts it into the sum_cube function
    #using -bound*a0/2 as a start and 10 as bound it starts -5*a0 on every dimension and gets to 5*a0
def H_integral(f1,f2,Lf2,Npoints,delta,start,wavepos1,wavepos2,atompos1,atompos2,wavecharge1,wavecharge2,atomcharge1,atomcharge2):
    F = np.zeros((Npoints,Npoints,Npoints))
    for i in range(Npoints):
       for j in range(Npoints):
           for k in range(Npoints):
               F[i][j][k] = f1(start+i*delta,start+j*delta,start+k*delta,wavepos1,0,0,wavecharge1)*Hamilton(f2,Lf2,start+i*delta,start+j*delta,start+k*delta,wavepos2,atompos1,atompos2,wavecharge2,atomcharge1,atomcharge2)

    return sum_cube(F, Npoints, delta)


# Function to compute an element of the overlap matrix S, no need for hamiltonian here
def S_integral(f1,f2,Lf2,Npoints,delta,start,wavepos1,wavepos2,wavecharge1,wavecharge2):
    F = np.zeros((Npoints,Npoints,Npoints))
    for i in range(Npoints):
        for j in range(Npoints):
            for k in range(Npoints):
                F[i][j][k] = f1(start+i*delta,start+j*delta,start+k*delta,wavepos1,0,0,wavecharge1)*f2(start+i*delta,start+j*delta,start+k*delta,wavepos2,0,0,wavecharge2)

    return sum_cube(F, Npoints, delta)

time_a = time.time() #takes time to know how long it takes

# Hamilton Matrix H
H = np.zeros((8,8))
orb = [s2,px2,py2,pz2,s2,px2,py2,pz2]
L = [Ls2,Lpx2,Lpy2,Lpz2,Ls2,Lpx2,Lpy2,Lpz2]

for i in range(8):
    for j in range(8):
        print(i,j)
        if j < 4 and i < 4:
            H[i][j] = H_integral(orb[i],orb[j],L[j],Npoints,delta,-bound/2*a0,x01,x01,x01,x02,Z1,Z1,Z1,Z2)
        elif j > 4 and i < 4:
            H[i][j]= H_integral(orb[i],orb[j],L[j],Npoints,delta,-bound/2*a0,x01,x02,x01,x02,Z1,Z2,Z1,Z2)
        elif j < 4 and i > 4:
            H[i][j]= H_integral(orb[i],orb[j],L[j],Npoints,delta,-bound/2*a0,x02,x01,x01,x02,Z2,Z1,Z1,Z2)
        elif j > 4 and i > 4:
            H[i][j]= H_integral(orb[i],orb[j],L[j],Npoints,delta,-bound/2*a0,x02,x02,x01,x02,Z2,Z2,Z1,Z2)
            
# Overlap Matrix S
S = np.zeros((8,8))

# S is symmetric, therefore only the upper triangular matrix was computed.
# This matrix is then added to the transposed version.


S[0][4]= S_integral(s2,s2,Ls2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[0][5]= S_integral(s2,px2,Lpx2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[0][6]= S_integral(s2,py2,Lpy2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[0][7]= S_integral(s2,pz2,Lpz2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)


S[1][4]= S_integral(px2,s2,Ls2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[1][5]= S_integral(px2,px2,Lpx2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[1][6]= S_integral(px2,py2,Lpy2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[1][7]= S_integral(px2,pz2,Lpz2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)

S[2][4]= S_integral(py2,s2,Ls2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[2][5]= S_integral(py2,px2,Lpx2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[2][6]= S_integral(py2,py2,Lpy2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[2][7]= S_integral(py2,pz2,Lpz2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)

S[3][4]= S_integral(pz2,s2,Ls2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[3][5]= S_integral(pz2,px2,Lpx2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[3][6]= S_integral(pz2,py2,Lpy2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)
S[3][7]= S_integral(pz2,pz2,Lpz2,Npoints,delta,-bound/2*a0,x01,x02,Z1,Z2)


S = np.transpose(S) + S
S = S + np.eye(8,8)


print(H) # while H[i][j] = H[j][i] is true for pretty much all cases but there are a few deviations in the last decimal places
savetxt('H_ematrixwithoutcorrectionCO_float.txt', H/e, fmt = '%.4f')

H2 = np.zeros((8,8))

for i in range(0,8):
    for j in range(0,8):
        H2[i][j] = (H[i][j] + H[j][i])/2

print()
print(H2)

print(S)



Sinv = np.linalg.inv(S) # inverting overlap matrix
[Eval,Evec] = np.linalg.eig(np.matmul(Sinv,H2))
Eval = Eval/e # dividing by e so it's in eV


idx = Eval.argsort()[::1] # sorting them so the lowest eigenvalue is first
Eval = Eval[idx]
Evec = Evec[:,idx] # putting the eigenvectors in the same order

print(Eval)
print('')
print(Evec)

#savetxt('EigenvaluesCO.txt', Eval, fmt = '%.4f',delimiter='\n')
#savetxt('EigenvectorsCO_power.txt', Evec, fmt = '%.4e')
#savetxt('EigenvectorsCO_float.txt', Evec, fmt = '%.4f')
#savetxt('HmatrixCO.txt', H2, fmt = '%.2e')
#savetxt('H_ematrixCO.txt', H2/e, fmt = '%.2f')
#savetxt('SmatrixCO.txt', S, fmt = '%.2f')



time_b = time.time() # taking second time
print()
print('It took','%.0f' %((time_b - time_a)//60),'minutes and','%.1f' %((time_b - time_a)%60),'seconds to complete the calculations') # how long it took to calculate

