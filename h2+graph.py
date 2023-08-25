# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:15:14 2023

@author: danie
"""

import torch
import matplotlib.pyplot as plt
import scipy

#constants
a0 = 5.29177210903e-11

#create some graphs for dissertation

num_points = 10000
boundary = 0.3e-9

x = torch.linspace(-boundary,boundary,num_points)
x_pm = x*10**12

x = x.reshape(-1,1)
R = 67e-12
R_pm = R*10**12

def sOrbital(r):
    rho = r/a0
    psi = torch.exp(-rho)/(torch.sqrt(torch.tensor(torch.pi*a0**3)))
    psi = psi.reshape(-1,1)
    return psi

# Cartesian Translation & Scaling: 
x1 = torch.abs(x + R)    
x2 = torch.abs(x - R)
    
psi1 = sOrbital(x1)
psi2 = sOrbital(x2)

#!!!!!!!!!!!!!Need to be normalised
#xlim is -300pm,300pm

psi_plus = torch.add(psi1,psi2)
psi_minus = torch.sub(psi1,psi2)
 
print(max(psi_plus))

psi_abs_plus = torch.pow(torch.abs(psi_plus),2)
psi_abs_minus = torch.pow(torch.abs(psi_minus),2)

N_plus = scipy.integrate.simpson(psi_abs_plus.reshape((-1)),x.reshape((-1)))
N_minus = scipy.integrate.simpson(psi_abs_minus.reshape((-1)),x.reshape((-1)))

psi_plus /= torch.tensor([N_plus])
psi_minus /= torch.tensor([N_minus])

psi_abs_plus = torch.pow(torch.abs(psi_plus),2)
psi_abs_minus = torch.pow(torch.abs(psi_minus),2)

figure, axis = plt.subplots(2,2,layout='compressed',dpi=200)

axis[0,0].plot(x_pm.cpu(),psi_plus.cpu())
axis[0,0].axvline(x=R_pm,color='r',linestyle='--')
axis[0,0].axvline(x=-R_pm,color='r',linestyle='--')
axis[0,0].annotate('A',(-R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[0,0].annotate('B',(R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
#axis[0,0].xlabel('$x$')
axis[0,0].set_xticks([-300,-150,0,150,300])
axis[0,0].annotate('$x$ (pm)',(0,0),textcoords="offset points",xytext=(0,-30),ha='center')
axis[0,0].set_title('$\Psi_{+}$')

axis[0,1].plot(x_pm.cpu(),psi_minus.cpu())
axis[0,1].axvline(x=R_pm,color='r',linestyle='--')
axis[0,1].axvline(x=-R_pm,color='r',linestyle='--')
axis[0,1].axhline(0,color='grey',linestyle='--')
axis[0,1].annotate('A',(-R_pm,0),textcoords="offset points",xytext=(0,-58),ha='center')
axis[0,1].annotate('B',(R_pm,0),textcoords="offset points",xytext=(0,-58),ha='center')
#axis[0,1].xlabel('$x$')
axis[0,1].set_xticks([-300,-150,0,150,300])
axis[0,1].annotate('$x$ (pm)',(0,0),textcoords="offset points",xytext=(0,-73),ha='center')
axis[0,1].set_title('$\Psi_{-}$')

axis[1,0].plot(x_pm.cpu(),psi_abs_plus.cpu())
axis[1,0].axvline(x=R_pm,color='r',linestyle='--')
axis[1,0].axvline(x=-R_pm,color='r',linestyle='--')
axis[1,0].annotate('A',(-R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[1,0].annotate('B',(R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[1,0].set_xticks([-300,-150,0,150,300])
#axis[1,0].xlabel('$x$')
axis[1,0].annotate('$x$ (pm)',(0,0),textcoords="offset points",xytext=(0,-30),ha='center')
axis[1,0].set_title('$|\Psi_{+}|^{2}$')

axis[1,1].plot(x_pm.cpu(),psi_abs_minus.cpu())
axis[1,1].axvline(x=R_pm,color='r',linestyle='--')
axis[1,1].axvline(x=-R_pm,color='r',linestyle='--')
axis[1,1].annotate('A',(-R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[1,1].annotate('B',(R_pm,0),textcoords="offset points",xytext=(0,-15),ha='center')
#axis[1,1].xlabel('$x$')
axis[1,1].set_xticks([-300,-150,0,150,300])
axis[1,1].annotate('$x$ (pm)',(0,0),textcoords="offset points",xytext=(0,-30),ha='center')
axis[1,1].set_title('$|\Psi_{-}|^{2}$')

#plt.savefig('H2+_wavefunctions.png')

