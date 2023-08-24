# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:15:14 2023

@author: danie
"""

import torch
import matplotlib.pyplot as plt

#constants
a0 = 5.29177210903e-11

#create some graphs for dissertation

num_points = 10000
boundary = 0.3e-9

x = torch.linspace(-boundary,boundary,num_points)

x = x.reshape(-1,1)
R = 0.5e-10

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

psi_plus = torch.add(psi1,psi2)
psi_minus = torch.sub(psi1,psi2)
psi_abs_plus = torch.pow(torch.abs(psi_plus),2)
psi_abs_minus = torch.pow(torch.abs(psi_minus),2)

figure, axis = plt.subplots(2,2,layout='constrained')

axis[0,0].plot(x.cpu(),psi_plus.cpu())
axis[0,0].axvline(x=R,color='r',linestyle='--')
axis[0,0].axvline(x=-R,color='r',linestyle='--')
axis[0,0].annotate('A',(-R,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[0,0].annotate('B',(R,0),textcoords="offset points",xytext=(0,-15),ha='center')
#axis[0,0].xlabel('$x$')
axis[0,0].set_title('$\Psi_{+}$')

axis[0,1].plot(x.cpu(),psi_minus.cpu())
axis[0,1].axvline(x=R,color='r',linestyle='--')
axis[0,1].axvline(x=-R,color='r',linestyle='--')
axis[0,1].axhline(0,color='grey',linestyle='--')
axis[0,1].annotate('A',(-R,0),textcoords="offset points",xytext=(0,-57),ha='center')
axis[0,1].annotate('B',(R,0),textcoords="offset points",xytext=(0,-57),ha='center')
#axis[0,1].xlabel('$x$')
axis[0,1].set_title('$\Psi_{-}$')

axis[1,0].plot(x.cpu(),psi_abs_plus.cpu())
axis[1,0].axvline(x=R,color='r',linestyle='--')
axis[1,0].axvline(x=-R,color='r',linestyle='--')
axis[1,0].annotate('A',(-R,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[1,0].annotate('B',(R,0),textcoords="offset points",xytext=(0,-15),ha='center')
#axis[1,0].xlabel('$x$')
axis[1,0].set_title('$|\Psi_{+}|^{2}$')

axis[1,1].plot(x.cpu(),psi_abs_minus.cpu())
axis[1,1].axvline(x=R,color='r',linestyle='--')
axis[1,1].axvline(x=-R,color='r',linestyle='--')
axis[1,1].annotate('A',(-R,0),textcoords="offset points",xytext=(0,-15),ha='center')
axis[1,1].annotate('B',(R,0),textcoords="offset points",xytext=(0,-15),ha='center')
#axis[1,1].xlabel('$x$')
axis[1,1].set_title('$|\Psi_{-}|^{2}$')

