# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:05:58 2023

@author: danie
"""

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
lineBoxW=2

font = {'size'   : 18}
matplotlib.rc('font', **font)
plt.rcParams['font.size']=28
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
    
def set_params():
    params = dict()
    
    boundaries = 18
    params['xL']= -boundaries; params['xR']= boundaries
    params['yL']= -boundaries; params['yR']= boundaries
    params['zL']= -boundaries; params['zR']= boundaries
    params['BCcutoff'] = 17.5

    params['RxL'] = 0.2; params['RxR']= 4
    params['Ry']= 0 ; params['Rz']= 0  
    
    params['cutOff']= 0.005
    params['lossPath'] = "data/loss_atom.pkl" ; 
    params['EnergyPath'] = "data/energy_atom.pkl" ; 
    params['saveModelPath'] = "models/atomsym.pt"
    params['loadModelPath']="models/atomsym.pt"
    params['EnrR_path'] = "data/energy_R_atom.pkl"
       
    params['sc_step'] = 3000; params['sc_decay']=.7 ## WAS 3000
    params['sc_sampling']= 1

    params['n_train'] = 100000; params['n_test'] = 80
    params['epochs'] = int(5e3); #2e3        
    params['lr'] = 8e-3; 
    
    #number of protons in nucleus
    params['Z'] = 1
    #number of electrons
    params['N_electrons'] = 1

    params['inversion_symmetry'] = 1  
    
    return params

def  exactE():
    """
    Exact energy Taken by H. Wind, J. Chem. Phys. 42, 2371 (1965); https://doi.org/10.1063/1.1696302
    """

    R_exact = np.round(np.arange(0.2, 4.1, .1),2)  
    e_exact  = np.zeros([len(R_exact),1])
    e_exact = [-1.8008, -1.6715, -1.5545, 
               -1.4518, -1.3623, -1.2843, -1.2159, -1.1558,
               -1.1026, -1.0554, -1.0132,  -0.9754, -0.9415,
               -0.9109, -0.8832, -0.8582, -0.8355, -0.8149,
               -0.7961, -0.7790, -0.7634, -0.7492, -0.7363,
               -0.7244, -0.7136, -0.7037, -0.6946, -0.6863,
               -0.6786, -0.6716, -0.6651, -0.6591, -0.6536,
               -0.6485, -0.6437, -0.6392, -0.6351,-0.6312,
               -0.6276] #, -0.62, -0.62, -0.62, -0.615,
               # -0.61, -0.61, -0.61, -0.605,-0.60,
               # -0.60]
    return R_exact, e_exact

##################### ACTIVATION FUNCTIONS    
class toR(torch.nn.Module):
    @staticmethod
    def forward(input):
        r2 = input[:,0].pow(2) + input[:,1].pow(2) + input[:,2].pow(2)
        r = torch.sqrt(r2); r = r.reshape(-1,1)
        return r

class atomicAct_s(torch.nn.Module):
    @staticmethod
    def forward(input):
        return  torch.exp(-input) 

## Differential Operators using autograd: 
    
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

## Misc physical functions

def V(x,y,z, R, params):
    """
    Potential energy function.
    For each electron calculate coulomb potential from all other electrons
    """
    #####!!!!!!!!!!!!!!!!How to input r_i and r_ij?
    r_i = 1
    r_ij = 2
    
    potential = 0
    for i in range(params['N_electrons']):
        potential -= params['Z']/r_i
        for j in range(i+1):
            if j != i:
                potential += 1/r_ij
    return potential
    
def hamiltonian(x,y,z,R,psi,params):
    """
    Returns Hamiltonian for this setup
    """
    laplacian = lapl(x,y,z,psi)        
    return  -0.5*laplacian + V(x,y,z,R,params)*psi
        
## Misc helper functions 

def sampling(params, n_points, linearSampling=False):
    # Sampling from a 4d space: 3d variable (x,y,z) and 1d parameter (R) space
    xR = params['xR']; xL = params['xL']; yR = params['yR'];
    yL = params['yL']; zR = params['zR']; zL = params['zL']; 
    cutOff=params['cutOff']
    
    if linearSampling == True:
        x = torch.linspace(xL,xR,n_points,requires_grad=False)
        y = torch.linspace(yL,yR,n_points,requires_grad=False)
        z = torch.linspace(zL,zR,n_points,requires_grad=False)
        R = torch.linspace(params['RxL'],params['RxR'],n_points,requires_grad=False)
    else: 
        x = (xL - xR) * torch.rand(n_points,1) + xR
        y = (yL - yR) * torch.rand(n_points,1) + yR
        z = (zL - zR) * torch.rand(n_points,1) + zR
        R = (params['RxL'] - params['RxR'])* torch.rand(n_points,1) + params['RxR']
        
    #r1,r2 = radial(x,y,z,R,params)
    #x[r1<cutOff] = cutOff
    #x[r2<cutOff] = cutOff
    x,y,z = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1); R=R.reshape(-1,1)        
    x.requires_grad=True; y.requires_grad=True; z.requires_grad=True; R.requires_grad=True     
    return x,y,z,R

def saveLoss(params, lossDictionary):
    with open(params['lossPath'], 'wb') as f:
        pickle.dump(lossDictionary, f)
        
def returnGate():
    modelTest = NN_atom(params);     modelTest.loadModel(params)   
    R = torch.linspace(params['RxL'],params['RxR'],params['n_train'], requires_grad=False)
    R = R.reshape(-1,1)
    R.requires_grad=True
    f = modelTest.netDecayL(R)
    f = modelTest.sig(f)
    f = modelTest.netDecay(f) 
    return R.cpu().detach().numpy(), f.cpu().detach().numpy()

def integra3d(x,y,z, f):   
    # 3d integration using Simpson method of scipy
    f = f.detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    z = z.detach().numpy()
    I = simps( [simps( [simps(fx, x) for fx in fy], y) for fy in f ]  ,z)
    return I

def plotLoss(params,  saveFig=True):
    with open(params['lossPath'], 'rb') as f:
        loaded_dict = pickle.load(f)

    plt.figure(figsize=[19,8])
    plt.subplot(1,2,1)
    plt.plot(loaded_dict['Ltot'],label='total',linewidth=lineW*2)
    plt.plot(loaded_dict['Lpde'],label='pde',linewidth=lineW)    
    plt.plot(loaded_dict['Lbc'] ,label='bc',linewidth=lineW)
    plt.ylabel('Loss'); plt.xlabel('epochs')
    plt.axvline(params['epochs'], c='r', linestyle='--', linewidth=lineW*1.5, alpha=0.7)
    plt.yscale('log')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loaded_dict['Energy'],'-k',linewidth=lineW)
    plt.ylabel('Energy'); plt.xlabel('epochs')
    plt.axvline(params['epochs'], c='r', linestyle='--', linewidth=lineW*1.5, alpha=0.7)
    plt.tight_layout()
    
    if saveFig==True:
        plt.savefig('figures/atom_figure.jpg', format='jpg')
        
##----------------------- Network Class ---------------------
# Neural Network Architecture 
class NN_atom(nn.Module):
    def __init__(self,params,dense_neurons=16,dense_neurons_E=32,netDecay_neurons=10): 
        super(NN_atom,self).__init__()

        self.sig =  nn.Sigmoid()          
        self.toR= toR()
        self.actAO_s = atomicAct_s()  
        self.Lin_H1 = torch.nn.Linear(2, dense_neurons) 
        self.Lin_H2 = torch.nn.Linear(dense_neurons, dense_neurons, bias=True) 
        
        self.Lin_out = torch.nn.Linear(dense_neurons, 1)                
        
        self.Lin_E1 = torch.nn.Linear(1, dense_neurons_E) 
        self.Lin_E2 = torch.nn.Linear(dense_neurons_E, dense_neurons_E) 

        self.Lin_Eout = torch.nn.Linear(dense_neurons_E, 1)                
        nn.init.constant_(self.Lin_Eout.bias[0], -1 ) 

        self.Ry = params['Ry'];  self.Rz = params['Rz']
        self.P = params['inversion_symmetry']
        self.netDecayL = torch.nn.Linear(1, netDecay_neurons, bias=True)  
        self.netDecay = torch.nn.Linear(netDecay_neurons, 1, bias=True)  

    def forward(self,x,y,z,R):        
        ## ENERGY PARAMETER
        e = self.Lin_E1(R); e = self.sig(e)
        e = self.Lin_E2(e); e = self.sig(e)
        E = self.Lin_Eout(e)
        
        ## ATOMIC Layer: Radial part and physics-based activation
        fi_r1,  fi_r2  = self.atomicUnit(x,y,z, R)        
        fi_r1m, fi_r2m = self.atomicUnit(-x,y,z,R)        
        ## LCAO SOLUTION
        N_LCAO = self.lcao_solution(fi_r1, fi_r2)
        ## NONLINEAR HIDDEN LAYERS        
        B  = self.base(fi_r1,fi_r2) + self.P*self.base(fi_r1m,fi_r2m)
        NN = self.Lin_out(B)
    
        f = self.netDecayL(R)
        f = self.sig(f)
        f = self.netDecay(f) 
        NN = NN*f

        Nout = NN + N_LCAO        
        return Nout, E

    def atomicUnit(self,x,y,z,R):
        x1 = x - R; 
        y1 = y- self.Ry; z1 = z - self.Rz     # Cartesian Translation & Scaling:    
        rVec1=torch.cat((x1,y1,z1),1)
        r1 = self.toR(rVec1) 
        fi_r1 = self.actAO_s(r1);  # s- ATOMIC ORBITAL ACTIVATION
        # -- 
        x2 = x + R; 
        y2 = y + self.Ry; z2 = z + self.Rz        
        rVec2=torch.cat((x2,y2,z2),1)
        r2 = self.toR(rVec2);         fi_r2 = self.actAO_s(r2);  
        return fi_r1, fi_r2

    def lcao_solution(self,fi_r1, fi_r2,  ):
        ## LCAO solution: Linear combination
        N_LCAO= (fi_r1 + self.P* fi_r2)        
        return N_LCAO
    
    def base(self,fi_r1,fi_r2):
        ## NONLINEAR HIDDEN LAYERS; Black box
        fi_r = torch.cat((fi_r1, fi_r2),1)    
        fi_r = self.Lin_H1(fi_r);         fi_r = self.sig(fi_r) 
        fi_r = self.Lin_H2(fi_r);         fi_r = self.sig(fi_r) 
        # fi_r = self.Lin_H3(fi_r);         fi_r = self.sig(fi_r) 
        return fi_r
        
    def freezeBase(self):
        #for p in self.parameters():
        #p.requires_grad=False
        self.Lin_H1.weight.requires_grad=False
        self.Lin_H1.bias.requires_grad=False
        self.Lin_H2.weight.requires_grad=False
        self.Lin_H2.bias.requires_grad=False    
        self.Lin_out.weight.requires_grad=False
        self.Lin_out.bias.requires_grad=False
        
    def freezeDecayUnit(self):
        self.netDecayL.weight.requires_grad=False
        self.netDecayL.bias.requires_grad=False
        self.netDecay.weight.requires_grad=False
        self.netDecay.bias.requires_grad=False
        
    def parametricPsi(self, x,y,z,R):
        N, E = self.forward(x,y,z,R) 
        return  N, E
   
    def loadModel(self,params):
        checkpoint = torch.load(params['loadModelPath'])
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval(); 

    def saveModel(self,params, optimizer):
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        },  params['saveModelPath'])   
        
    def LossFunctions(self, x,y,z,R,params, bIndex1, bIndex2):
        lam_bc, lam_pde = 1 , 1    #lam_tr = 1e-9
        psi, E = self.parametricPsi(x,y,z,R)
        #--# PDE       
        res = hamiltonian(x,y,z,R,psi,params) - E*psi                
        LossPDE = (res.pow(2)).mean() * lam_pde
        Ltot= LossPDE         
        #--# BC
        Lbc =  lam_bc *( (psi[bIndex1].pow(2)).mean() 
               + (psi[bIndex2].pow(2)).mean() )

        Ltot= LossPDE + Lbc
        # 
        #--# Trivial
        # Ltriv = 1/(psi.pow(2)).mean()* lam_tr ;    Ltot = Ltot + Ltriv 
        return Ltot, LossPDE, Lbc, E
    
def train(params, loadWeights=False, freezeUnits=False):
    lr = params['lr']; 
    model = NN_atom(params);     # modelBest=copy.deepcopy(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    print('train with Adam')   
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    # print('train with SGD')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['sc_step'], gamma=params['sc_decay'])    
    Llim =  10 ; optEpoch=0    

    epochs=params['epochs'] # For Adam 
    total_epochs = epochs
    # total_epochs = params['epochs_LB'] + epochs
    Ltot_h = np.zeros([total_epochs,1]); Lpde_h = np.zeros([total_epochs,1])
    Lbc_h= np.zeros([total_epochs,1]);   E_h = np.zeros([total_epochs,1]); 
    # Ltr_h = np.zeros([total_epochs,1]); Linv_h= np.zeros([total_epochs,1])
    
    ## LOADING pre-trained model if PATH file exists and loadWeights=True
    if path.exists(params['loadModelPath']) and loadWeights==True:
        print('loading model')
        model.loadModel(params) 

    if  freezeUnits==True:
        print('Freezeing Basis unit and Gate')
        model.freezeDecayUnit()
        model.freezeBase()
        
    TeP0 = time.time() # for counting the training time
        
    n_points = params['n_train'] # the training batch size
    x,y,z,R = sampling(params,n_points, linearSampling=False)
    
    #r1,r2 = radial(x, y, z,R, params)
    #bIndex1 = torch.where(r1 >= params['BCcutoff'])
    #bIndex2 = torch.where(r2 >= params['BCcutoff'])
    
    #####--------------------------------------------------------------------    
    ############### TRAINING LOOP ########################    
    #####--------------------------------------------------------------------
    for tt in range(epochs):        
        optimizer.zero_grad()
        
        if tt % params['sc_sampling']==0 and tt < 0.9*epochs:
            x,y,z,R = sampling(params, n_points, linearSampling=False)            
            #r1,r2 = radial(x, y, z,R, params)
            #bIndex1 = torch.where(r1 >= params['BCcutoff']   )
            #bIndex2 = torch.where(r2 >= params['BCcutoff']   )        
        
        Ltot, LossPDE, Lbc, E = model.LossFunctions(x,y,z,R,params)#, bIndex1, bIndex2)
        
        Ltot.backward(retain_graph=False); optimizer.step(); 
        # if  tt < 2001:
        #     scheduler.step()
        
        if (tt+1)%100 == 0:
            print(f'epoch {tt+1}/{epochs}')

        # keep history           
        Ltot_h[tt] = Ltot.cpu().data.numpy();  Lpde_h[tt] = LossPDE.cpu().data.numpy()
        Lbc_h[tt]  =  Lbc.cpu().data.numpy();  E_h[tt]    = E[-1].cpu().data.numpy()
        # Ltr_h[tt]  = Ltriv.data.numpy();  
        
        #    Keep the best model (lowest loss). Checking after 50% of the total epochs 
        if  tt > 0.5*epochs  and Ltot < Llim:
            Llim=Ltot
            model.saveModel(params, optimizer)
            optEpoch = tt
            
    print('Optimal epoch: ', optEpoch)

    #####--------------------------------------------------------------------
    ##### END TRAINING
    #####--------------------------------------------------------------------
    TePf = time.time();    runTime = TePf - TeP0        

    lossDictionary = {'Ltot':  Ltot_h,  'Lbc': Lbc_h, 'Lpde':Lpde_h, 'Energy': E_h }
    saveLoss(params, lossDictionary)

    print('Runtime (min): ', runTime/60)    
    print('last learning rate: ', scheduler.get_last_lr())
    # return E,R
    
params = set_params()

params['epochs'] = int(5e3);  nEpoch1 = params['epochs']
params['n_train'] = 100000 
params['lr'] = 8e-3;

#### ----- Training: Single model ---=---------
train(params,loadWeights=False);  

#plotLoss(params,saveFig=False)

Rg, gate = returnGate()
plt.plot(Rg, gate, linewidth=lineW)

#### ----- Fine Tuning ----------

# params=set_params()
params['loadModelPath'] = "models/atomsym.pt"
params['lossPath'] = "data/loss_atom_fineTune.pkl" ; 
params['EnergyPath'] = "data/energy_atom_fineTune.pkl" ; 
params['saveModelPath'] ="models/atom_fineTune.pt"

# params['sc_step'] = 10000; params['sc_decay']=.7
params['sc_sampling'] = 1

params['epochs'] = int(2e3); nEpoch2 = params['epochs']
params['n_train'] = 100000 
params['lr'] = 5e-4;

train(params, loadWeights=True, freezeUnits=True); 

#plotLoss(params, saveFig=False)