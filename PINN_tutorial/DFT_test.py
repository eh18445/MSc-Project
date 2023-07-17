# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:23:56 2023

@author: danie
"""

#Given an atom perform DFT calculations
#using torch-dftd

from ase.build import molecule
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

atoms = molecule("CH3CH2OCH3")
# device="cuda:0" for fast GPU computation.
calc = TorchDFTD3Calculator(atoms=atoms,device="cpu",damping="bj")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"energy {energy} eV")
print(f"forces {forces}")
