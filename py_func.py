# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:59:01 2018

@author: hif10
"""

import re
import numpy as np

def getData(dataFile):
    line = dataFile.readline()
    if "Iteration" in line:
        sptline = line.split()
        nAtoms = int(sptline[0])
        iIter = int(re.match("(\d*),",sptline[2])[1])
        
        dataFile.readline()
        lattice = np.zeros((3,3))
        lattice[0,:] = np.array(dataFile.readline().split(),dtype=float)
        lattice[1,:] = np.array(dataFile.readline().split(),dtype=float)
        lattice[2,:] = np.array(dataFile.readline().split(),dtype=float)
        
        dataFile.readline()
        R = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            R[i,:] = np.array(sptline[1:4])
        
        dataFile.readline()
        forces = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            forces[i,:] = np.array(sptline[1:4])
            
        dataFile.readline()
        velocities = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            velocities[i,:] = np.array(sptline[1:4])
            
        dataFile.readline()
        energies = np.zeros((nAtoms))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            energies[i] = float(sptline[1])

        return nAtoms, iIter, lattice, R, forces, velocities,energies 
    
    else:
        return getData(dataFile)
