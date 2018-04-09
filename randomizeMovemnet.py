#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:19:30 2018

@author: yufeng
"""

import numpy as np
import re
import argparse


def getData(dataFile):
    line = dataFile.readline()
    if "Iteration" in line:
        sptline = line.split()
        print(sptline)
        nAtoms = int(sptline[0])
        iIter = int(re.match("(\d*),", sptline[2])[1])
        iEp = float(sptline[5])
        return nAtoms, iIter, iEp
    else:
        return getData(dataFile)


def copyFile(nAtoms, inFile, outFile):
    nLine = 10 + 4*nAtoms
    for j in range(nLine):
        outFile.write(inFile.readline())

nTestSet = 100       # test set
nValidSet = 100      # validation set
nTrainSet = -1       # training set

numIter = 1000
Etol = 0.1 #eV

# fileName = "MOVEMENT"

parser = argparse.ArgumentParser()
parser.add_argument("fileName", type=str, help="PWMat MD MOVEMENT file")
parser.add_argument("--numIter", type=int, help="Number of iterations in the MOVEMENT file to be considered")
parser.add_argument("--nTestSet", type=int, help="Number of iterations in the test set")
parser.add_argument("--nValidSet", type=int, help="Number of iterations in the validation set")
parser.add_argument("--eTol", type=float, help="Tolerance in Ep to drop cases")

args = parser.parse_args()
fileName = str(getattr(args, "fileName"))
if getattr(args, "numIter") is not None:
    numIter = int(getattr(args, "numIter"))
if getattr(args,"nTestSet") is not None:
    nTestSet = int(getattr(args, "nTestSet"))
if getattr(args, "nValidSet") is not None:
    nValidSet = int(getattr(args, "nValidSet"))
if getattr(args, "eTol") is not None:
    Etol = float(getattr(args, "eTol"))

nDataSet = np.array([nTestSet,nValidSet,nTrainSet])

if np.sum(nDataSet < 0) > 1:
    print("Error, more than one set has a negative value.")

nAtomList = np.arange(numIter)
numAtomList = np.zeros(numIter).astype(int)
EpList = np.zeros(numIter)
EiList = np.zeros(numIter)

file = open(fileName, 'r')
for i in range(numIter):
    line = file.readline()
    numAtomList[i] = int(line.split()[0])
    EpList[i] = float(line.split()[5])
    file.readline()
    [file.readline() for k in range(3)]
    file.readline()
    [file.readline() for k in range(numAtomList[i])]
    file.readline()
    [file.readline() for k in range(numAtomList[i])]
    file.readline()
    [file.readline() for k in range(numAtomList[i])]
    file.readline()
    Ei = 0
    for j in range(numAtomList[i]):
        Ei += float(file.readline().split()[1])
    EiList[i] = Ei
    file.readline()
file.close()

eErr = np.mean(EpList - EiList)
eMask = np.abs((EpList - EiList) - (eErr)) < 0.1
eIdx = np.where(eMask)[0]
np.random.shuffle(eIdx)

nDataSet[nDataSet==-1] = np.sum(eMask) - np.sum(nDataSet[nDataSet>0])
testSetList = np.sort(eIdx[:nDataSet[0]])
validSetList = np.sort(eIdx[nDataSet[0]:nDataSet[0]+nDataSet[1]])
trainSetList = np.sort(eIdx[nDataSet[0]+nDataSet[1]:])

inFile = open(fileName, 'r')
testFile = open(fileName+".test", "w+")
validFile = open(fileName+".valid", "w+")
trainFile = open(fileName+".train", "w+")

for i in range(numIter):
    if i in testSetList:
        copyFile(numAtomList[i],inFile,testFile)
    elif i in validSetList:
        copyFile(numAtomList[i],inFile,validFile)
    elif i in trainSetList:
        copyFile(numAtomList[i],inFile,trainFile)

'''
file = open(fileName, 'r')
for i in range(numIter):
    lattice = np.zeros((3,3))
    nAtoms, iIter, iEp = line = getData(file)
    file.readline()
    lattice[0,:] = np.array(file.readline().split(),dtype=float)
    lattice[1,:] = np.array(file.readline().split(),dtype=float)
    lattice[2,:] = np.array(file.readline().split(),dtype=float)
    file.readline()
    if i in testSetList:
        print("Test set")
        print(nAtoms, iIter, iEp)
    elif i in validSetList:
        print("Validation set")
        print(nAtoms, iIter, iEp)
    elif i in trainSetList:
        print("Training set")
        print(nAtoms, iIter, iEp)
    else:
        print("i = ",i,"does not belong to any of the datasets")
file.close()
'''
#file = open(fileName)
