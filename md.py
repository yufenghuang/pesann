# -*- coding: utf-8 -*-

import numpy as np
import py_func as pyf
import tensorflow as tf
import tf_func as tff

import pandas as pd

import os
import sys

import re


def getRVmmt(mmtFile):
    line = mmtFile.readline()

    sptline = line.split()
    nAtoms = int(sptline[0])

    mmtFile.readline()
    lattice = np.zeros((3, 3))
    lattice[0, :] = np.array(mmtFile.readline().split(), dtype=float)
    lattice[1, :] = np.array(mmtFile.readline().split(), dtype=float)
    lattice[2, :] = np.array(mmtFile.readline().split(), dtype=float)

    mmtFile.readline()
    R = np.zeros((nAtoms, 3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        R[i, :] = np.array(sptline[1:4])

    mmtFile.readline()
    velocities = np.zeros((nAtoms,3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        velocities[i,:] = np.array(sptline[1:4])

    R[R > 1] = R[R > 1] - np.floor(R[R > 1])
    R[R < 0] = R[R < 0] - np.floor(R[R < 0])

    return nAtoms, lattice, R, velocities

def getRFVmmt(mmtFile):
    line = mmtFile.readline()

    sptline = line.split()
    nAtoms = int(sptline[0])

    mmtFile.readline()
    lattice = np.zeros((3, 3))
    lattice[0, :] = np.array(mmtFile.readline().split(), dtype=float)
    lattice[1, :] = np.array(mmtFile.readline().split(), dtype=float)
    lattice[2, :] = np.array(mmtFile.readline().split(), dtype=float)

    mmtFile.readline()
    R = np.zeros((nAtoms, 3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        R[i, :] = np.array(sptline[1:4])

    mmtFile.readline()
    forces = np.zeros((nAtoms,3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        forces[i,:] = np.array(sptline[1:4])

    mmtFile.readline()
    velocities = np.zeros((nAtoms,3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        velocities[i,:] = np.array(sptline[1:4])

    R[R > 1] = R[R > 1] - np.floor(R[R > 1])
    R[R < 0] = R[R < 0] - np.floor(R[R < 0])

    return nAtoms, lattice, R, forces, velocities


def getData11(dataFile):
    line = dataFile.readline()
    if "Iteration" in line:
        sptline = line.split()
        nAtoms = int(sptline[0])
        iIter = int(re.match("(\d*),", sptline[2])[1])

        dataFile.readline()
        lattice = np.zeros((3, 3))
        lattice[0, :] = np.array(dataFile.readline().split(), dtype=float)
        lattice[1, :] = np.array(dataFile.readline().split(), dtype=float)
        lattice[2, :] = np.array(dataFile.readline().split(), dtype=float)

        dataFile.readline()
        R = np.zeros((nAtoms, 3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            R[i, :] = np.array(sptline[1:4])

        dataFile.readline()
        forces = np.zeros((nAtoms, 3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            forces[i, :] = np.array(sptline[1:4])

        dataFile.readline()
        velocities = np.zeros((nAtoms, 3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            velocities[i, :] = np.array(sptline[1:4])

        R[R > 1] = R[R > 1] - np.floor(R[R > 1])
        R[R < 0] = R[R < 0] - np.floor(R[R < 0])

        return nAtoms, iIter, lattice, R, forces, velocities

    else:
        return getData11(dataFile)


def specialrun1(params):
    J = 1/1.602177e-19 # eV
    meter = 1e10 # Angstroms
    s = 1e12 # ps
    mole = 6.022141e23 # atoms
    kg = 1e3*mole # grams/mole

    mSi = 28.09 # grams/mol

    constA = J/(kg*meter**2/s**2)

    dt = float(params["dt"])

#    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
#    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs, fll, fln = tff.tf_getEF2(tfCoord, tfLattice, params)
#    tfEp = (tfEs - tfEngyB) / tfEngyA
#    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, lattice, R, V0 = getRVmmt(mmtFile)

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    V0 = np.zeros_like(V0)
    Vpos = np.zeros_like(R0)
    Vneg = np.zeros_like(R0)
    
    atom=50

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp, Fll, Fln = sess.run((tfEs, tfFs, fll, fln), feed_dict=feedDict)
        Fp = -Fp
        Fll = -Fll
        Fln = -Fln
        
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2*constA)
        Etot = Epot + Ekin
        
        Vneg[atom, 0] = V0[atom, 0] - 0.5*Fp[atom, 0]/mSi*dt / constA
        Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
        
        R1 = R0 + Vpos * dt
#        R1 = R0 + dt
        
        print(nAtoms)
        print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
        for iAtom in range(nAtoms):
            print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
            
        print("Energies:")
        for iAtom in range(nAtoms):
            print("Energy"+str(iAtom), Ep[iAtom])

        print("Forces:")
        for iAtom in range(nAtoms):
            print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
            sys.stdout.flush()
            
        print("Fll:")
        for iAtom in range(nAtoms):
            print("Fll"+str(iAtom), Fll[iAtom, 0], Fll[iAtom, 1], Fll[iAtom, 2])
            sys.stdout.flush()
            
        print("Fln:")
        for iAtom in range(nAtoms):
            print("Fln"+str(iAtom), Fln[iAtom, 0], Fln[iAtom, 1], Fln[iAtom, 2])
            sys.stdout.flush()


                
        for iStep in range(1,params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)
                
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp, Fll, Fln = sess.run((tfEs, tfFs, fll, fln), feed_dict=feedDict)
            Fp = -Fp
            Fll = -Fll
            Fln = -Fln
        
            V0[atom, 0] = Vneg[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            
            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2*constA)
            Etot = Epot + Ekin

            Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            R1 = R0 + Vpos * dt
#            R1 = R0 + dt
            
            print(nAtoms)
            print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
            for iAtom in range(nAtoms):
                print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
                
            print("Energies:")
            for iAtom in range(nAtoms):
                print("Energy"+str(iAtom), Ep[iAtom])
    
            print("Forces:")
            for iAtom in range(nAtoms):
                print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()
                
            print("Fll:")
            for iAtom in range(nAtoms):
                print("Fll"+str(iAtom), Fll[iAtom, 0], Fll[iAtom, 1], Fll[iAtom, 2])
                sys.stdout.flush()
                
            print("Fln:")
            for iAtom in range(nAtoms):
                print("Fln"+str(iAtom), Fln[iAtom, 0], Fln[iAtom, 1], Fln[iAtom, 2])
                sys.stdout.flush()

def specialrun3(params):
    J = 1/1.602177e-19 # eV
    meter = 1e10 # Angstroms
    s = 1e12 # ps
    mole = 6.022141e23 # atoms
    kg = 1e3*mole # grams/mole

    mSi = 28.09 # grams/mol

    constA = J/(kg*meter**2/s**2)

    dt = float(params["dt"])

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs, tfXi, tfdXi, tfdEdXi = tff.tf_getEF3(tfCoord, tfLattice, params)
    
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA
    tfdEp = tfdEdXi / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, lattice, R, V0 = getRVmmt(mmtFile)

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    V0 = np.zeros_like(V0)
    Vpos = np.zeros_like(R0)
    Vneg = np.zeros_like(R0)
    
    atom=50

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp, Xi, dXi, dEp = sess.run((tfEp, tfFp, tfXi, tfdXi, tfdEp), feed_dict=feedDict)
        Fp = -Fp
        
        out = pd.DataFrame(np.concatenate((R0[atom, 0, None], Ep[atom], Xi[atom], dXi[atom].flatten(), dEp[atom]))[None,:])
        out.to_csv('pd_out.csv', mode='a',header=False,index=False)
        
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2*constA)
        Etot = Epot + Ekin
        
        Vneg[atom, 0] = V0[atom, 0] - 0.5*Fp[atom, 0]/mSi*dt / constA
        Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
        
        R1 = R0 + Vpos * dt
#        R1 = R0 + dt
        
        print(nAtoms)
        print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
        for iAtom in range(nAtoms):
            print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
            
        print("Energies:")
        for iAtom in range(nAtoms):
            print("Energy"+str(iAtom), Ep[iAtom])

        print("Forces:")
        for iAtom in range(nAtoms):
            print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
            sys.stdout.flush()
        
        for iStep in range(1,params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)
                
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp, Xi, dXi, dEp = sess.run((tfEp, tfFp, tfXi, tfdXi, tfdEp), feed_dict=feedDict)
            Fp = -Fp
            
            out = pd.DataFrame(np.concatenate((R0[atom, 0, None], Ep[atom], Xi[atom], dXi[atom].flatten(), dEp[atom]))[None,:])
            out.to_csv('pd_out.csv', mode='a',header=False,index=False)
        
            V0[atom, 0] = Vneg[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            
            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2*constA)
            Etot = Epot + Ekin

            Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            R1 = R0 + Vpos * dt
#            R1 = R0 + dt
            
            print(nAtoms)
            print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
            for iAtom in range(nAtoms):
                print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
                
            print("Energies:")
            for iAtom in range(nAtoms):
                print("Energy"+str(iAtom), Ep[iAtom])
    
            print("Forces:")
            for iAtom in range(nAtoms):
                print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()
                
def specialrun4(params):
    # this one confirms the consistency between numerical and analytical forces

    J = 1/1.602177e-19 # eV
    meter = 1e10 # Angstroms
    s = 1e12 # ps
    mole = 6.022141e23 # atoms
    kg = 1e3*mole # grams/mole

    mSi = 28.09 # grams/mol

    constA = J/(kg*meter**2/s**2)

    dt = float(params["dt"])

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, lattice, R, V0 = getRVmmt(mmtFile)

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    V0 = np.zeros_like(V0)
    Vpos = np.zeros_like(R0)
    Vneg = np.zeros_like(R0)
    
    atom=50

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp
                
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2*constA)
        Etot = Epot + Ekin
        
        Vneg[atom, 0] = V0[atom, 0] - 0.5*Fp[atom, 0]/mSi*dt / constA
        Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
        
        R1 = R0 + Vpos * dt
#        R1 = R0 + dt
        
        print(nAtoms)
        print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
        for iAtom in range(nAtoms):
            print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
            
        print("Energies:")
        for iAtom in range(nAtoms):
            print("Energy"+str(iAtom), Ep[iAtom])

        print("Forces:")
        for iAtom in range(nAtoms):
            print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
            sys.stdout.flush()
        
        for iStep in range(1,params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)
                
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
                    
            V0[atom, 0] = Vneg[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            
            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2*constA)
            Etot = Epot + Ekin

            Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            R1 = R0 + Vpos * dt
#            R1 = R0 + dt
            
            print(nAtoms)
            print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
            for iAtom in range(nAtoms):
                print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
                
            print("Energies:")
            for iAtom in range(nAtoms):
                print("Energy"+str(iAtom), Ep[iAtom])
    
            print("Forces:")
            for iAtom in range(nAtoms):
                print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()

def specialrun5(params):
    # fixing the moving distances and printing the xyz's for Lin-Wang
    # dt is the ∆x for each step

    J = 1/1.602177e-19 # eV
    meter = 1e10 # Angstroms
    s = 1e12 # ps
    mole = 6.022141e23 # atoms
    kg = 1e3*mole # grams/mole

    mSi = 28.09 # grams/mol

    constA = J/(kg*meter**2/s**2)

    dt = float(params["dt"])

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, lattice, R, V0 = getRVmmt(mmtFile)

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    V0 = np.zeros_like(V0)
    Vpos = np.zeros_like(R0)
    Vneg = np.zeros_like(R0)
    
    atom=50

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp
                
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2*constA)
        Etot = Epot + Ekin
        
        Vneg[atom, 0] = V0[atom, 0] - 0.5*Fp[atom, 0]/mSi*dt / constA
        Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
        
        R1 = R0.copy()
#        R1[atom,0] = R0[atom,0] + dt
        R1[atom,0] = R0[atom,0] - dt
        
        print(nAtoms)
        print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
        # for iAtom in range(nAtoms):
        #     print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2])

        for iAtom in range(nAtoms):
            print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
            
        print("Energies:")
        for iAtom in range(nAtoms):
            print("Energy"+str(iAtom), Ep[iAtom])

        print("Forces:")
        for iAtom in range(nAtoms):
            print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
            sys.stdout.flush()
        
        for iStep in range(1,params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            V0[atom, 0] = Vneg[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA

            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2*constA)
            Etot = Epot + Ekin

            Vpos[atom, 0] = V0[atom, 0] + 0.5*Fp[atom, 0]/mSi*dt / constA
            #            R1 = R0 + Vpos * dt
            R1[atom,0] = R0[atom,0] - dt
            
            print(nAtoms)
            print(iStep,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
            # for iAtom in range(nAtoms):
            #     print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2])

            for iAtom in range(nAtoms):
                print("Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])

            print("Energies:")
            for iAtom in range(nAtoms):
                print("Energy"+str(iAtom), Ep[iAtom])

            print("Forces:")
            for iAtom in range(nAtoms):
                print("Force"+str(iAtom), Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()


def specialrun6(params):
    # fixing the moving distances and printing the xyz's for Lin-Wang
    # dt is the ∆x for each step

    mmtFile = "MOVEMENT.dms"

    nCase = 0
    with open(mmtFile, 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        with open(mmtFile, 'r') as datafile:
            for i in range(nCase):
                nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
                feedDict = {tfCoord: R, tfLattice: lattice}
                Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)

                R0 = R.dot(lattice.T)

                print(nAtoms)
                print("Epot(DFT):", np.sum(e), "Epot(NN):", np.sum(Ep))
                for iAtom in range(nAtoms):
                    print("Coord Si"+str(iAtom), R0[iAtom, 0], R0[iAtom, 1], R0[iAtom,2])
                for iAtom in range(nAtoms):
                    print("Force Si"+str(iAtom), -f[iAtom, 0], -f[iAtom, 1], -f[iAtom, 2], Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])

def specialrun7(params):
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    # Tensorflow placeholders
    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))
    tfEngy = tf.placeholder(tf.float32, shape=(None))
    tfFors = tf.placeholder(tf.float32, shape=(None, 3))
    tfLearningRate = tf.placeholder(tf.float32)

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    tfLoss = tf.reduce_mean(tf.squared_difference(tfEs, tfEngy)) + \
             float(params['feRatio']) * tf.reduce_mean(tf.squared_difference(tfFs, tfFors))

    with tf.variable_scope("GradientDescent", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.GradientDescentOptimizer(tfLearningRate).minimize(tfLoss)

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def getError(mCase, tfileName):
        n = 0
        Ese = 0
        Eae = 0
        EseTot = 0
        Fse1 = 0
        Fse2 = 0
        with open(tfileName, "r") as tfile:
            for jCase in range(mCase):
                nAtoms, iIter, lattice, R, f, v, e = pyf.getData(tfile)
                engy = e.reshape([-1, 1])
                feedDict = {
                    tfCoord: R,
                    tfLattice: lattice,
                }
                (Ep, Fp) = sess.run((tfEp, tfFp), feed_dict=feedDict)
                EseTot += (np.sum(Ep) - np.sum(engy)) ** 2
                Ese += np.sum((Ep - engy) ** 2)
                Eae += np.sum(np.abs(Ep - engy))

                Fse1 += np.sum((np.sqrt(np.sum(f ** 2, 1)) - np.sqrt(np.sum(Fp ** 2, 1))) ** 2)
                Fse2 += np.sum((f - Fp) ** 2)

                n += len(engy)
        ErmseTot = np.sqrt(EseTot / mCase)
        Ermse = np.sqrt(Ese / n)
        Emae = Eae / n
        Frmse1 = np.sqrt(Fse1 / n)
        Frmse2 = np.sqrt(Fse2 / (3 * n))
        print("Total Ermse: ", ErmseTot)
        print("Total Ermse per atom:", ErmseTot / nAtoms)
        print("Ermse: ", Ermse)
        print("Emae : ", Emae)
        print("Frmse (magnitude): ", Frmse1)
        print("Frmse (component): ", Frmse2)
        sys.stdout.flush()

    if params["restart"]:
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        print("Model restored")

    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1
    vCase = 0
    if (params["validate"] >= 0) & (params["validationSet"] != ""):
        with open(str(params["validationSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    vCase += 1

    tCase = 0
    if (params["test"] >= 0) & (params["testSet"] != ""):
        with open(str(params["testSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    tCase += 1

    for iEpoch in range(params["epoch"]):
        file = open(str(params["inputData"]), 'r')
        for iCase in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(file)
            engy = e.reshape([-1, 1])
            feedDict = {
                tfCoord: R,
                tfLattice: lattice,
                tfEngy: engy * params['engyScalerA'] + params['engyScalerB'],
                tfFors: f * params['engyScalerA'],
                tfLearningRate: float(params['learningRate']),
            }
            sess.run(tfOptimizer, feed_dict=feedDict)

            (Ei, Fi) = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Ermse = np.sqrt(np.mean((Ei - engy) ** 2))
            Frmse = np.sqrt(np.mean((Fi - f) ** 2))
            print(iEpoch, iCase, "Ermse:", Ermse)
            print(iEpoch, iCase, "Frmse:", Frmse)
            sys.stdout.flush()

        file.close()

        (Ei, Fi) = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Ermse = np.sqrt(np.mean((Ei - engy) ** 2))
        Frmse = np.sqrt(np.mean((Fi - f) ** 2))
        print(iEpoch, "Ermse:", Ermse)
        print(iEpoch, "Frmse:", Frmse)
        sys.stdout.flush()

        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(vCase, str(params['validationSet']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(tCase, str(params['testSet']))
        sys.stdout.flush()

    if params["validate"] == 0:
        getError(vCase, str(params['validationSet']))
    if params["test"] == 0:
        getError(tCase, str(params['testSet']))

    save_path = saver.save(sess, str(params['logDir']) + "/tf.chpt")
    return save_path

def specialrun8(params):
    # fixing the moving distances and printing the xyz's for Lin-Wang
    # dt is the ∆x for each step

    mmtFile = "MOVEMENT.Oct.17"

    # nCase = 0
    # with open(mmtFile, 'r') as datafile:
    #     for line in datafile:
    #         if "Iteration" in line:
    #             nCase += 1

    nCase = 100

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        with open(mmtFile, 'r') as datafile:
            nAtoms1, iIter1, lattice1, R1, f1, v1, e1 = pyf.getData(datafile)
            R0 = R1.dot(lattice1.T)
            x1 = R0

            nAtoms2, iIter2, lattice2, R2, f2, v2, e2 = pyf.getData(datafile)
            R0 = R2.dot(lattice2.T)
            x2 = R0

            for i in range(nCase):

                R20 = R0

                nAtoms3, iIter3, lattice3, R3, f3, v3, e3 = pyf.getData(datafile)
                R0 = R3.dot(lattice3.T)
                x3 = R0

                v = R3 - R1 + 0.5
                v = v - np.floor(v) - 0.5
                v = v.dot(lattice2.T)
                v = v / np.sqrt(np.sum(v ** 2))

                feedDict = {tfCoord: R2, tfLattice: lattice2}
                Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)

                EiRMSE = np.sqrt(np.sum((Ep - e2[:,None]) ** 2) / nAtoms2)
                FiRMSE = np.sqrt(np.sum((Fp - f2) ** 2) / (nAtoms2 * 3))
                crossF = np.sum(Fp * f2) / np.sqrt(np.sum(Fp ** 2) * np.sum(f2 ** 2))

                print(nAtoms2)
                print("Epot(DFT)", np.sum(e2), "Epot(NN)", np.sum(Ep), "F(DFT)", np.sum(f2 * v), "F(NN)", np.sum(Fp * v),
                      "Ei(RMSE)", EiRMSE, "Fi(RMSE)", FiRMSE, "Fnn.Fdft", crossF)

                for iAtom in range(nAtoms2):
                    print("Si"+str(iAtom+1), R20[iAtom, 0], R20[iAtom, 1], R20[iAtom,2], Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2],
                          f2[iAtom,0], f2[iAtom,1], f2[iAtom,2])

                sys.stdout.flush()

                R1 = R2
                nAtoms2 = nAtoms3
                lattice2 = lattice3
                R2 = R3
                f2 = f3
                e2 = e3

def specialrun9(params):
    import pandas as pd
    numFeat = params['n2bBasis'] + params['n3bBasis'] ** 3
    tfFeat = tf.placeholder(tf.float32, shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float32, shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)

    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])

    tfEs_tot = tf.reduce_sum(tf.reshape(tfEs, shape=[-1,256]), axis=1)
    tfEngy_tot = tf.reduce_sum(tf.reshape(tfEngy, shape=[-1,256]), axis=1)

    tfLoss = tf.reduce_mean((tfEs_tot - tfEngy_tot) ** 2)

    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.AdamOptimizer(tfLR).minimize(tfLoss)

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def getError(fFile, eFile):
        featDF = pd.read_csv(fFile, header=None, index_col=False).values
        engyDF = pd.read_csv(eFile, header=None, index_col=False).values
        feedDict2 = {tfFeat: featDF * params['featScalerA'] + params['featScalerB'],
                     tfEngy: engyDF * params['engyScalerA'] + params['engyScalerB']}
        Ep2 = sess.run(tfEs, feed_dict=feedDict2)
        Ep2 = (Ep2 - params['engyScalerB']) / params['engyScalerA']

        Ep2 = np.sum(Ep2.reshape((-1,256)),axis=1)
        engyDF = np.sum(engyDF.reshape((-1,256)),axis=1)

        Ermse = np.sqrt(np.mean((Ep2 - engyDF) ** 2))
        Emae = np.mean(np.abs(Ep2 - engyDF))
        print("Ermse is: ", Ermse)
        print("Ermse/atom is", Ermse/256)
        print("Emae is : ", Emae)
        sys.stdout.flush()

    if params["restart"]:
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        print("Model restored")

    if params['chunkSize'] == 0:
        dfFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False).values
        dfEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False).values

    for iEpoch in range(params['epoch']):
        if params['chunkSize'] > 0:
            pdFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False,
                                 chunksize=int(params['chunkSize']), iterator=True)
            pdEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False,
                                 chunksize=int(params['chunkSize']), iterator=True)

            for pdF in pdFeat:
                pdE = next(pdEngy)
                dfFeat = pdF.values
                dfEngy = pdE.values
                feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'],
                            tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'],
                            tfLR: params['learningRate']}

                sess.run(tfOptimizer, feed_dict=feedDict)
        #                print("running",iEpoch)

        elif params['chunkSize'] == 0:
            feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'],
                        tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'],
                        tfLR: params['learningRate']}

            sess.run(tfOptimizer, feed_dict=feedDict)
        else:
            print("Invalid chunkSize, not within [0,inf]. chunkSize=", params['chunkSize'])

        if iEpoch % 10 == 0:
            Ep, loss = sess.run((tfEs, tfLoss), feed_dict=feedDict)
            Ep = (Ep - params['engyScalerB']) / params['engyScalerA']
            Ep2 = np.sum(Ep.reshape((-1,256)),axis=1)
            dfEngy2 = np.sum(dfEngy.reshape((-1,256)),axis=1)
            Ermse = np.sqrt(np.mean((Ep2 - dfEngy2) ** 2))
            # Ermse = np.sqrt(np.mean((Ep - dfEngy) ** 2))
            print(iEpoch, loss, Ermse)
            sys.stdout.flush()

        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print(str(iEpoch) + "th epoch")
                getError('v' + str(params['featFile']), 'v' + str(params['engyFile']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print(str(iEpoch) + "th epoch")
                getError('t' + str(params['featFile']), 't' + str(params['engyFile']))

    if params["validate"] == 0:
        getError('v' + str(params['featFile']), 'v' + str(params['engyFile']))
    if params["test"] == 0:
        getError('t' + str(params['featFile']), 't' + str(params['engyFile']))

    save_path = saver.save(sess, str(params['logDir']) + "/tf.chpt")
    return save_path

def specialrun10(params):
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    # Tensorflow placeholders
    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))
    tfEngy = tf.placeholder(tf.float32, shape=(None))
    tfFors = tf.placeholder(tf.float32, shape=(None, 3))
    tfLearningRate = tf.placeholder(tf.float32)

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    tfLoss = tf.squared_difference(tf.reduce_sum(tfEs), tf.reduce_sum(tfEngy)) + \
             float(params['feRatio']) * tf.reduce_mean(tf.squared_difference(tfFs, tfFors))

    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.AdamOptimizer(tfLearningRate).minimize(tfLoss)

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def getError(mCase, tfileName):
        n = 0
        Ese = 0
        Eae = 0
        EseTot = 0
        Fse1 = 0
        Fse2 = 0
        with open(tfileName, "r") as tfile:
            for jCase in range(mCase):
                nAtoms, iIter, lattice, R, f, v, e = pyf.getData(tfile)
                engy = e.reshape([-1, 1])
                feedDict = {
                    tfCoord: R,
                    tfLattice: lattice,
                }
                (Ep, Fp) = sess.run((tfEp, tfFp), feed_dict=feedDict)
                EseTot += (np.sum(Ep) - np.sum(engy)) ** 2
                Ese += np.sum((Ep - engy) ** 2)
                Eae += np.sum(np.abs(Ep - engy))

                Fse1 += np.sum((np.sqrt(np.sum(f ** 2, 1)) - np.sqrt(np.sum(Fp ** 2, 1))) ** 2)
                Fse2 += np.sum((f - Fp) ** 2)

                n += len(engy)
        ErmseTot = np.sqrt(EseTot / mCase)
        Ermse = np.sqrt(Ese / n)
        Emae = Eae / n
        Frmse1 = np.sqrt(Fse1 / n)
        Frmse2 = np.sqrt(Fse2 / (3 * n))
        print("Total Ermse: ", ErmseTot)
        print("Total Ermse per atom:", ErmseTot / nAtoms)
        print("Ermse: ", Ermse)
        print("Emae : ", Emae)
        print("Frmse (magnitude): ", Frmse1)
        print("Frmse (component): ", Frmse2)
        sys.stdout.flush()

    if params["restart"]:
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        print("Model restored")

    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1
    vCase = 0
    if (params["validate"] >= 0) & (params["validationSet"] != ""):
        with open(str(params["validationSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    vCase += 1

    tCase = 0
    if (params["test"] >= 0) & (params["testSet"] != ""):
        with open(str(params["testSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    tCase += 1

    for iEpoch in range(params["epoch"]):
        file = open(str(params["inputData"]), 'r')
        for iCase in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(file)
            engy = e.reshape([-1, 1])
            feedDict = {
                tfCoord: R,
                tfLattice: lattice,
                tfEngy: engy * params['engyScalerA'] + params['engyScalerB'],
                tfFors: f * params['engyScalerA'],
                tfLearningRate: float(params['learningRate']),
            }
            sess.run(tfOptimizer, feed_dict=feedDict)

            (Ei, Fi) = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Ermse = np.sqrt(np.mean((Ei - engy) ** 2))
            Frmse = np.sqrt(np.mean((Fi - f) ** 2))
            print(iEpoch, iCase, "Ermse:", Ermse)
            print(iEpoch, iCase, "Frmse:", Frmse)
            sys.stdout.flush()

        file.close()

        (Ei, Fi) = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Ermse = np.sqrt(np.mean((Ei - engy) ** 2))
        Frmse = np.sqrt(np.mean((Fi - f) ** 2))
        print(iEpoch, "Ermse:", Ermse)
        print(iEpoch, "Frmse:", Frmse)
        sys.stdout.flush()

        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(vCase, str(params['validationSet']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(tCase, str(params['testSet']))
        sys.stdout.flush()

    if params["validate"] == 0:
        getError(vCase, str(params['validationSet']))
    if params["test"] == 0:
        getError(tCase, str(params['testSet']))

    save_path = saver.save(sess, str(params['logDir']) + "/tf.chpt")
    return save_path

def specialrun11(params):
    # special MD run
    J = 1 / 1.602177e-19  # eV
    meter = 1e10  # Angstroms
    s = 1e12  # ps
    mole = 6.022141e23  # atoms
    kg = 1e3 * mole  # grams/mole

    mSi = 28.09  # grams/mol

    constA = J / (kg * meter ** 2 / s ** 2)

    bohr = 0.529177249

    dt = float(params["dt"])

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        # nAtoms, lattice, R, F0, V0 = getRFVmmt(mmtFile)
        nAtoms, iIter, lattice, R, F0, V0 = getData11(mmtFile)

    # print(V0)
    # print("Initial Kinetic Energy: V0", np.sum(0.5 * mSi * V0 ** 2 * constA))
    # V0 = V0 * 1000
    # print("Initial Kinetic Energy: V0*1000", np.sum(0.5 * mSi * V0 ** 2 * constA))
    # V0 = V0*1000
    # print(V0)

    V0 = V0*1000*bohr

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    # V0 = V
    # Vpos = np.zeros_like(R0)
    # Vneg = np.zeros_like(R0)

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp

        # for iAtom in range(len(R1)):
        #     print("Si", Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2], F0[iAtom, 0], F0[iAtom,1], F0[iAtom,2])

        # Vpos = 0.5 * Fp / mSi * dt * constA

        Vneg = V0 - 0.5*Fp/mSi*dt / constA

        Vpos = Vneg + Fp/mSi*dt / constA

        # V0 = Vneg + 0.5*Fp/mSi*dt * constA

        R1 = R0 + Vpos * dt

        Epot = np.sum(Ep)
        Ekin = np.sum(0.5 * mSi * V0 ** 2 * constA)
        Etot = Epot + Ekin

        print(nAtoms)
        print(0, "Epot=", "{:.12f}".format(Epot), "Ekin=", "{:.12f}".format(Ekin), "Etot=", "{:.12f}".format(Etot))
        for iAtom in range(len(R1)):
            print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2], Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
        sys.stdout.flush()

        for iStep in range(1, params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T

            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            V0 = Vneg + 0.5 * Fp / mSi * dt / constA
            Vpos = Vneg + Fp / mSi * dt / constA
            R1 = R0 + Vpos * dt

            Epot = np.sum(Ep)
            Ekin = np.sum(0.5 * mSi * V0 ** 2 * constA)
            Etot = Epot + Ekin

            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):
                print(nAtoms)
                print(iStep, "Epot=", "{:.12f}".format(Epot), "Ekin=", "{:.12f}".format(Ekin), "Etot=",
                      "{:.12f}".format(Etot))
                for iAtom in range(len(R1)):
                    print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2], Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()
