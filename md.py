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
