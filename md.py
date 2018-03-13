# -*- coding: utf-8 -*-

import numpy as np
import py_func as pyf
import tensorflow as tf
import tf_func as tff

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

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float64)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float64)

    tfCoord = tf.placeholder(tf.float64, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float64, shape=(3, 3))

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
    
    iAtom=50

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp
        
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2/constA)
        Etot = Epot + Ekin
        
        Vneg[iAtom, 0] = V0[iAtom, 0] - 0.5*Fp[iAtom, 0]/mSi*dt * constA
        Vpos[iAtom, 0] = V0[iAtom, 0] + 0.5*Fp[iAtom, 0]/mSi*dt * constA
        
        R1 = R0 + Vpos * dt
        
        print(nAtoms)
        print(0,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
        print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
        print("Forces: ", Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
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
        
            V0[iAtom, 0] = Vneg[iAtom, 0] + 0.5*Fp[iAtom, 0]/mSi*dt * constA
            
            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2/constA)
            Etot = Epot + Ekin

            Vpos[iAtom, 0] = V0[iAtom, 0] + 0.5*Fp[iAtom, 0]/mSi*dt * constA
            R1 = R0 + Vpos * dt
            
            print(nAtoms)
            print(iStep,"Epot=", Epot, "Ekin=",Ekin, "Etot=",Etot)
            print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
            print("Forces: ", Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
            sys.stdout.flush()

