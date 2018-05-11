import numpy as np
import py_func as pyf
import np_func as npf
import tensorflow as tf
import tf_func as tff

import pandas as pd

import os
import sys

import re

# MD constants:
J = 1 / 1.602177e-19  # eV
meter = 1e10  # Angstroms
s = 1e12  # ps
mole = 6.022141e23  # atoms
kg = 1e3 * mole  # grams/mole
mSi = 28.09  # grams/mol
constA = J / (kg * meter ** 2 / s ** 2)

bohr = 0.529177249
kB = 1.38e-23 # J/K

eV2J = 1.602177e-19 # converting eV to Joules

#velocity verlet
def vverlet(R0, Vneg, dt, Fp):
    V0 = Vneg + 0.5 * Fp / mSi * dt / constA
    Vpos = Vneg + Fp / mSi * dt / constA
    R1 = R0 + Vpos * dt
    return R1, Vpos, V0

def getMDEnergies(Ep, V0):
    Epot = np.sum(Ep)
    Ekin = np.sum(0.5 * mSi * V0 ** 2 * constA)
    Etot = Epot + Ekin
    return Epot, Ekin, Etot

def printXYZ(iStep, R0, V0, Fp, Ep, *other):
    Epot, Ekin, Etot = getMDEnergies(Ep, V0)
    print(len(R0))
    print(iStep, "Epot", "{:.12f}".format(Epot), "Ekin", "{:.12f}".format(Ekin), "Etot",
          "{:.12f}".format(Etot), " ".join(["{:.12f}".format(float(o)) for o in other]))
    for iAtom in range(len(R0)):
        print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2], Fp[iAtom, 0],
              Fp[iAtom, 1], Fp[iAtom, 2])
    sys.stdout.flush()

def read_structure(fileName, format):
    if format == "xyz":
        pass
    elif format == "mmt":
        with open(fileName, 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0*1000*bohr
    else:
        print("Unrecognized input format!!", format)

    return nAtoms, lattice, R, V0

# Andersen thermostat
def andersen(params):
    dt = float(params["dt"])
    coll_prob = float(params["coll_prob"])

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

        # define the function for Ep and Fp
        def getEF(R0, lattice):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            return R0, Ep, Fp

        # reassign velocities based on maxwell-boltzmann distribution
        def reassignV(velocities, prob, T):
            rv = np.random.rand(len(velocities))
            sigmaV = np.sqrt(kB * T/(mSi/kg))*meter/s
            velocities[rv < prob] = np.random.randn(np.sum(rv < prob), 3) * sigmaV
            return velocities

        def andersenIntegrator(R0, V0, F0):

            R1 = R0 + V0*dt + F0/(2*mSi) * dt**2 /constA
            R1, E1, F1 = getEF(R1, lattice)
            V1 = V0 + (F1+F0)/(2*mSi) * dt /constA

            return R1, E1, V1, F1

        # initialize the atomic positions and velocities
        nAtoms, lattice, R, V1 = read_structure(params["inputData"], params["format"])
        R1 = R.dot(lattice.T)
        feedDict = {tfCoord: R, tfLattice: lattice}
        E1, F1 = sess.run((tfEp, -tfFp), feed_dict=feedDict)

        Tend = np.sum(0.5 * mSi * V1 ** 2 * constA) / J /(3/2*nAtoms) / kB
        Tbegin = Tend

        if params["Tbegin"] >= 0:
            Tbegin = params["Tbegin"]
            V1 = reassignV(V1, 2, Tbegin)

        if params["Tend"] >= 0:
            Tend = params["Tend"]

        # change temperature from Tbegin to Tend
        if params["dTstep"] > 2:
            dT = (Tend - Tbegin) / (params["dTstep"] - 1)

            T1 = Tbegin
            for iStep in range(params["dTstep"]):
                R0, E0, V0, F0, T0 = R1, E1, V1, F1, T1
                R1, E1, V1, F1 = andersenIntegrator(R0, V0, F0)
                T1 = T0 + dT
                V1 = reassignV(V1, coll_prob, T1)

                Epot, Ekin, Etot = getMDEnergies(E0, V0)

                Temp = Ekin / (3 / 2 * nAtoms) * eV2J / kB

                if ((iStep + 1) % int(params["nstep"]) == 0):
                    print(nAtoms)
                    print("iStep", iStep,
                          "lattice", " ".join([str(x) for x in lattice.reshape(-1)]),
                          "Epot", "{:.12f}".format(Epot), "Ekin", "{:.12f}".format(Ekin),
                          "Etot", "{:.12f}".format(Etot), "Temp", "{:.12f}".format(Temp),
                          "Target", "{:.12f}".format(T0))
                    for iAtom in range(nAtoms):
                        print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2],
                              V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2],
                              F0[iAtom, 0], F0[iAtom, 1], F0[iAtom, 2])
                    sys.stdout.flush()

        # MD loop
        for iStep in range(params["epoch"]):
            R0, E0, V0, F0 = R1, E1, V1, F1
            R1, E1, V1, F1 = andersenIntegrator(R0, V0, F0)
            V1 = reassignV(V1, coll_prob, Tend)
            Epot, Ekin, Etot = getMDEnergies(E0, V0)

            Temp = Ekin/(3/2*nAtoms)*eV2J/kB

            if ((iStep + 1) % int(params["nstep"]) == 0):
                print(nAtoms)
                print("iStep", iStep,
                      "lattice", " ".join([str(x) for x in lattice.reshape(-1)]),
                      "Epot", "{:.12f}".format(Epot), "Ekin", "{:.12f}".format(Ekin),
                      "Etot", "{:.12f}".format(Etot), "Temp", "{:.12f}".format(Temp),
                      "Target", "{:.12f}".format(Tend))
                for iAtom in range(nAtoms):
                    print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2],
                          V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2],
                          F0[iAtom, 0], F0[iAtom, 1], F0[iAtom, 2])
                sys.stdout.flush()