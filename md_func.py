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

def readXYZ(xyzFile):
    lattice_ = np.zeros(9)
    with open(xyzFile,'r') as xyzFH: #FH: file handler
        nAtoms = int(xyzFH.readline())
        line2 = xyzFH.readline().split()
        for i in range(len(line2)):
            if line2[i] == 'lattice':
                lattice_ = np.array([float(x) for x in line2[i+1:i+10]]).reshape((3,3))
        R_ = np.zeros((nAtoms, 3))
        V_ = np.zeros((nAtoms, 3))
        F_ = np.zeros((nAtoms, 3))
        for i in range(nAtoms):
            line = np.array([x for x in xyzFH.readline().split()[1:10]])
            R_[i] = line[:3]
            if len(line)>3:
                V_[i] = line[3:6]
            if len(line)>6:
                F_[i] = line[6:9]
    return nAtoms, lattice_, R_, V_, F_

def read_structure(fileName, format):
    if format == "xyz":
        nAtoms, lattice, R, V0, F = readXYZ(fileName)
        R = np.linalg.solve(lattice, R.T).T
        R = R - np.floor(R)

    elif format == "mmt":
        with open(fileName, 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0*1000*bohr
    else:
        print("Unrecognized input format!!", format)

    return nAtoms, lattice, R, V0

# Andersen thermostat
def andersen(params):
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(params["fracMem"])
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0, lattice):
            R = np.linalg.solve(lattice, R0.T).T
            R = R - np.floor(R)
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

        dt = float(params["dt"])
        coll_prob = float(params["coll_prob"])
        Tend = np.sum(0.5 * mSi * V1 ** 2 * constA) / J /(3/2*nAtoms) / kB
        Tbegin = Tend

        if params["Tbegin"] >= 0:
            Tbegin = params["Tbegin"]
            V1 = reassignV(V1, 2, Tbegin)

        if params["Tend"] >= 0:
            Tend = params["Tend"]

        # change temperature from Tbegin to Tend
        assert params["epoch"] > 2, "the number of epochs must be greater than 2..." \
                                     "The current value is " + str(params["epoch"])

        dT = (Tend - Tbegin) / (params["epoch"] - 1)
        T1 = Tbegin
        for iStep in range(params["epoch"]):
            R0, E0, V0, F0, T0 = R1, E1, V1, F1, T1
            R1, E1, V1, F1 = andersenIntegrator(R0, V0, F0)
            T1 = T0 + dT
            V1 = reassignV(V1, coll_prob, T1)

            Epot, Ekin, Etot = getMDEnergies(E0, V0)

            Temp = Ekin / (3 / 2 * nAtoms) * eV2J / kB

            if (iStep % int(params["nstep"]) == 0):
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

            if ((iStep + 1) % int(params["nstep"] * 10) == 0):
                with open("md.xyz", 'w') as xyzFile:
                    xyzFile.write(str(nAtoms) + "\n")
                    xyzFile.write(
                        "iStep " + str(iStep) + " lattice " + " ".join([str(x) for x in lattice.reshape(-1)]) +
                        " Epot " + "{:.12f}".format(Epot) + " Ekin " + "{:.12f}".format(Ekin) +
                        " Etot " + "{:.12f}".format(Etot) + " Temp " + "{:.12f}".format(Temp) +
                        "\n")
                    for iAtom in range(nAtoms):
                        xyzFile.write(
                            "Si " + str(R0[iAtom, 0]) + " " + str(R0[iAtom, 1]) + " " + str(R0[iAtom, 2]) +
                            " " + str(V0[iAtom, 0]) + " " + str(V0[iAtom, 1]) + " " + str(V0[iAtom, 2]) +
                            " " + str(F0[iAtom, 0]) + " " + str(F0[iAtom, 1]) + " " + str(F0[iAtom, 2]) + "\n")


# cleaned up version of NVE
def NVE(params):
    # special MD run
    dt = float(params["dt"])

    # setup Tensorflow
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)
    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))
    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(params["fracMem"])
    with tf.Session(config=config) as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp

        def getEF(R0, lattice):
            R = np.linalg.solve(lattice, R0.T).T
            R = R - np.floor(R)
            R0 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            return R0, Ep, Fp

        # initialize the atomic positions and velocities
        nAtoms, lattice, R, V0 = read_structure(params["inputData"], params["format"])
        R1, Ep, Fp = getEF(R.dot(lattice.T), lattice)
        Vpos = V0 - 0.5 * Fp / mSi * dt / constA

        # MD loop
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp = getEF(R0, lattice)
            R1, Vpos, V0 = vverlet(R0, Vneg, dt, Fp)

            Epot, Ekin, Etot = getMDEnergies(Ep, V0)
            Temp = Ekin / (3 / 2 * nAtoms) * eV2J / kB

            # printing the output
            if (iStep % int(params["nstep"]) == 0):
                print(nAtoms)
                print("iStep", iStep,
                      "lattice", " ".join([str(x) for x in lattice.reshape(-1)]),
                      "Epot", "{:.12f}".format(Epot), "Ekin", "{:.12f}".format(Ekin),
                      "Etot", "{:.12f}".format(Etot), "Temp", "{:.12f}".format(Temp))
                for iAtom in range(nAtoms):
                    print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2],
                          V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2],
                          Fp[iAtom, 0], Fp[iAtom, 1], Fp[iAtom, 2])
                sys.stdout.flush()

            if ((iStep+1) % int(params["nstep"]*10) == 0):
                with open("md.xyz", 'w') as xyzFile:
                    xyzFile.write(str(nAtoms) + "\n")
                    xyzFile.write("iStep " + str(iStep) + " lattice " + " ".join([str(x) for x in lattice.reshape(-1)]) +
                                  " Epot " + "{:.12f}".format(Epot) + " Ekin " + "{:.12f}".format(Ekin) +
                                  " Etot " + "{:.12f}".format(Etot) + " Temp " + "{:.12f}".format(Temp) +
                                  "\n")
                    for iAtom in range(nAtoms):
                        xyzFile.write("Si " + str(R0[iAtom, 0]) + " " + str(R0[iAtom, 1]) + " " + str(R0[iAtom, 2]) +
                                      " " + str(V0[iAtom, 0]) + " " + str(V0[iAtom, 1]) + " " + str(V0[iAtom, 2]) +
                                      " " + str(Fp[iAtom, 0]) + " " + str(Fp[iAtom, 1]) + " " + str(Fp[iAtom, 2]) + "\n")

# calculate heat current J(t) for the calculation of the
# heat current auto-correlation function (HCACF)
def hcacf(params):
    # setup Tensorflow
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)
    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))
    tfEs, tfFs1, tfFs2 = tff.tf_getEFln(tfCoord, tfLattice, params)

    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = (tfFs1 + tf.reduce_sum(tfFs2, axis=1)) / tfEngyA
    tfFpq = tfFs2 / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(params["fracMem"])
    with tf.Session(config=config) as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0, lattice):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp, Fpq = sess.run((tfEp, -tfFp, -tfFpq), feed_dict=feedDict)

            return R.dot(lattice.T), Ep, Fp, Fpq

        def getJ(R0, V0, Fln):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))

            Vln = np.zeros((nAtoms, maxNb, 3))
            Vln[idxNb>0] = V0[idxNb[idxNb>0]-1]

            Jpot = np.sum(Rln * np.sum(Fln * Vln, axis=2)[:, :, None], axis=1)

            return Jpot

        # initialize the atomic positions and velocities
        dt = float(params["dt"])
        nAtoms, lattice, R, V0 = read_structure(params["inputData"], params["format"])
        R1, Ep, Fp, _ = getEF(R.dot(lattice.T), lattice)
        Vpos = V0 - 0.5 * Fp / mSi * dt / constA

        # Thermal conductivity MD loop
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Fpq = getEF(R0, lattice)
            R1, Vpos, V0 = vverlet(R0, Vneg, dt, Fp)

            # printing the output
            Ek = np.sum(0.5 * mSi * V0 ** 2 * constA, axis=1)
            J0 = (Ep+Ek[:,None])*V0
            J1 = getJ(R0, V0, Fpq)
            Jt = np.sum(J0 + J1, axis=0)

            Epot, Ekin, Etot = getMDEnergies(Ep, V0)
            Temp = Ekin / (3 / 2 * nAtoms) * eV2J / kB

            print(iStep, "Epot", "{:.12f}".format(Epot), "Ekin", "{:.12f}".format(Ekin),
                  "Etot", "{:.12f}".format(Etot),
                  "Jx", "{:.12f}".format(Jt[0]), "Jy", "{:.12f}".format(Jt[1]), "Jz", "{:.12f}".format(Jt[2]))

            if ((iStep+1) % int(params["nstep"]*10) == 0):
                with open("md.xyz", 'w') as xyzFile:
                    xyzFile.write(str(nAtoms) + "\n")
                    xyzFile.write("iStep " + str(iStep) + " lattice " + " ".join([str(x) for x in lattice.reshape(-1)]) +
                                  " Epot " + "{:.12f}".format(Epot) + " Ekin " + "{:.12f}".format(Ekin) +
                                  " Etot " + "{:.12f}".format(Etot) + " Temp " + "{:.12f}".format(Temp) +
                                  " Jx " + "{:.12f}".format(Jt[0]) +
                                  " Jy " + "{:.12f}".format(Jt[1]) +
                                  " Jz " + "{:.12f}".format(Jt[2]) + "\n")
                    for iAtom in range(nAtoms):
                        xyzFile.write("Si " + str(R0[iAtom, 0]) + " " + str(R0[iAtom, 1]) + " " + str(R0[iAtom, 2]) +
                                      " " + str(V0[iAtom, 0]) + " " + str(V0[iAtom, 1]) + " " + str(V0[iAtom, 2]) +
                                      " " + str(Fp[iAtom, 0]) + " " + str(Fp[iAtom, 1]) + " " + str(Fp[iAtom, 2]) + "\n")
