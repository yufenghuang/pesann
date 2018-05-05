import numpy as np
import py_func as pyf
import np_func as npf
import tensorflow as tf
import tf_func as tff

import pandas as pd

import os
import sys

import re

# functions used for Gaussian descriptors
def fc(Rij, Rc):
    Rij_ = np.array(Rij)
    R_ = np.zeros(Rij_.shape)
    R_[Rij_<Rc] = 0.5*(np.cos(Rij_[Rij_<Rc]*np.pi/Rc)+1)
    return R_

def g1(Rij,Rs_,eta_, Rc):
    return np.exp(-eta_*(Rij-Rs_)**2)*fc(Rij, Rc)

def g2(Rij,Rik,Rjk,eta_,zeta_,ll_, Rc): ##ll: lambda
    cos = (Rij**2+Rik**2-Rjk**2)/(2*Rij*Rik)
    return 2.**(1-zeta_)*(1+ll_*cos)**zeta_ * np.exp(-eta_*(Rij**2+Rik**2+Rjk**2))* \
            fc(Rij, Rc)*fc(Rik, Rc)*fc(Rjk, Rc)

def getGaussianFeats(ll, eta, zeta, Rs, Rc, Ri, Di, Dj, Dc):
    feats = np.zeros((len(Ri), eta.size * Rs.size + ll.size * zeta.size * eta.size))
    for i in range(Rs.size):
        for j in range(eta.size):
            ifeat = i * eta.size + j
            G1 = np.zeros(Ri.shape)
            G1[Ri > 0.] = g1(Ri[Ri > 0.], Rs[i], eta[j], Rc)
            feats[:, ifeat] = G1.sum(axis=1)

    for i in range(ll.size):
        for j in range(eta.size):
            for k in range(zeta.size):
                ifeat = Rs.size * eta.size + i * eta.size * zeta.size + j * zeta.size + k
                G2 = np.zeros(Dc.shape)
                G2[Dc > 0.] = g2(Di[Dc > 0.], Dj[Dc > 0.], Dc[Dc > 0.], eta[j], zeta[k], ll[i], Rc)
                feats[:, ifeat] = G2.sum(axis=2).sum(axis=1)
    return feats

#   Generate features for the Gaussian symmetry functions
def specialTask01(engyFile, featFile, inputData, params):
    Rc = float(params["dcut"])
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size

    # featFile = str(params["featFile"])
    # engyFile = str(params["engyFile"])

    nCase = 0
    with open(inputData, 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    with open(inputData, 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
            idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params["dcut"]))
            Rhat, Ri, Dc = npf.getStruct(coord)
            Di = Ri[:,:,None] * np.ones(maxNb)
            Di[Dc == 0] = 0
            Dj = np.transpose(Di, [0,2,1])
            feat = getGaussianFeats(ll, eta, zeta, Rs, Rc, Ri, Di, Dj, Dc)
            engy = e.reshape([-1, 1])
            pd.DataFrame(feat).to_csv(featFile, mode='a', header=False, index=False)
            pd.DataFrame(engy).to_csv(engyFile, mode='a', header=False, index=False)

# Train the NN with Gaussian symmetry functions
def specialTask02(params):
    Rc = float(params["dcut"])
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size
    numFeat = num2bFeat + num3bFeat

    tfFeat = tf.placeholder(tf.float32, shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float32, shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)

    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])

    tfLoss = tf.reduce_mean((tfEs - tfEngy) ** 2)

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

        # print(Ep2[:,:])
        # print(engyDF[:,:])

        Ermse = np.sqrt(np.mean((Ep2.reshape(-1) - engyDF.reshape(-1)) ** 2))
        Emae = np.mean(np.abs(Ep2.reshape(-1) - engyDF.reshape(-1)))
        print("Ermse is: ", Ermse)
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
            Ermse = np.sqrt(np.mean((Ep - dfEngy) ** 2))
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

# Train the NN with the total energy (just the energy, no forces)
# function copied from specialrun09
def specialTask03(params):
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

# Train the NN with the total energy and forces
# function copied from specialrun10
def specialTask04(params):
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

# Export the predicted values of Ei, Etot, and Fi from a given input.
# The corresponding outputs are Ei_DFTvsNN, Etot_DFTvsNN, Fi_DFTvsNN.
# There are two columns for each file, with the first corresponding to the DFT values
# and the second corresponding to the NN values.
def specialTask05(params):
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

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if params["restart"]:
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        print("Model restored")
    else:
        print("Need to use the restart keyword to load the NN model")
        exit()

    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    fileEi = open("Ei_DFTvsNN", 'w')
    fileEtot = open("Etot_DFTvsNN", 'w')
    fileFi = open("Fi_DFTvsNN", 'w')

    with open(str(params["inputData"]), 'r') as file:
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

            (Ei, Fi) = sess.run((tfEp, tfFp), feed_dict=feedDict)

            Ei=Ei.reshape(-1)
            engy = engy.reshape(-1)

            fileEtot.write(str(np.sum(engy)) + " " + str(np.sum(Ei)) + "\n")
            for iAtom in range(nAtoms):
                fileEi.write(str(engy[iAtom]) + " " + str(Ei[iAtom]) + "\n")
                fileFi.write(" ".join(f[iAtom].astype(str)) + " " + " ".join(Fi[iAtom].astype(str)) + "\n")

    fileEi.close()
    fileEtot.close()
    fileFi.close()

# Obtain DFT vs NN along an MD trajectory
def specialTask06(params):

    mmtFile = params["inputData"]

    # nCase = 0
    # with open(mmtFile, 'r') as datafile:
    #     for line in datafile:
    #         if "Iteration" in line:
    #             nCase += 1

    nCase = params['epoch']

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

    if (params["repulsion"] == "1/R") or (params["repulsion"] == "1/R12") or (params["repulsion"] == "exp(-R)"):
        tfEs, tfFs = tff.tf_getEF_repulsion(tfCoord, tfLattice, params)
    else:
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

# Thermal conductivity MD run:

def MDstep(R0, Vneg, dt, Fp):

    J = 1 / 1.602177e-19  # eV
    meter = 1e10  # Angstroms
    s = 1e12  # ps
    mole = 6.022141e23  # atoms
    kg = 1e3 * mole  # grams/mole

    mSi = 28.09  # grams/mol

    constA = J / (kg * meter ** 2 / s ** 2)

    V0 = Vneg + 0.5 * Fp / mSi * dt / constA
    Vpos = Vneg + Fp / mSi * dt / constA
    R1 = R0 + Vpos * dt

    return R1, Vpos, V0


# cleaned up version of NVE
def specialTask07(params):
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

    if (params["repulsion"] == "1/R") or (params["repulsion"] == "1/R12") or (params["repulsion"] == "exp(-R)"):
        tfEs, tfFs = tff.tf_getEF_repulsion(tfCoord, tfLattice, params)
    else:
        tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        # nAtoms, lattice, R, F0, V0 = getRFVmmt(mmtFile)
        nAtoms, iIter, lattice, R, F0, V0 = getData11(mmtFile)

    V0 = V0*1000*bohr
    R1 = R.dot(lattice.T)

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp

        Vpos = V0 - 0.5*Fp/mSi*dt / constA

        for iStep in range(params["epoch"]):
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



def specialTask08(params):

    from scipy.special import erfc

    def m(x):
        return erfc(12*np.abs(x-0.5)-3)/2

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

    tfFln = tff.tf_getFln(tfCoord, tfLattice, params) / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)

    V0 = V0*1000*bohr

    R0 = R.dot(lattice.T)

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp, Fln = sess.run((tfEp, tfFp, tfFln), feed_dict=feedDict)
        Fp = -Fp
        Fln = -Fln
        idxNb, Rln, maxNb, nAtoms = npf.getNb(R, lattice, float(params['dcut']))
        adjMat, Fln = npf.adjList2adjMat(idxNb, Fln)
        _, Fln = npf.adjMat2adjList(adjMat, Fln.transpose([1,0,2]))
        Rln = -Rln
        Jx = np.sum(Ep*V0[:,0]) + np.sum(Rln[:,:,0] * np.sum(Fln*V0[:,None,:],axis=2))

        Vneg = V0 - 0.5*Fp/mSi*dt / constA

        Vpos = Vneg + Fp/mSi*dt / constA

        R1 = R0 + Vpos * dt
        Rhalf = R0 + m(R0[:,0]/lattice[0,0]) * Vpos * dt



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

'''

def old_specialTask01(params):
    #   Generate features for the Gaussian symmetry functions
    #
    Rc = 6.2
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size

    featFile = str(params["featFile"])
    engyFile = str(params["engyFile"])

    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    with open(str(params["inputData"]), 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
            idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params["dcut"]))
            Rhat, Ri, Dc = npf.getStruct(coord)
            Di = Ri[:,:,None] * np.ones(256)
            Di[Dc == 0] = 0
            Dj = np.transpose(Di, [0,2,1])
            feat = getGaussianFeats(ll, eta, zeta, Rs, Ri, Di, Dj, Dc)
            engy = e.reshape([-1, 1])
            pd.DataFrame(feat).to_csv(featFile, mode='a', header=False, index=False)
            pd.DataFrame(engy).to_csv(engyFile, mode='a', header=False, index=False)


def old_specialTask02(params):
    #   Train for the Gaussian basis
    #

    Rc = 6.2
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size
    numFeat = num2bFeat + num3bFeat

    #   NN

    tfFeat = tf.placeholder(tf.float32, shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float32, shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)

    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    tfLoss = tf.reduce_mean((tfEs - tfEngy) ** 2)

    return 0

'''
