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

# MD constants:
J = 1 / 1.602177e-19  # eV
meter = 1e10  # Angstroms
s = 1e12  # ps
mole = 6.022141e23  # atoms
kg = 1e3 * mole  # grams/mole
mSi = 28.09  # grams/mol
constA = J / (kg * meter ** 2 / s ** 2)
bohr = 0.529177249

def MDstep(R0, Vneg, dt, Fp):
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
    print(iStep, "Epot=", "{:.12f}".format(Epot), "Ekin=", "{:.12f}".format(Ekin), "Etot=",
          "{:.12f}".format(Etot), " ".join(["{:.12f}".format(float(o)) for o in other]))
    for iAtom in range(len(R0)):
        print("Si", R0[iAtom, 0], R0[iAtom, 1], R0[iAtom, 2], V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2], Fp[iAtom, 0],
              Fp[iAtom, 1], Fp[iAtom, 2])
    sys.stdout.flush()


# cleaned up version of NVE
def specialTask07(params):
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

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            return R0, Ep, Fp

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        # MD loop
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):
                printXYZ(iStep, R0, V0, Fp, Ep)


# Thermal conductivity (MD method)
def specialTask08(params):
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
    tfFln = tff.tf_getFln(tfCoord, tfLattice, params) / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}
            Es, Ep, Fp = sess.run(((tfEs-0.5)/tfEngyA, tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            return R.dot(lattice.T), Ep, Fp, Es

        def getJ(R0, V0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}

            Fln = sess.run(-tfFln, feed_dict=feedDict)

            idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
            adjMat, Fln = npf.adjList2adjMat(idxNb, Fln)
            _,_,Fln = npf.adjMat2adjList(adjMat, Fln.transpose([1, 0, 2]))
            Rln = -Rln

            Jpot = np.sum(Rln * np.sum(Fln * V0[:, None, :], axis=2)[:, :, None], axis=1)

            return Jpot

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        # MD equilibrium loop
        for iStep in range(1000):
            R0 = R1
            Vneg = Vpos
            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, 0.001, Fp)
            # printing the output
            if (iStep % 10 == 0) or \
                    ((iStep % 10 != 0) & (iStep == params["epoch"] - 1)):
                printXYZ(iStep, R0, V0, Fp, Ep)

        # Thermal conductivity MD loop
        Jt0 = 0
        JxOut = open("Jx", 'w')
        JyOut = open("Jy", 'w')
        JzOut = open("Jz", 'w')

        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):

                # only calculate <J(t)J(0)> when printing
                Ek = np.sum(0.5 * mSi * V0 ** 2 * constA, axis=1)
                J0 = (Ep+Ek[:,None])*V0
                J1 = getJ(R0, V0)
                Jt = J0 + J1
                if iStep == 0:
                    Jt0 = Jt

                printXYZ(iStep, R0, V0, Fp, Ep, np.sum(Jt0*Jt, axis=0)[0])
                JxOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 0]]) + "\n")
                JyOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 1]]) + "\n")
                JzOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 2]]) + "\n")

# Improved heat conductivity calculation
def specialTask09(params):
    # special MD run
    dt = float(params["dt"])

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

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
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

            adjMat, FlnMat, RlnMat, VlnMat = npf.adjList2adjMat(idxNb, Fln, Rln, Vln)
            _,_,FlnList = npf.adjMat2adjList(adjMat, FlnMat.transpose([1, 0, 2]))

            Jpot2 = np.sum(-Rln * np.sum(FlnList * V0[:, None, :], axis=2)[:, :, None], axis=1)

            Jpot3 = np.sum(np.sum(FlnMat * V0[None, :, :],axis=2)[:,:,None]*RlnMat, axis=1)

            V0ln = V0[None,:,:] * np.ones((nAtoms,1 ,1))
            V0ln[adjMat<1] = np.zeros(3)

            adjMat2, FlnMat2 = npf.adjList2adjMat(idxNb, FlnList)
            Jpot4 = np.sum(np.sum(FlnMat2 * V0[:,None,:], axis=2)[:,:, None] * -RlnMat, axis=1)


            # check Jpot, Jpot2, Jpot3
            print("====================")
            print('Jpot-Jpot2', np.max(np.abs(Jpot-Jpot2)))
            # print(Jpot-Jpot2)
            print("Jpot-Jpot3", np.max(np.abs(Jpot-Jpot3)))
            # print(Jpot-Jpot3)
            print("Jpot2-Jpot3", np.max(np.abs(Jpot2-Jpot3)))
            # print(Jpot2-Jpot3)

            print("Jpot2-Jpot4", np.max(np.abs(Jpot2 - Jpot4)))

            for iAtom in range(nAtoms):
                print(Jpot3[iAtom], Jpot4[iAtom])

            # check Vln
            print("V0ln-VlnMat: ", np.max(np.abs(V0ln-VlnMat)))
            # print(V0ln-VlnMat)

            # check RlnMat
            print("RlnMat: Rij + Rji: ", np.max(np.abs(RlnMat.transpose([1,0,2]) + RlnMat)))

            # check FlnMat
            print("FlnMat: ", np.max(np.abs(FlnMat2.transpose([1,0,2]) - FlnMat)))

            # check dUi/dRj * vj vs. dUj/dRi * vi
            print("dUi/dRj * vj", np.max(np.abs(np.sum(FlnMat2 * V0[:,None,:], axis=2) - np.sum(FlnMat * V0[None, :, :],axis=2).T)))

            print(np.sum(Jpot3), np.sum(Jpot4))

            return Jpot2

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        # MD equilibrium loop
        # for iStep in range(1000):
        #     R0 = R1
        #     Vneg = Vpos
        #     R0, Ep, Fp, Fpq = getEF(R0)
        #     R1, Vpos, V0 = MDstep(R0, Vneg, 0.001, Fp)
        #     # printing the output
        #     if (iStep % 10 == 0) or \
        #             ((iStep % 10 != 0) & (iStep == params["epoch"] - 1)):
        #         printXYZ(iStep, R0, V0, Fp, Ep)

        # Thermal conductivity MD loop
        Jt0 = 0
        JxOut = open("Jx", 'w')
        JyOut = open("Jy", 'w')
        JzOut = open("Jz", 'w')

        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Fpq = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):

                # only calculate <J(t)J(0)> when printing
                Ek = np.sum(0.5 * mSi * V0 ** 2 * constA, axis=1)
                J0 = (Ep+Ek[:,None])*V0
                J1 = getJ(R0, V0, Fpq)
                Jt = J0 + J1
                if iStep == 0:
                    Jt0 = Jt

                print("<Jx(t)Jx(0)>", np.sum(Jt0*Jt, axis=0)[0])
                # printXYZ(iStep, R0, V0, Fp, Ep, np.sum(Jt0*Jt, axis=0)[0])
                # JxOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 0]]) + "\n")
                # JyOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 1]]) + "\n")
                # JzOut.write(str(iStep) + " " + " ".join([str(x) for x in Jt[:, 2]]) + "\n")



# Thermal conductivity (MD method)
# This is a back up of the working method
# need to modify this slightly
def old_specialTask08(params):

    from scipy.special import erfc
    def m(x):
        return erfc(12*np.abs(x-0.5)-3)/2

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
    tfFln = tff.tf_getFln(tfCoord, tfLattice, params) / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}
            Es, Ep, Fp = sess.run(((tfEs-0.5)/tfEngyA, tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            return R.dot(lattice.T), Ep, Fp, Es

        def getJhalf(R0, V0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}

            Fln = sess.run(-tfFln, feed_dict=feedDict)

            idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
            adjMat, Fln = npf.adjList2adjMat(idxNb, Fln)
            _,_,Fln = npf.adjMat2adjList(adjMat, Fln.transpose([1, 0, 2]))
            Rln = -Rln

            Jhalf = np.sum(Rln * np.sum(Fln * V0[:, None, :], axis=2)[:, :, None], axis=1)

            return Jhalf

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        # MD equilibrium loop
        for iStep in range(1000):
            R0 = R1
            Vneg = Vpos
            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, 0.001, Fp)
            # printing the output
            if (iStep % 10 == 0) or \
                    ((iStep % 10 != 0) & (iStep == params["epoch"] - 1)):
                printXYZ(iStep, R0, V0, Fp, Ep)

        # Thermal conductivity MD loop
        Jt0 = 0
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):

                # only calculate <J(t)J(0)> when printing
                Rhalf = R0 + m(R0[:, 0] / lattice[0, 0])[:, None] * Vpos * dt
                J0 = Ep*V0
                J1 = getJhalf(R0, V0)
                Jt = J0 + J1
                # J0 = Es * V0 # shifting to zero
                # J1 = getJhalf(R0, m(R0[:, 0] / lattice[0, 0])[:, None] * V0)
                # J2 = getJhalf(Rhalf, (1 - m(R0[:, 0] / lattice[0, 0])[:, None]) * V0)  # What velocity?
                # Jt = J0 + J1 + J2
                if iStep == 0:
                    Jt0 = Jt

                printXYZ(iStep, R0, V0, Fp, Ep, np.sum(Jt0*Jt, axis=0)[0])


# Thermal conductivity (Kang, Jun):
def old2_specialTask09(params):
    from scipy.special import erfc
    def m(x):
        return erfc(12*np.abs(x-0.5)-3)/2

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
    tfFln = tff.tf_getFln(tfCoord, tfLattice, params) / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}
            Es, Ep, Fp = sess.run(((tfEs-0.5)/tfEngyA, tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            return R.dot(lattice.T), Ep, Fp, Es

        def getJhalf(Rhalf, Ein, dR, region=1):
            lattice2 = lattice.copy()
            lattice2[0,0] = lattice[0,0]*2
            R = np.linalg.solve(lattice2, Rhalf.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice2}
            E0, F0 = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            R = np.linalg.solve(lattice2, (Rhalf+dR).T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice2}
            E1, F1 = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            dE1 = (E1 - E0)[:,0]
            dF1 = (F0 + F1)/2

            dEk = np.sum(dF1 * dR, axis=1)

            if region == 1:
                R = R - [0.5, 0, 0]
            elif region == 2:
                R[R[:,0]>0.5] = R[R[:,0]>0.5] - [1,0,0]
            else:
                print("ERROR! Please choose either region 1 or 2")

            Rshifted = R.dot(lattice2.T)
            print("Region", region, "Rshifted Max & Min: ", np.max(Rshifted[:,0]), np.min(Rshifted[:,0]))

            Jhalf = Rshifted[:,0] * (dE1 + dEk)/dt

            R = np.linalg.solve(lattice2, (Rhalf+0.5*dR).T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice2}
            Fl, Fln = sess.run((tfFp, tfFln), feed_dict=feedDict)
            idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice2, float(params['dcut']))
            adjMat, FlnMat = npf.adjList2adjMat(idxNb, Fln)
            _, _, Fln = npf.adjMat2adjList(adjMat, FlnMat.transpose([1, 0, 2]))

            dRmat = np.zeros((nAtoms, maxNb, 3))
            dRmat[idxNb>0] = dR[idxNb[idxNb>0]-1]

            dE2 = np.sum(Fl * dR + np.sum(Fln * dRmat,axis=1),axis=1)

            dE3 = E0 - Ein

            return Jhalf, dE1, dE2, dE3, E1

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        E0 = 0

        # Thermal conductivity MD loop
        # Jt0 = 0
        # J00 = 0
        # J10 = 0
        # J20 = 0
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)
            Ek = np.sum(0.5 * mSi * V0 ** 2 * constA, axis=1)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):

                J0 = (Ep[:, 0] + Ek) * V0[:, 0]
                J0 = np.concatenate([J0, J0])

                R0new = np.concatenate([R0, R0 + lattice[0]], axis=0)
                Epnew  = np.concatenate([Ep, Ep], axis=0)
                VposNew = np.concatenate([Vpos, Vpos], axis=0)

                J1, dE1, dE2, dE3, Eout = getJhalf(R0new, Epnew, m(R0new[:, 0] / (2*lattice[0, 0]))[:, None] * VposNew * dt,1)
                # Rhalf = R0 + m(R0[:, 0] / lattice[0, 0])[:, None] * Vpos * dt
                Rhalf = R0new + m(R0new[:, 0] / (2*lattice[0, 0]))[:, None] * VposNew * dt
                J2, dE1, dE2, dE3, Eout = getJhalf(Rhalf, Eout, np.concatenate([R1, R1 + lattice[0]], axis=0)-Rhalf,2)

                print(iStep)
                for iAtom in range(len(dE1)):
                    print(iAtom, dE1[iAtom], dE2[iAtom], (dE1-dE2)[iAtom], dE3[iAtom])

                Jt = J0 + J1 + J2
                if iStep == 0:
                    Jt0 = Jt

                print("iStep: ", iStep, np.sum(Jt0*Jt, axis=0), np.mean(Jt))


# Thermal conductivity (Kang, Jun):
def old_specialTask09(params):
    from scipy.special import erfc
    def m(x):
        return erfc(12*np.abs(x-0.5)-3)/2

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
    tfFln = tff.tf_getFln(tfCoord, tfLattice, params) / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

    with tf.Session() as sess:

        # initialize Tensorflow flow
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")

        # define the function for Ep and Fp
        def getEF(R0):
            R = np.linalg.solve(lattice, R0.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])

            feedDict = {tfCoord: R, tfLattice: lattice}
            Es, Ep, Fp = sess.run(((tfEs-0.5)/tfEngyA, tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp

            return R.dot(lattice.T), Ep, Fp, Es

        def getJhalf(Rhalf, Ein, dR, region=1):
            R = np.linalg.solve(lattice, Rhalf.T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice}
            E0, F0 = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            R = np.linalg.solve(lattice, (Rhalf+dR).T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice}
            E1, F1 = sess.run((tfEp, -tfFp), feed_dict=feedDict)

            dE1 = (E1 - E0)[:,0]
            dF1 = (F0 + F1)/2

            dEk = np.sum(dF1 * dR, axis=1)

            if region == 1:
                R = R - [0.5, 0, 0]
            elif region == 2:
                R[R[:,0]>0.5] = R[R[:,0]>0.5] - [1,0,0]
            else:
                print("ERROR! Please choose either region 1 or 2")

            Rshifted = R.dot(lattice.T)
            print("Region", region, "Rshifted Max & Min: ", np.max(Rshifted[:,0]), np.min(Rshifted[:,0]))

            Jhalf = Rshifted[:,0] * (dE1 + dEk)/dt

            R = np.linalg.solve(lattice, (Rhalf+0.5*dR).T).T
            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            feedDict = {tfCoord: R, tfLattice: lattice}
            Fl, Fln = sess.run((tfFp, tfFln), feed_dict=feedDict)
            idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
            adjMat, FlnMat = npf.adjList2adjMat(idxNb, Fln)
            _, _, Fln = npf.adjMat2adjList(adjMat, FlnMat.transpose([1, 0, 2]))

            dRmat = np.zeros((nAtoms, maxNb, 3))
            dRmat[idxNb>0] = dR[idxNb[idxNb>0]-1]

            dE2 = np.sum(Fl * dR + np.sum(Fln * dRmat,axis=1),axis=1)

            dE3 = E0 - Ein

            return Jhalf, dE1, dE2, dE3, E1

            # idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
            # Vij = np.zeros((nAtoms, maxNb, 3))
            # Vij[idxNb>0] = Vhalf[idxNb[idxNb>0]-1]
            # Fln = sess.run(tfFln, feed_dict=feedDict)
            # adjMat, FlnMat = npf.adjList2adjMat(idxNb, Fln)
            # _, _, Fln = npf.adjMat2adjList(adjMat, FlnMat.transpose([1,0,2]))
            # Ehalf, Fi = sess.run((tfEp, tfFp), feed_dict=feedDict)
            #
            # dE_anal = np.sum(Vij * Fln, axis=2).sum(axis=1) + np.sum(Fi * Vhalf, axis=1)
            # print("Analtical dEi/dt and deltaEi")
            # print(dE_anal[:10]*dt)
            # print(dE_anal[:10])

            # Ehalf, Fhalf = sess.run((tfEp, tfFp), feed_dict=feedDict)

            # _, Ehalf, Fhalf, __ = getEF(Rhalf)
            #
            # dEdt = (Ehalf - E0 + np.sum(Fhalf*dR, axis=1)[:,None])/dt

            # print("Numerical dEi/dt and deltaEi")
            # print((Ehalf-E0)[:10,0])
            # print(dEdt[:10,0])

            # Jhalf = Rhalf[:,0] * dEdt[:,0]
            # Jhalf = Rhalf[:,0] * dE_anal

            # return Jhalf, Ehalf

        # initialize the atomic positions and velocities
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, iIter, lattice, R, F0, V0 = pyf.getData11(mmtFile)
            V0 = V0 * 1000 * bohr
            R1 = R.dot(lattice.T)
            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            Vpos = V0 - 0.5*Fp/mSi*dt / constA

        E0 = 0

        # # MD equilibrium loop
        # for iStep in range(1000):
        #     R0 = R1
        #     Vneg = Vpos
        #     E0 = Ep
        #     R0, Ep, Fp, Es = getEF(R0)
        #     R1, Vpos, V0 = MDstep(R0, Vneg, 0.0001, Fp)
        #
        #     R = np.linalg.solve(lattice, R0.T).T
        #     R[R > 1] = R[R > 1] - np.floor(R[R > 1])
        #     R[R < 0] = R[R < 0] - np.floor(R[R < 0])
        #     feedDict = {tfCoord: R, tfLattice: lattice}
        #     idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
        #     Fln = sess.run(-tfFln, feed_dict=feedDict)
        #     adjMat, FlnMat = npf.adjList2adjMat(idxNb, Fln)
        #     _, _, Fln = npf.adjMat2adjList(adjMat, FlnMat.transpose([1,0,2]))
        #
        #     dR = np.zeros((nAtoms, maxNb, 3))
        #     dR[idxNb>0] = (R1-R0)[idxNb[idxNb>0]-1]
        #
        #     dE = -np.sum(Fp * (R1-R0),axis=1) - np.sum(Fln * dR,axis=2).sum(axis=1)
        #
        #     # printing the output
        #     if (iStep % 10 == 0) or \
        #             ((iStep % 10 != 0) & (iStep == params["epoch"] - 1)):
        #         # printXYZ(iStep, R0, V0, Fp, Ep)
        #         print(iStep)
        #         for iAtom in range(nAtoms):
        #             print(Ep[iAtom]-E0[iAtom], dE[iAtom])

        # Thermal conductivity MD loop
        # Jt0 = 0
        # J00 = 0
        # J10 = 0
        # J20 = 0
        for iStep in range(params["epoch"]):
            R0 = R1
            Vneg = Vpos

            R0, Ep, Fp, Es = getEF(R0)
            R1, Vpos, V0 = MDstep(R0, Vneg, dt, Fp)
            Ek = np.sum(0.5 * mSi * V0 ** 2 * constA, axis=1)

            # printing the output
            if (iStep % int(params["nstep"]) == 0) or \
                    ((iStep % int(params["nstep"]) != 0) & (iStep == params["epoch"] - 1)):

                J0 = (Ep[:, 0] + Ek) * V0[:, 0]
                J1, dE1, dE2, dE3, Eout = getJhalf(R0, Ep, m(R0[:, 0] / lattice[0, 0])[:, None] * Vpos * dt,1)
                Rhalf = R0 + m(R0[:, 0] / lattice[0, 0])[:, None] * Vpos * dt
                J2, dE1, dE2, dE3, Eout = getJhalf(Rhalf, Eout, R1-Rhalf,2)

                print(iStep)
                for iAtom in range(nAtoms):
                    print(iAtom, dE1[iAtom], dE2[iAtom], (dE1-dE2)[iAtom], dE3[iAtom])

                # only calculate <J(t)J(0)> when printing
                # Rhalf = R0 + m(R0[:, 0] / lattice[0, 0])[:, None] * Vpos * dt
                # Rhalf[:,0] = Rhalf[:,0] - 0.5*lattice[0,0]
                # R1new = R1.copy()
                # R1new[R1[:,0]>lattice[0,0]/2] = R1new[R1[:,0]>lattice[0,0]/2] - lattice[0]
                # print("Rhalf range", np.min(Rhalf[:, 0]), np.max(Rhalf[:, 0]))
                # print("R1new range", np.min(R1new[:,0]), np.max(R1new[:,0]))
                # J0 = (Ep[:,0]+Ek)*V0[:,0]
                # J1, E1 = getJhalf(Rhalf, Ep, dt * Vpos*m(R0[:, 0] / lattice[0, 0])[:, None])
                # J2, E2 = getJhalf(R1new, E1, dt * Vpos*(1-m(R0[:, 0] / lattice[0, 0])[:, None]))
                Jt = J0 + J1 + J2
                if iStep == 0:
                    Jt0 = Jt

                print("iStep: ", iStep, np.sum(Jt0*Jt, axis=0), np.mean(Jt))
                    # J00 = J0
                    # J10 = J1
                    # J20 = J2

                # printXYZ(iStep, R0, V0, Fp, Ep, np.sum(Jt0*Jt, axis=0)
                #          np.sum(J00*J0, axis=0), np.sum(J10*J1, axis=0), np.sum(J20*J2, axis=0))


def old_specialTask08(params):

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
        idxNb, Rln, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params['dcut']))
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
