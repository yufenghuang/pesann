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
